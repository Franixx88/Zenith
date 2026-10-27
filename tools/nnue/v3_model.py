"""ZNNUE v3: 512 FT, 16 mirrored banks, pairwise, 8 integer 512-16-32-1 heads.

Training uses integer-scaled fake quantization with straight-through gradients.
The FT factorizer is folded into each bank on export. The native network never
needs the factorizer, floating point, or a separate model for each game phase.
"""
import math
import struct
import numpy as np
import torch
from torch import nn

INPUTS, HIDDEN, BANKS, OUTPUTS, L1, L2 = 768, 512, 16, 8, 16, 32
QA, QB, CP = 255, 64, 400
HEADER = 128
HEAD_BYTES = (HIDDEN * L1 + L1 * L2 + L2) * 2 + (L1 + L2 + 1) * 4
PAYLOAD = HIDDEN * 2 + BANKS * INPUTS * HIDDEN * 2 + OUTPUTS * HEAD_BYTES
FILE_BYTES = HEADER + PAYLOAD


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        magnitude=x.abs()
        base=torch.floor(magnitude)
        return x.sign() * (base + ((magnitude-base) >= 0.5).to(x.dtype))

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class _Trunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.trunc(x)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def ste_round(x):
    return _Round.apply(x)


def ste_trunc(x):
    return _Trunc.apply(x)


def canonical(x):
    king = x[:, 320:384].argmax(dim=1)
    mirror = (king % 8) >= 4
    permutation = torch.arange(INPUTS, device=x.device) ^ 7
    result = torch.where(mirror[:, None], x.index_select(1, permutation), x)
    king = king ^ (mirror.long() * 7)
    bucket = (king // 16) * 4 + king % 8
    return result, bucket


def material_bucket(x):
    return ((x.sum(dim=1).long() - 2).clamp(min=0) // 4).clamp(max=OUTPUTS - 1)


def feature_transform(x, weight, bias):
    x, buckets = canonical(x)
    result = x.new_zeros((len(x), HIDDEN))
    for bucket in range(BANKS):
        positions = (buckets == bucket).nonzero(as_tuple=True)[0]
        if positions.numel():
            values = nn.functional.linear(x.index_select(0, positions),
                weight[:, bucket * INPUTS:(bucket + 1) * INPUTS], bias)
            result.index_copy_(0, positions, values)
    return result.clamp(0, QA)


class Net(nn.Module):
    architecture = "v3"
    king_buckets = BANKS
    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(INPUTS * BANKS, HIDDEN)
        self.common = nn.Parameter(torch.empty(HIDDEN, INPUTS))
        nn.init.uniform_(self.common, -1 / math.sqrt(INPUTS), 1 / math.sqrt(INPUTS))
        nn.init.zeros_(self.ft.weight)
        nn.init.constant_(self.ft.bias, 0.1)
        self.l1_w = nn.Parameter(torch.empty(OUTPUTS, L1, HIDDEN))
        self.l1_b = nn.Parameter(torch.full((OUTPUTS, L1), 0.2))
        self.l2_w = nn.Parameter(torch.empty(OUTPUTS, L2, L1))
        self.l2_b = nn.Parameter(torch.full((OUTPUTS, L2), 0.2))
        self.out_w = nn.Parameter(torch.empty(OUTPUTS, 1, L2))
        self.out_b = nn.Parameter(torch.zeros(OUTPUTS, 1))
        nn.init.uniform_(self.l1_w, -1 / math.sqrt(HIDDEN), 1 / math.sqrt(HIDDEN))
        nn.init.uniform_(self.l2_w, -1 / math.sqrt(L1), 1 / math.sqrt(L1))
        nn.init.uniform_(self.out_w, -1 / math.sqrt(L2), 1 / math.sqrt(L2))

    def effective_ft(self):
        return (self.ft.weight.reshape(HIDDEN, BANKS, INPUTS) + self.common[:, None, :]).reshape(HIDDEN, BANKS * INPUTS)

    def forward_cp(self, x, xo):
        n = len(x)
        weight = ste_round(self.effective_ft() * QA)
        bias = ste_round(self.ft.bias * QA)
        a = feature_transform(torch.cat((x, xo), dim=0), weight, bias)
        pair = ste_trunc(a[:, :HIDDEN // 2] * a[:, HIDDEN // 2:] / QA)
        features = torch.cat((pair[:n], pair[n:]), dim=1)
        buckets = material_bucket(x)
        result = x.new_zeros(n)
        for bucket in range(OUTPUTS):
            positions = (buckets == bucket).nonzero(as_tuple=True)[0]
            if positions.numel():
                h1 = nn.functional.linear(features.index_select(0, positions),
                    ste_round(self.l1_w[bucket] * QB), ste_round(self.l1_b[bucket] * QA * QB))
                h1 = ste_trunc(h1 / QB).clamp(0, QA)
                h2 = nn.functional.linear(h1, ste_round(self.l2_w[bucket] * QB),
                                          ste_round(self.l2_b[bucket] * QA * QB))
                h2 = ste_trunc(h2 / QB).clamp(0, QA)
                raw = nn.functional.linear(h2, ste_round(self.out_w[bucket] * QB),
                                           ste_round(self.out_b[bucket] * QA * QB)).squeeze(1)
                # 400/(255*64)=5/204: raw*5 is exactly representable in f32.
                cp = ste_trunc(raw * 5 / 204).clamp(-28000, 28000)
                result.index_copy_(0, positions, cp)
        return result

    def forward(self, x, xo):
        return self.forward_cp(x, xo) / CP

    @torch.no_grad()
    def constrain(self):
        self.common.clamp_(-2, 2)
        effective = self.effective_ft().clamp(-2, 2).reshape(HIDDEN, BANKS, INPUTS)
        self.ft.weight.copy_((effective - self.common[:, None, :]).reshape(HIDDEN, BANKS * INPUTS))
        self.ft.bias.clamp_(-16, 16)
        for weight in [self.l1_w, self.l2_w, self.out_w]: weight.clamp_(-127 / QB, 127 / QB)
        for bias in [self.l1_b, self.l2_b, self.out_b]: bias.clamp_(-4, 4)


def _q(value, scale, bound, dtype):
    # Exactly the same fp32 scaling and ties-away rounding as the QAT forward.
    rounded = ste_round(value.detach() * scale).cpu().numpy()
    if not np.isfinite(rounded).all() or np.abs(rounded).max() > bound:
        raise ValueError("NNUE v3 quantization bounds")
    return rounded.astype(dtype)


def quantize(model):
    return dict(version=3, ft_b=_q(model.ft.bias, QA, 4096, np.int16),
        ft_w=np.ascontiguousarray(_q(model.effective_ft(), QA, 512, np.int16).T),
        l1_w=_q(model.l1_w, QB, 127, np.int16), l1_b=_q(model.l1_b, QA * QB, 65536, np.int32),
        l2_w=_q(model.l2_w, QB, 127, np.int16), l2_b=_q(model.l2_b, QA * QB, 65536, np.int32),
        out_w=_q(model.out_w, QB, 127, np.int16), out_b=_q(model.out_b, QA * QB, 65536, np.int32))


def checksum(data):
    h = 0xCBF29CE484222325
    for i, value in enumerate(data):
        h = ((h ^ (0 if 48 <= i < 56 else value)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def to_bytes(q):
    head = bytearray(HEADER)
    head[:8] = b"ZNNUE\0\0\0"
    struct.pack_into("<8I", head, 8, 3, INPUTS, HIDDEN, BANKS, QA, QB, CP, OUTPUTS)
    struct.pack_into("<Q", head, 40, PAYLOAD)
    struct.pack_into("<Q", head, 56, 3)
    struct.pack_into("<4I", head, 64, L1, L2, HEADER, 1)
    head += q['ft_b'].astype('<i2').tobytes()
    head += q['ft_w'].astype('<i2').tobytes()
    for bucket in range(OUTPUTS):
        for name, dtype in [('l1_w','<i2'),('l1_b','<i4'),('l2_w','<i2'),('l2_b','<i4'),('out_w','<i2'),('out_b','<i4')]:
            head += q[name][bucket].astype(dtype).tobytes()
    if len(head) != FILE_BYTES: raise ValueError("NNUE v3 tensor dimensions")
    struct.pack_into("<Q", head, 48, checksum(head))
    return bytes(head)


def read_net(path):
    with open(path, 'rb') as f: data = f.read(FILE_BYTES + 1)
    if len(data) != FILE_BYTES or data[:8] != b"ZNNUE\0\0\0": raise ValueError("NNUE v3 file size/magic")
    if (struct.unpack_from('<8I',data,8) != (3,INPUTS,HIDDEN,BANKS,QA,QB,CP,OUTPUTS)
        or struct.unpack_from('<Q',data,40)[0] != PAYLOAD or struct.unpack_from('<Q',data,56)[0] != 3
        or struct.unpack_from('<4I',data,64) != (L1,L2,HEADER,1) or any(data[80:128])):
        raise ValueError("NNUE v3 unsupported header")
    if struct.unpack_from('<Q',data,48)[0] != checksum(data): raise ValueError("NNUE v3 checksum")
    offset = HEADER
    def read(shape,dtype):
        nonlocal offset
        count=math.prod(shape); value=np.frombuffer(data,dtype=dtype,count=count,offset=offset).reshape(shape).astype(np.int64)
        offset+=count*np.dtype(dtype).itemsize
        return value
    q=dict(version=3, ft_b=read((HIDDEN,),'<i2'),ft_w=read((BANKS*INPUTS,HIDDEN),'<i2'))
    names=[('l1_w',(L1,HIDDEN),'<i2'),('l1_b',(L1,),'<i4'),('l2_w',(L2,L1),'<i2'),('l2_b',(L2,),'<i4'),('out_w',(1,L2),'<i2'),('out_b',(1,),'<i4')]
    parts={name:[] for name,_,_ in names}
    for _ in range(OUTPUTS):
        for name,shape,dtype in names: parts[name].append(read(shape,dtype))
    q.update({name:np.stack(values) for name,values in parts.items()})
    for name,value in q.items():
        if name=='version': continue
        bound=4096 if name=='ft_b' else 512 if name=='ft_w' else 65536 if name.endswith('_b') else 127
        if np.abs(value).max()>bound: raise ValueError(f'NNUE v3 {name} bounds')
    return q


@torch.no_grad()
def quantized_cp(q,x,xo):
    dev=x.device
    tensor=lambda name:torch.as_tensor(q[name],device=dev,dtype=torch.float64)
    n=len(x); a=feature_transform(torch.cat((x,xo),0).double(),tensor('ft_w').T,tensor('ft_b'))
    pair=torch.trunc(a[:,:HIDDEN//2]*a[:,HIDDEN//2:]/QA)
    features=torch.cat((pair[:n],pair[n:]),1); buckets=material_bucket(x)
    raw=torch.zeros(n,device=dev,dtype=torch.float64)
    w1,b1,w2,b2,wo,bo=[tensor(name) for name in ['l1_w','l1_b','l2_w','l2_b','out_w','out_b']]
    for bucket in range(OUTPUTS):
        positions=(buckets==bucket).nonzero(as_tuple=True)[0]
        if positions.numel():
            h1=torch.trunc(nn.functional.linear(features.index_select(0,positions),w1[bucket],b1[bucket])/QB).clamp(0,QA)
            h2=torch.trunc(nn.functional.linear(h1,w2[bucket],b2[bucket])/QB).clamp(0,QA)
            raw.index_copy_(0,positions,nn.functional.linear(h2,wo[bucket],bo[bucket]).squeeze(1))
    return raw,torch.trunc(raw*CP/(QA*QB)).clamp(-28000,28000)


def spec_forward(q,board):
    """Independent NumPy integer implementation, using chess piece lists."""
    import chess
    halves=[]
    for pov in (board.turn,not board.turn):
        king=board.king(pov) ^ (56 if pov==chess.BLACK else 0)
        flip=7 if king%8>=4 else 0; k=king^flip; bucket=(k//16)*4+k%8
        a=q['ft_b'].astype(np.int64).copy()
        for square,piece in board.piece_map().items():
            relative=square^(56 if pov==chess.BLACK else 0)^flip
            feature=bucket*INPUTS+((0 if piece.color==pov else 1)*6+piece.piece_type-1)*64+relative
            a+=q['ft_w'][feature]
        a=np.clip(a,0,QA); halves.append(a[:256]*a[256:]//QA)
    features=np.concatenate(halves); bucket=min(7,max(0,(len(board.piece_map())-2)//4))
    # Negative hidden preactivations clip to zero, so floor/trunc agree after clamp.
    h1=np.clip((q['l1_w'][bucket]@features+q['l1_b'][bucket])//QB,0,QA)
    h2=np.clip((q['l2_w'][bucket]@h1+q['l2_b'][bucket])//QB,0,QA)
    raw=int((q['out_w'][bucket]@h2+q['out_b'][bucket])[0])
    cp=(abs(raw)*CP//(QA*QB))*(1 if raw>=0 else -1)
    return raw,max(-28000,min(28000,cp))
