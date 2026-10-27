"""V3 integer QAT, mirrored banks, material heads and native export parity."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import chess
import numpy as np
import torch
import v3_model as V
import train_nnue as T
import check_parity as P
from records import RECORD

ROOT = Path(__file__).resolve().parents[2]

def records(boards):
    rows = np.zeros(len(boards), dtype=RECORD)
    for row, board in zip(rows,boards):
        bits = bytearray()
        for color in [board.turn,not board.turn]:
            for piece in range(1,7):
                mask=sum(1<<(s^(56 if not board.turn else 0)) for s in board.pieces(piece,color))
                bits.extend(mask.to_bytes(8,'little'))
        row['bits']=np.frombuffer(bits,dtype=np.uint8)
    return rows

def positions():
    result=[]
    # All 16 banks, both mirror contexts, all 8 output heads.
    for rank in range(8):
        for file in range(8):
            board=chess.Board.empty()
            board.set_piece_at(chess.square(file,rank),chess.Piece(chess.KING,chess.WHITE))
            board.set_piece_at(chess.square(7 if file<4 else 0,7 if rank<4 else 0),chess.Piece(chess.KING,chess.BLACK))
            for s in range(64):
                if len(board.piece_map()) >= 2+4*((rank*8+file)%8): break
                color=bool(len(board.piece_map())%2)
                if color and chess.BB_KNIGHT_ATTACKS[s] & board.kings & board.occupied_co[chess.BLACK]: continue
                if board.piece_at(s) is None: board.set_piece_at(s,chess.Piece(chess.KNIGHT,color))
            # Synthetic placements need not be legal games; each has exactly two kings.
            result.extend([board,board.mirror(),board.transform(chess.flip_horizontal)])
    rng=random.Random(71); board=chess.Board()
    for _ in range(128):
        if board.is_game_over(): board=chess.Board()
        board.push(rng.choice(list(board.legal_moves))); result.append(board.copy())
    return result

class V3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(41)

    def test_qat_matches_integer_oracles_and_mirrors_and_native_backends(self):
        boards=positions(); rows=records(boards)
        x,xo=T.unpack(torch.from_numpy(rows['bits'].copy()),torch.device('cpu'),torch.as_tensor(T.opponent_inverse_permutation()))
        model=V.Net()
        # Distinct banks/heads and both signs, independent of initialization luck.
        with torch.no_grad():
            for b in range(16): model.ft.weight[:,b*768:(b+1)*768].add_(b/512)
            model.out_b.copy_(torch.linspace(-0.7,0.7,8).reshape(8,1))
        q=T.quantize(model); raw,cp=T.quantized_cp(q,x,xo)
        torch.testing.assert_close(model.forward_cp(x,xo).double(),cp,atol=0,rtol=0)
        for i,b in enumerate(boards): self.assertEqual((int(raw[i]),int(cp[i])),V.spec_forward(q,b))
        for i in range(0,192,3): self.assertEqual(int(raw[i]),int(raw[i+2]))
        self.assertEqual(set(V.material_bucket(x).tolist()),set(range(8)))
        self.assertEqual(set(V.canonical(x)[1].tolist()),set(range(16)))
        with tempfile.TemporaryDirectory(dir=ROOT/'target') as tmp:
            base=Path(tmp); net=base/'v3 model.znnue';net.write_bytes(T.to_bytes(q))
            reloaded=P.read_net(net)
            self.assertEqual(net.stat().st_size,V.FILE_BYTES)
            np.testing.assert_array_equal(q['ft_w'],reloaded['ft_w'])
            for flags in (['--scalar'],[]):
                run=subprocess.run([str(ROOT/'target/release/examples/nnue.exe'),'verify',str(net),*flags],
                    input='\n'.join(b.fen() for b in boards)+'\n',capture_output=True,text=True,timeout=90)
                self.assertEqual(run.returncode,0,run.stderr)
                native=[tuple(map(int,line.rsplit('\t',2)[1:])) for line in run.stdout.splitlines()]
                self.assertEqual(native,[(int(a),int(b)) for a,b in zip(raw,cp)])

    def test_ste_is_exact_and_gradients_reach_only_selected_bank_and_head(self):
        z=torch.tensor([-2.5,-0.51,-0.1,0.1,0.51,2.5],requires_grad=True)
        torch.testing.assert_close(V.ste_round(z),torch.tensor([-3.,-1.,0.,0.,1.,3.]),atol=0,rtol=0)
        V.ste_round(z).sum().backward();torch.testing.assert_close(z.grad,torch.ones_like(z))
        model=V.Net();rows=records([chess.Board()])
        x,xo=T.unpack(torch.from_numpy(rows['bits'].copy()),torch.device('cpu'),torch.as_tensor(T.opponent_inverse_permutation()))
        model.forward_cp(x,xo).sum().backward()
        g=model.ft.weight.grad.reshape(512,16,768).abs().sum((0,2))
        selected=set(V.canonical(torch.cat([x,xo]))[1].tolist())
        self.assertGreater(model.common.grad.abs().sum(),0)
        for b in range(16): self.assertEqual(bool(g[b]>0),b in selected)
        self.assertGreater(model.out_w.grad[7].abs().sum(),0)
        self.assertEqual(model.out_w.grad[:7].abs().sum(),0)
        with self.assertRaises(ValueError): T.initialize_from_checkpoint(model,{'model':T.Net().state_dict()})

    def test_exact_cpu_resume_includes_factorizer_and_optimizer(self):
        boards=positions()[-64:];rows=records(boards);rows['score']=np.arange(len(rows))*8-250
        with tempfile.TemporaryDirectory(dir=ROOT/'target') as tmp:
            base=Path(tmp);data=base/'data'
            # Use the real preparer's manifest contract instead of hand-written metadata.
            prep=ROOT/'target/release/examples/prepare_nnue.exe'
            tsv=''.join(f'{b.fen()}\t24\t{i*8-250}\t\n' for i,b in enumerate(boards))
            # Prepared data rejects tactical positions by default; retain them for this fixture.
            subprocess.run([str(prep),'--out',str(data),'--shards','1','--val-mod','5','--keep-tactical',
                '--require-complete','--source','synthetic-v3-smoke-only'],input=tsv+f'#zenith-tsv-complete\t{len(boards)}\n',
                capture_output=True,text=True,check=True,timeout=90)
            common=[sys.executable,str(ROOT/'tools/nnue/train_nnue.py'),'--data',str(data),'--architecture','v3',
                '--epochs','2','--batch-size','16','--block-rows','64','--val-rows','16','--val-chunk','16','--threads','1','--warmup-steps','0']
            for tail in ([ '--out',str(base/'full') ],['--out',str(base/'split'),'--stop-after-epochs','1'],
                         ['--out',str(base/'split'),'--resume',str(base/'split/last.pt')]):
                subprocess.run([*common,*tail],capture_output=True,text=True,check=True,timeout=120)
            self.assertEqual((base/'full/last.znnue').read_bytes(),(base/'split/last.znnue').read_bytes())
            a=torch.load(base/'full/last.pt',weights_only=True);b=torch.load(base/'split/last.pt',weights_only=True)
            self.assertEqual(a['history'],b['history']);self.assertTrue(a['contract']['factorizer'])
            for name in a['model']: torch.testing.assert_close(a['model'][name],b['model'][name],atol=0,rtol=0)

if __name__=='__main__': unittest.main()
