//! V3: mirrored 16-bank 512 FT, integer pairwise activation, 8 x (512-16-32-1).
//! All layer accumulations fit exactly in f32 integers as well as i32. This is
//! useful for quantization-aware training; runtime inference remains integer.
use super::{
    Backend, FeatureChanges, INPUTS, MAX_FEATURE_BIAS, MAX_FEATURE_WEIGHT, PIECES, QA, QB,
    feature_index, kernels, score_from_raw, visit,
};
use crate::board::{Board, Color, Piece};

pub const HIDDEN: usize = 512;
pub const BANKS: usize = 16;
pub const OUTPUTS: usize = 8;
pub const L1: usize = 16;
pub const L2: usize = 32;
pub const HEAD_WEIGHT_BOUND: i16 = 127;
pub const HEAD_BIAS_BOUND: i32 = 65_536;
pub const HEADER: usize = 128;
const HEAD_BYTES: usize = (HIDDEN * L1 + L1 * L2 + L2) * 2 + (L1 + L2 + 1) * 4;
pub const FILE_BYTES: usize =
    HEADER + HIDDEN * 2 + BANKS * INPUTS * HIDDEN * 2 + OUTPUTS * HEAD_BYTES;

#[derive(Clone)]
pub struct Head {
    /// Output-major matrices. The selected head alone runs at inference.
    pub l1_weights: Vec<i16>,
    pub l1_bias: [i32; L1],
    pub l2_weights: Vec<i16>,
    pub l2_bias: [i32; L2],
    pub output_weights: [i16; L2],
    pub output_bias: i32,
}
impl Default for Head {
    fn default() -> Self {
        Self {
            l1_weights: vec![0; HIDDEN * L1],
            l1_bias: [0; L1],
            l2_weights: vec![0; L1 * L2],
            l2_bias: [0; L2],
            output_weights: [0; L2],
            output_bias: 0,
        }
    }
}
pub struct Parameters {
    pub feature_bias: [i16; HIDDEN],
    /// Bank-major, then feature-major, then neuron.
    pub feature_weights: Vec<i16>,
    pub heads: Vec<Head>,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            feature_bias: [0; HIDDEN],
            feature_weights: vec![0; BANKS * INPUTS * HIDDEN],
            heads: vec![Head::default(); OUTPUTS],
        }
    }
}
impl Parameters {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.feature_weights.len() != BANKS * INPUTS * HIDDEN || self.heads.len() != OUTPUTS {
            return Err("NNUE v3 feature/head dimensions".into());
        }
        if self
            .feature_weights
            .iter()
            .any(|v| v.unsigned_abs() > MAX_FEATURE_WEIGHT as u16)
            || self
                .feature_bias
                .iter()
                .any(|v| v.unsigned_abs() > MAX_FEATURE_BIAS as u16)
        {
            return Err("NNUE v3 FT parameter bounds".into());
        }
        for head in &self.heads {
            if head.l1_weights.len() != HIDDEN * L1 || head.l2_weights.len() != L1 * L2 {
                return Err("NNUE v3 head matrix dimensions".into());
            }
            if head
                .l1_weights
                .iter()
                .chain(&head.l2_weights)
                .chain(&head.output_weights)
                .any(|v| v.unsigned_abs() > HEAD_WEIGHT_BOUND as u16)
                || head
                    .l1_bias
                    .iter()
                    .chain(&head.l2_bias)
                    .chain(std::iter::once(&head.output_bias))
                    .any(|v| v.unsigned_abs() > HEAD_BIAS_BOUND as u32)
            {
                return Err("NNUE v3 head parameter bounds".into());
            }
        }
        Ok(())
    }

    pub(crate) fn accumulator(&self, board: &Board, backend: Backend) -> V3Accumulator<'_> {
        let mut a = V3Accumulator {
            parameters: self,
            backend,
            board: *board,
            sums: [self.feature_bias; 2],
            contexts: [0; 2],
            refreshes: 0,
        };
        for p in [Color::White, Color::Black] {
            a.refresh(board, p, context(board, p));
        }
        a
    }

    pub(super) fn raw_reference(&self, board: &Board) -> i64 {
        let mut sums = [self.feature_bias; 2];
        for pov in [Color::White, Color::Black] {
            let ctx = context(board, pov);
            for square in 0..64 {
                if let Some((color, piece)) = board.piece_at(square) {
                    let offset = index(ctx, pov, color, piece, square) * HIDDEN;
                    for (neuron, sum) in sums[pov.index()].iter_mut().enumerate() {
                        *sum += self.feature_weights[offset + neuron];
                    }
                }
            }
        }
        self.forward(&sums, board, Backend::Scalar)
    }

    fn forward(&self, sums: &[[i16; HIDDEN]; 2], board: &Board, backend: Backend) -> i64 {
        let mut input = [0i16; HIDDEN];
        for (half, pov) in [board.side_to_move, board.side_to_move.other()]
            .into_iter()
            .enumerate()
        {
            kernels::pair(
                backend,
                &sums[pov.index()],
                (&mut input[half * 256..(half + 1) * 256])
                    .try_into()
                    .unwrap(),
            );
        }
        let head = &self.heads[output_bucket(board.occupied().count_ones())];
        let mut a1 = [0i16; L1];
        let mut a2 = [0i16; L2];
        for (i, a) in a1.iter_mut().enumerate() {
            let raw = i64::from(head.l1_bias[i])
                + kernels::output(
                    backend,
                    &input,
                    &head.l1_weights[i * HIDDEN..(i + 1) * HIDDEN],
                );
            *a = (raw / i64::from(QB)).clamp(0, i64::from(QA)) as i16;
        }
        for (i, a) in a2.iter_mut().enumerate() {
            let raw = i64::from(head.l2_bias[i])
                + kernels::output(backend, &a1, &head.l2_weights[i * L1..(i + 1) * L1]);
            *a = (raw / i64::from(QB)).clamp(0, i64::from(QA)) as i16;
        }
        i64::from(head.output_bias) + kernels::output(backend, &a2, &head.output_weights)
    }

    pub(super) fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; HEADER];
        bytes[..8].copy_from_slice(b"ZNNUE\0\0\0");
        for (i, value) in [
            3,
            INPUTS as u32,
            HIDDEN as u32,
            BANKS as u32,
            QA as u32,
            QB as u32,
            400,
            OUTPUTS as u32,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[8 + i * 4..12 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[40..48].copy_from_slice(&((FILE_BYTES - HEADER) as u64).to_le_bytes());
        bytes[56..64].copy_from_slice(&3u64.to_le_bytes()); // mirror + pairwise flags
        for (i, value) in [L1 as u32, L2 as u32, HEADER as u32, 1]
            .into_iter()
            .enumerate()
        {
            bytes[64 + i * 4..68 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        let put16 = |bytes: &mut Vec<u8>, values: &[i16]| {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        };
        let put32 = |bytes: &mut Vec<u8>, values: &[i32]| {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        };
        put16(&mut bytes, &self.feature_bias);
        put16(&mut bytes, &self.feature_weights);
        for head in &self.heads {
            put16(&mut bytes, &head.l1_weights);
            put32(&mut bytes, &head.l1_bias);
            put16(&mut bytes, &head.l2_weights);
            put32(&mut bytes, &head.l2_bias);
            put16(&mut bytes, &head.output_weights);
            put32(&mut bytes, &[head.output_bias]);
        }
        debug_assert_eq!(bytes.len(), FILE_BYTES);
        let hash = super::format::checksum(&bytes);
        bytes[48..56].copy_from_slice(&hash.to_le_bytes());
        bytes
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != FILE_BYTES || &bytes[..8] != b"ZNNUE\0\0\0" {
            return Err(format!("NNUE v3 requires exactly {FILE_BYTES} bytes"));
        }
        let read32 = |o| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let read64 = |o| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let fields = [
            3,
            INPUTS as u32,
            HIDDEN as u32,
            BANKS as u32,
            QA as u32,
            QB as u32,
            400,
            OUTPUTS as u32,
        ];
        if fields
            .iter()
            .enumerate()
            .any(|(i, &v)| read32(8 + 4 * i) != v)
            || read64(40) != (FILE_BYTES - HEADER) as u64
            || read64(56) != 3
            || [L1 as u32, L2 as u32, HEADER as u32, 1]
                .iter()
                .enumerate()
                .any(|(i, &v)| read32(64 + 4 * i) != v)
            || bytes[80..HEADER].iter().any(|&v| v != 0)
        {
            return Err("NNUE v3 unsupported header".into());
        }
        if read64(48) != super::format::checksum(bytes) {
            return Err("NNUE v3 checksum mismatch".into());
        }
        let mut p = Self::default();
        let mut offset = HEADER;
        fn get16(bytes: &[u8], offset: &mut usize, target: &mut [i16]) {
            for v in target {
                *v = i16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap());
                *offset += 2;
            }
        }
        fn get32(bytes: &[u8], offset: &mut usize, target: &mut [i32]) {
            for v in target {
                *v = i32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
                *offset += 4;
            }
        }
        get16(bytes, &mut offset, &mut p.feature_bias);
        get16(bytes, &mut offset, &mut p.feature_weights);
        for h in &mut p.heads {
            get16(bytes, &mut offset, &mut h.l1_weights);
            get32(bytes, &mut offset, &mut h.l1_bias);
            get16(bytes, &mut offset, &mut h.l2_weights);
            get32(bytes, &mut offset, &mut h.l2_bias);
            get16(bytes, &mut offset, &mut h.output_weights);
            get32(bytes, &mut offset, std::slice::from_mut(&mut h.output_bias));
        }
        p.validate()?;
        Ok(p)
    }
}

pub fn context(board: &Board, pov: Color) -> usize {
    let king = board.bitboard(pov, Piece::King).trailing_zeros() as usize
        ^ if pov == Color::Black { 56 } else { 0 };
    let mirrored = king % 8 >= 4;
    let canonical = king ^ if mirrored { 7 } else { 0 };
    (canonical / 16) * 4 + canonical % 8 + BANKS * usize::from(mirrored)
}
pub fn output_bucket(pieces: u32) -> usize {
    ((pieces.saturating_sub(2) / 4) as usize).min(OUTPUTS - 1)
}
fn index(ctx: usize, pov: Color, color: Color, piece: Piece, square: u8) -> usize {
    (ctx % BANKS) * INPUTS
        + (feature_index(pov, color, piece, square) ^ if ctx >= BANKS { 7 } else { 0 })
}

#[derive(Clone)]
pub struct V3Accumulator<'net> {
    parameters: &'net Parameters,
    backend: Backend,
    board: Board,
    sums: [[i16; HIDDEN]; 2],
    contexts: [usize; 2],
    refreshes: u32,
}
impl V3Accumulator<'_> {
    pub fn board(&self) -> &Board {
        &self.board
    }
    pub fn lanes(&self, pov: Color) -> &[i16; HIDDEN] {
        &self.sums[pov.index()]
    }
    pub fn bucket_refreshes(&self) -> u32 {
        self.refreshes
    }
    pub fn raw_score(&self) -> i64 {
        self.parameters
            .forward(&self.sums, &self.board, self.backend)
    }
    pub fn evaluate(&self) -> i32 {
        score_from_raw(self.raw_score())
    }
    fn change(&mut self, pov: Color, color: Color, piece: Piece, square: u8, adding: bool) {
        let offset = index(self.contexts[pov.index()], pov, color, piece, square) * HIDDEN;
        kernels::update(
            self.backend,
            &mut self.sums[pov.index()],
            &self.parameters.feature_weights[offset..offset + HIDDEN],
            adding,
        );
    }
    fn refresh(&mut self, board: &Board, pov: Color, ctx: usize) {
        self.contexts[pov.index()] = ctx;
        self.sums[pov.index()] = self.parameters.feature_bias;
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                visit(board.bitboard(color, piece), |square| {
                    self.change(pov, color, piece, square, true)
                });
            }
        }
    }
    pub fn update(&mut self, next: &Board) -> FeatureChanges {
        let mut refreshed = [false; 2];
        self.refreshes = 0;
        for pov in [Color::White, Color::Black] {
            let ctx = context(next, pov);
            if ctx != self.contexts[pov.index()] {
                self.refresh(next, pov, ctx);
                refreshed[pov.index()] = true;
                self.refreshes += 1;
            }
        }
        let mut changes = FeatureChanges::default();
        for adding in [false, true] {
            for color in [Color::White, Color::Black] {
                for piece in PIECES {
                    let before = self.board.bitboard(color, piece);
                    let after = next.bitboard(color, piece);
                    let changed = if adding {
                        after & !before
                    } else {
                        before & !after
                    };
                    if adding {
                        changes.added += changed.count_ones();
                    } else {
                        changes.removed += changed.count_ones();
                    }
                    visit(changed, |square| {
                        for pov in [Color::White, Color::Black] {
                            if !refreshed[pov.index()] {
                                self.change(pov, color, piece, square, adding);
                            }
                        }
                    });
                }
            }
        }
        self.board = *next;
        changes
    }
}
