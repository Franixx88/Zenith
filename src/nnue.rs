//! Zenith NNUE: legacy 256-lane v1/v2 and mirrored, layered 512-lane v3.
//! Two perspectives, scalar reference and runtime AVX2 integer inference.
//! Search can load validated models; no trained network is bundled.
//! Without an EvalFile the playing evaluator remains classical.

use crate::board::{Board, Color, Piece};

mod format;
pub mod v3;
// Audited pointer intrinsics are confined to this private implementation module.
#[allow(unsafe_code)]
mod kernels;
pub use format::{FILE_BYTES, FORMAT_VERSION, FORMAT_VERSION_V2, V2_FILE_BYTES};

pub const INPUTS: usize = 768;
pub const HIDDEN: usize = 256;
pub const KING_BUCKETS: usize = 4;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const CP_SCALE: i32 = 400;
pub const MAX_SCORE: i32 = 28_000;
pub const MAX_FEATURE_WEIGHT: i16 = 512;
pub const MAX_FEATURE_BIAS: i16 = 4096;
pub const MAX_OUTPUT_WEIGHT: i16 = 8192;
pub const MAX_OUTPUT_BIAS: i32 = 16_777_216;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Avx2,
}
impl Backend {
    pub fn available(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                {
                    std::is_x86_feature_detected!("avx2")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            }
        }
    }
}
const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// Dataset adapter only: four-field analysis FENs have no move counters. Supply
/// neutral 0/1 placeholders, never infer a real fifty-move/repetition history.
/// The engine's strict six-field UCI/FEN parser is intentionally unchanged.
pub fn parse_training_fen(fen: &str) -> Result<Board, String> {
    let fields: Vec<_> = fen.split_whitespace().collect();
    match fields.len() {
        4 => Board::from_fen(&format!("{} 0 1", fields.join(" "))),
        6 => Board::from_fen(fen),
        _ => Err("NNUE training FEN must have four or six fields".into()),
    }
}

/// Sparse input index. A1=0; black perspective flips ranks (xor 56), not files.
/// Relative color 0=own, 1=enemy; piece order P,N,B,R,Q,K. Kings are inputs too.
pub fn feature_index(perspective: Color, color: Color, piece: Piece, square: u8) -> usize {
    assert!(square < 64, "NNUE square must be in 0..64");
    let relative = square ^ if perspective == Color::Black { 56 } else { 0 };
    (usize::from(color != perspective) * 6 + piece.index()) * 64 + relative as usize
}

/// V2 uses four 4x4 quadrants in the perspective's rank-relative coordinates.
/// Files are not mirrored: changing the activation/feature symmetry is a
/// separate experiment. Both colors use their own king to select their bank.
pub fn king_bucket(board: &Board, perspective: Color) -> usize {
    let square = board.bitboard(perspective, Piece::King).trailing_zeros() as usize
        ^ if perspective == Color::Black { 56 } else { 0 };
    usize::from(square % 8 >= 4) + 2 * usize::from(square / 8 >= 4)
}

/// Quantized, feature-major tensors. Each feature owns HIDDEN contiguous weights.
pub struct Parameters {
    pub feature_bias: [i16; HIDDEN],
    pub feature_weights: Vec<i16>,
    /// Side-to-move perspective first, opponent perspective second.
    pub output_weights: [i16; 2 * HIDDEN],
    pub output_bias: i32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            feature_bias: [0; HIDDEN],
            feature_weights: vec![0; INPUTS * HIDDEN],
            output_weights: [0; 2 * HIDDEN],
            output_bias: 0,
        }
    }
}

/// Floating point training/export contract: activations clip to [0,1], output
/// is in units of CP_SCALE centipawns. No floating point is used during inference.
pub struct FloatParameters {
    pub feature_bias: Vec<f32>,
    pub feature_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: f32,
}

impl FloatParameters {
    /// Round to nearest, ties away from zero. Reject NaN, infinities and values
    /// outside v1 bounds rather than silently clipping a trained model.
    pub fn quantize(&self) -> Result<Network, String> {
        if self.feature_bias.len() != HIDDEN
            || ![INPUTS * HIDDEN, KING_BUCKETS * INPUTS * HIDDEN]
                .contains(&self.feature_weights.len())
            || self.output_weights.len() != 2 * HIDDEN
        {
            return Err("NNUE floating tensor dimensions do not match v1".into());
        }
        let mut parameters = Parameters::default();
        for (dst, &src) in parameters.feature_bias.iter_mut().zip(&self.feature_bias) {
            *dst = quantize(src, QA, i32::from(MAX_FEATURE_BIAS))? as i16;
        }
        for (dst, &src) in parameters
            .feature_weights
            .iter_mut()
            .zip(&self.feature_weights)
        {
            *dst = quantize(src, QA, i32::from(MAX_FEATURE_WEIGHT))? as i16;
        }
        for (dst, &src) in parameters
            .output_weights
            .iter_mut()
            .zip(&self.output_weights)
        {
            *dst = quantize(src, QB, i32::from(MAX_OUTPUT_WEIGHT))? as i16;
        }
        parameters.output_bias = quantize(self.output_bias, QA * QB, MAX_OUTPUT_BIAS)?;
        Network::new(parameters)
    }
}

fn quantize(value: f32, scale: i32, bound: i32) -> Result<i32, String> {
    let scaled = (f64::from(value) * f64::from(scale)).round();
    if !scaled.is_finite() || scaled.abs() > f64::from(bound) {
        return Err("NNUE quantization: non-finite or out-of-range parameter".into());
    }
    Ok(scaled as i32)
}

/// Immutable validated model. Accumulators borrow it, preventing stale sums
/// from accidentally being evaluated with a different network's weights.
pub struct Network {
    parameters: Parameters,
    backend: Backend,
    king_buckets: usize,
    v3: Option<Box<v3::Runtime>>,
}

impl Network {
    pub fn new(parameters: Parameters) -> Result<Self, String> {
        let king_buckets = parameters.feature_weights.len() / (INPUTS * HIDDEN);
        if ![1, KING_BUCKETS].contains(&king_buckets)
            || parameters.feature_weights.len() != king_buckets * INPUTS * HIDDEN
        {
            return Err("NNUE feature tensor requires one or four king banks".into());
        }
        for (name, values, bound) in [
            (
                "feature bias",
                parameters.feature_bias.as_slice(),
                MAX_FEATURE_BIAS,
            ),
            (
                "feature weights",
                parameters.feature_weights.as_slice(),
                MAX_FEATURE_WEIGHT,
            ),
            (
                "output weights",
                parameters.output_weights.as_slice(),
                MAX_OUTPUT_WEIGHT,
            ),
        ] {
            if values
                .iter()
                .any(|value| value.unsigned_abs() > bound as u16)
            {
                return Err(format!("NNUE {name} exceeds v1 bounds"));
            }
        }
        if parameters.output_bias.unsigned_abs() > MAX_OUTPUT_BIAS as u32 {
            return Err("NNUE output bias exceeds v1 bounds".into());
        }
        Ok(Self {
            parameters,
            v3: None,
            king_buckets,
            backend: if Backend::Avx2.available() {
                Backend::Avx2
            } else {
                Backend::Scalar
            },
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }
    pub fn king_buckets(&self) -> usize {
        self.king_buckets
    }
    pub fn format_version(&self) -> u32 {
        if self.v3.is_some() {
            3
        } else if self.king_buckets == 1 {
            FORMAT_VERSION
        } else {
            FORMAT_VERSION_V2
        }
    }
    pub fn file_bytes(&self) -> usize {
        if self.v3.is_some() {
            v3::FILE_BYTES
        } else if self.king_buckets == 1 {
            FILE_BYTES
        } else {
            V2_FILE_BYTES
        }
    }
    fn bucket(&self, board: &Board, perspective: Color) -> usize {
        if self.king_buckets == 1 {
            0
        } else {
            king_bucket(board, perspective)
        }
    }
    pub fn with_backend(mut self, backend: Backend) -> Result<Self, String> {
        if !backend.available() {
            return Err(format!(
                "NNUE backend {backend:?} is not supported on this CPU/OS"
            ));
        }
        self.backend = backend;
        Ok(self)
    }

    pub fn parameters(&self) -> &Parameters {
        assert!(self.v3.is_none(), "use v3_parameters for a v3 model");
        &self.parameters
    }

    pub fn v3_parameters(&self) -> Option<&v3::Parameters> {
        self.v3.as_deref().map(|model| &model.parameters)
    }
    pub fn hidden_size(&self) -> usize {
        if self.v3.is_some() {
            v3::HIDDEN
        } else {
            HIDDEN
        }
    }
    pub fn new_v3(parameters: v3::Parameters) -> Result<Self, String> {
        parameters.validate()?;
        Ok(Self {
            parameters: Parameters {
                feature_bias: [0; HIDDEN],
                feature_weights: Vec::new(),
                output_weights: [0; HIDDEN * 2],
                output_bias: 0,
            },
            backend: if Backend::Avx2.available() {
                Backend::Avx2
            } else {
                Backend::Scalar
            },
            king_buckets: v3::BANKS,
            v3: Some(Box::new(v3::Runtime::new(parameters))),
        })
    }

    pub fn accumulator(&self, board: &Board) -> Accumulator<'_> {
        if let Some(p) = &self.v3 {
            Accumulator::V3(p.accumulator(board, self.backend))
        } else {
            Accumulator::Legacy(self.legacy_accumulator(board))
        }
    }

    pub(crate) fn legacy_accumulator(&self, board: &Board) -> LegacyAccumulator<'_> {
        assert!(self.v3.is_none());
        let mut accumulator = LegacyAccumulator {
            network: self,
            board: *board,
            sums: [self.parameters.feature_bias; 2],
            buckets: [0; 2],
            bucket_refreshes: 0,
        };
        for perspective in [Color::White, Color::Black] {
            accumulator.refresh_half(board, perspective, self.bucket(board, perspective));
        }
        accumulator
    }

    pub(crate) fn v3_accumulator(&self, board: &Board) -> v3::V3Accumulator<'_> {
        self.v3
            .as_ref()
            .expect("v3 model")
            .accumulator(board, self.backend)
    }

    /// Deliberately separate dense forward pass for cross-checking sparse and
    /// incremental calculations. Rebuilds inputs; never reads an accumulator.
    pub fn evaluate_reference(&self, board: &Board) -> i32 {
        score_from_raw(self.raw_reference(board))
    }

    pub fn raw_reference(&self, board: &Board) -> i64 {
        if let Some(p) = &self.v3 {
            return p.parameters.raw_reference(board);
        }
        let mut raw = i64::from(self.parameters.output_bias);
        for (half, perspective) in [board.side_to_move, board.side_to_move.other()]
            .into_iter()
            .enumerate()
        {
            let bank_offset = self.bucket(board, perspective) * INPUTS;
            let mut active = [false; INPUTS];
            for square in 0..64 {
                if let Some((color, piece)) = board.piece_at(square) {
                    active[feature_index(perspective, color, piece, square)] = true;
                }
            }
            for neuron in 0..HIDDEN {
                let mut sum = i64::from(self.parameters.feature_bias[neuron]);
                for (feature, &on) in active.iter().enumerate() {
                    if on {
                        sum += i64::from(
                            self.parameters.feature_weights
                                [(bank_offset + feature) * HIDDEN + neuron],
                        );
                    }
                }
                raw += sum.clamp(0, i64::from(QA))
                    * i64::from(self.parameters.output_weights[half * HIDDEN + neuron]);
            }
        }
        raw
    }
}

fn score_from_raw(raw: i64) -> i32 {
    // Signed integer division truncates towards zero. i64 also covers raw*400.
    (raw * i64::from(CP_SCALE) / i64::from(QA * QB))
        .clamp(-i64::from(MAX_SCORE), i64::from(MAX_SCORE)) as i32
}

/// Number of physical piece-square removals/additions (not doubled perspectives).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureChanges {
    pub removed: u32,
    pub added: u32,
}

/// Exact unclipped sums for both colors plus their board snapshot. Clone before
/// exploring a child, or update back to a parent board to undo. No heap storage.
#[derive(Clone)]
pub struct LegacyAccumulator<'net> {
    network: &'net Network,
    board: Board,
    sums: [[i16; HIDDEN]; 2],
    buckets: [usize; 2],
    bucket_refreshes: u32,
}

impl LegacyAccumulator<'_> {
    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn lanes(&self, perspective: Color) -> &[i16; HIDDEN] {
        &self.sums[perspective.index()]
    }

    /// Number of perspective halves rebuilt by the most recent update.
    /// Construction is counted separately by the search's full-frame counter.
    pub fn bucket_refreshes(&self) -> u32 {
        self.bucket_refreshes
    }

    /// Delta from piece bitboards: handles captures, EP, all promotions, king
    /// moves and both castlings without duplicating move rules. Remove before
    /// adding. V2 refreshes only the half whose king crossed a bank boundary.
    /// Also accepts FEN resets, undo, and null moves (zero feature changes).
    pub fn update(&mut self, next: &Board) -> FeatureChanges {
        let mut changes = FeatureChanges::default();
        self.bucket_refreshes = 0;
        let mut refreshed = [false; 2];
        for perspective in [Color::White, Color::Black] {
            let bucket = self.network.bucket(next, perspective);
            if bucket != self.buckets[perspective.index()] {
                self.refresh_half(next, perspective, bucket);
                refreshed[perspective.index()] = true;
                self.bucket_refreshes += 1;
            }
        }
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
                        for perspective in [Color::White, Color::Black] {
                            if !refreshed[perspective.index()] {
                                self.change_half(perspective, color, piece, square, adding);
                            }
                        }
                    });
                }
            }
        }
        self.board = *next;
        changes
    }

    fn refresh_half(&mut self, board: &Board, perspective: Color, bucket: usize) {
        self.buckets[perspective.index()] = bucket;
        self.sums[perspective.index()] = self.network.parameters.feature_bias;
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                visit(board.bitboard(color, piece), |square| {
                    self.change_half(perspective, color, piece, square, true);
                });
            }
        }
    }

    fn change_half(
        &mut self,
        perspective: Color,
        color: Color,
        piece: Piece,
        square: u8,
        adding: bool,
    ) {
        let feature = self.buckets[perspective.index()] * INPUTS
            + feature_index(perspective, color, piece, square);
        let offset = feature * HIDDEN;
        kernels::update(
            self.network.backend,
            &mut self.sums[perspective.index()],
            &self.network.parameters.feature_weights[offset..offset + HIDDEN],
            adding,
        );
    }

    pub fn raw_score(&self) -> i64 {
        let mut raw = i64::from(self.network.parameters.output_bias);
        for (half, perspective) in [self.board.side_to_move, self.board.side_to_move.other()]
            .into_iter()
            .enumerate()
        {
            raw += kernels::output(
                self.network.backend,
                &self.sums[perspective.index()],
                &self.network.parameters.output_weights[half * HIDDEN..(half + 1) * HIDDEN],
            );
        }
        raw
    }

    /// Centipawns from side to move, bounded below the search mate-score band.
    /// Draw/mate adjudication and the fifty-move rule belong to search.
    pub fn evaluate(&self) -> i32 {
        score_from_raw(self.raw_score())
    }
}

/// Public arithmetic harness interface. Search keeps separately typed frame
/// vectors so v1/v2 do not copy the wider v3 buffers on every search node.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // Harness only; search stores separately typed, allocation-free frames.
pub enum Accumulator<'net> {
    Legacy(LegacyAccumulator<'net>),
    V3(v3::V3Accumulator<'net>),
}
impl Accumulator<'_> {
    pub fn board(&self) -> &Board {
        match self {
            Self::Legacy(a) => a.board(),
            Self::V3(a) => a.board(),
        }
    }
    pub fn lanes(&self, p: Color) -> &[i16] {
        match self {
            Self::Legacy(a) => a.lanes(p),
            Self::V3(a) => a.lanes(p),
        }
    }
    pub fn bucket_refreshes(&self) -> u32 {
        match self {
            Self::Legacy(a) => a.bucket_refreshes(),
            Self::V3(a) => a.bucket_refreshes(),
        }
    }
    pub fn update(&mut self, board: &Board) -> FeatureChanges {
        match self {
            Self::Legacy(a) => a.update(board),
            Self::V3(a) => a.update(board),
        }
    }
    pub fn raw_score(&self) -> i64 {
        match self {
            Self::Legacy(a) => a.raw_score(),
            Self::V3(a) => a.raw_score(),
        }
    }
    pub fn evaluate(&self) -> i32 {
        match self {
            Self::Legacy(a) => a.evaluate(),
            Self::V3(a) => a.evaluate(),
        }
    }
}

fn visit(mut bits: u64, mut action: impl FnMut(u8)) {
    while bits != 0 {
        let square = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        action(square);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketed_null_move_changes_only_output_perspective() {
        let board = Board::from_fen("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1").unwrap();
        let mut p = Parameters {
            feature_weights: vec![17; KING_BUCKETS * INPUTS * HIDDEN],
            ..Parameters::default()
        };
        p.output_weights[..HIDDEN].fill(7);
        p.output_weights[HIDDEN..].fill(-3);
        let network = Network::new(p).unwrap();
        let mut a = network.accumulator(&board);
        let before = a.clone();
        let null = board.null_move().unwrap();
        assert_eq!(a.update(&null), FeatureChanges::default());
        assert_eq!(a.bucket_refreshes(), 0);
        for color in [Color::White, Color::Black] {
            assert_eq!(a.lanes(color), before.lanes(color));
        }
        assert_eq!(a.raw_score(), network.raw_reference(&null));
        a.update(&board);
        assert_eq!(a.raw_score(), before.raw_score());
    }

    #[test]
    fn actual_null_move_clears_ep_and_swaps_output_without_feature_updates() {
        let board = Board::from_fen("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1").unwrap();
        let mut parameters = Parameters::default();
        parameters.feature_weights
            [feature_index(Color::White, Color::White, Piece::Pawn, 36) * HIDDEN] = 200;
        parameters.output_weights[0] = 64;
        parameters.output_weights[HIDDEN] = -32;
        let network = Network::new(parameters).unwrap();
        let mut accumulator = network.accumulator(&board);
        let before = accumulator.clone();
        let null = board.null_move().unwrap();
        assert_eq!(null.to_fen(), "7k/8/8/3pP3/8/8/8/K7 b - - 0 1");
        assert_eq!(accumulator.update(&null), FeatureChanges::default());
        for color in [Color::White, Color::Black] {
            assert_eq!(before.lanes(color), accumulator.lanes(color));
        }
        assert_eq!(before.raw_score(), 12800);
        assert_eq!(accumulator.raw_score(), -6400);
        assert_eq!(accumulator.raw_score(), network.raw_reference(&null));
        accumulator.update(&board);
        assert_eq!(accumulator.board(), &board);
        assert_eq!(accumulator.raw_score(), before.raw_score());
    }
}
