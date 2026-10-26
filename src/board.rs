//! Bitboard position representation and fully legal move generation.
//!
//! Squares use little-endian rank-file numbering: a1 = 0, h8 = 63.

use std::fmt;

mod move_list;
mod see;
pub(crate) use move_list::{EMPTY_MOVE, MAX_MOVES, MoveList};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenMode {
    All,
    Tactical,
    Quiet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub const fn other(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Piece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl Piece {
    pub const fn index(self) -> usize {
        self as usize
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
const PROMOTIONS: [Piece; 4] = [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight];
const WHITE_KING_SIDE: u8 = 1;
const WHITE_QUEEN_SIDE: u8 = 2;
const BLACK_KING_SIDE: u8 = 4;
const BLACK_QUEEN_SIDE: u8 = 8;
const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<Piece>,
}

impl Move {
    pub fn to_uci(self) -> String {
        let mut result = String::with_capacity(5);
        for square in [self.from, self.to] {
            result.push((b'a' + square % 8) as char);
            result.push((b'1' + square / 8) as char);
        }
        if let Some(piece) = self.promotion {
            result.push(piece_char(piece));
        }
        result
    }
}

impl fmt::Display for Move {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_uci())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    pieces: [[u64; 6]; 2],
    // Piece placement and castling. The public side field and legal en passant
    // are incorporated by hash(), so changing the public side cannot stale it.
    position_key: u64,
    pub side_to_move: Color,
    castling: u8,
    en_passant: Option<u8>,
    pub halfmove_clock: u16,
    pub fullmove_number: u32,
}

impl Default for Board {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Board {
    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS).expect("the built-in starting position is valid")
    }

    /// Parse the six standard FEN fields. Reject structurally impossible positions.
    /// This does not try to prove that a position is reachable from the starting board.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let fields: Vec<_> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err("FEN must contain exactly six fields".into());
        }
        let mut board = Self {
            pieces: [[0; 6]; 2],
            position_key: 0,
            side_to_move: match fields[1] {
                "w" => Color::White,
                "b" => Color::Black,
                _ => return Err("FEN active color must be w or b".into()),
            },
            castling: 0,
            en_passant: None,
            halfmove_clock: parse_unsigned(fields[4], "halfmove clock")?
                .try_into()
                .map_err(|_| "FEN halfmove clock exceeds 65535")?,
            fullmove_number: parse_unsigned(fields[5], "fullmove number")?,
        };
        if board.fullmove_number == 0 {
            return Err("FEN fullmove number must be at least 1".into());
        }
        let ranks: Vec<_> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN piece placement must contain eight ranks".into());
        }
        for (fen_rank, rank) in ranks.iter().enumerate() {
            let mut file = 0u8;
            let mut previous_digit = false;
            for byte in rank.bytes() {
                if (b'1'..=b'8').contains(&byte) {
                    if previous_digit {
                        return Err("FEN must not contain consecutive empty-square digits".into());
                    }
                    file += byte - b'0';
                    previous_digit = true;
                } else {
                    previous_digit = false;
                    let piece = match byte.to_ascii_lowercase() {
                        b'p' => Piece::Pawn,
                        b'n' => Piece::Knight,
                        b'b' => Piece::Bishop,
                        b'r' => Piece::Rook,
                        b'q' => Piece::Queen,
                        b'k' => Piece::King,
                        _ => return Err("FEN contains an invalid piece character".into()),
                    };
                    if file >= 8 {
                        return Err("FEN rank contains more than eight squares".into());
                    }
                    let color = if byte.is_ascii_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let square = (7 - fen_rank as u8) * 8 + file;
                    board.pieces[color.index()][piece.index()] |= bit(square);
                    file += 1;
                }
                if file > 8 {
                    return Err("FEN rank contains more than eight squares".into());
                }
            }
            if file != 8 {
                return Err("FEN rank must contain exactly eight squares".into());
            }
        }
        for color in [Color::White, Color::Black] {
            if board.bitboard(color, Piece::King).count_ones() != 1 {
                return Err("FEN must contain exactly one king of each color".into());
            }
            let pawns = board.bitboard(color, Piece::Pawn);
            if pawns & 0xff00_0000_0000_00ff != 0 {
                return Err("FEN pawns cannot occupy the first or eighth rank".into());
            }
            if pawns.count_ones() > 8 || board.occupied_by(color).count_ones() > 16 {
                return Err("FEN has too many pieces for one side".into());
            }
        }
        let white_king = board.bitboard(Color::White, Piece::King).trailing_zeros() as u8;
        if KING_ATTACKS[white_king as usize] & board.bitboard(Color::Black, Piece::King) != 0 {
            return Err("FEN kings cannot occupy adjacent squares".into());
        }
        if fields[2] != "-" {
            for right in fields[2].bytes() {
                let (flag, color, king, rook) = match right {
                    b'K' => (WHITE_KING_SIDE, Color::White, 4, 7),
                    b'Q' => (WHITE_QUEEN_SIDE, Color::White, 4, 0),
                    b'k' => (BLACK_KING_SIDE, Color::Black, 60, 63),
                    b'q' => (BLACK_QUEEN_SIDE, Color::Black, 60, 56),
                    _ => return Err("FEN contains an invalid castling right".into()),
                };
                if board.castling & flag != 0 {
                    return Err("FEN contains a duplicate castling right".into());
                }
                if board.piece_at(king) != Some((color, Piece::King))
                    || board.piece_at(rook) != Some((color, Piece::Rook))
                {
                    return Err(
                        "FEN castling rights require the king and rook on their home squares"
                            .into(),
                    );
                }
                board.castling |= flag;
            }
        }
        if fields[3] != "-" {
            let square = parse_square(fields[3].as_bytes())
                .ok_or("FEN contains an invalid en passant square")?;
            let (rank, pawn_square, origin_square) = match board.side_to_move {
                Color::White => (5, square.checked_sub(8), square.checked_add(8)),
                Color::Black => (2, square.checked_add(8), square.checked_sub(8)),
            };
            if square / 8 != rank
                || board.piece_at(square).is_some()
                || pawn_square.and_then(|sq| board.piece_at(sq))
                    != Some((board.side_to_move.other(), Piece::Pawn))
                || origin_square.and_then(|sq| board.piece_at(sq)).is_some()
                || board.halfmove_clock != 0
            {
                return Err(
                    "FEN en passant field is inconsistent with the last double pawn move".into(),
                );
            }
            board.en_passant = Some(square);
        }
        if board.in_check(board.side_to_move.other()) {
            return Err("FEN side that just moved cannot remain in check".into());
        }
        board.position_key = board.recompute_position_key();
        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut placement = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0u8;
            for file in 0..8 {
                if let Some((color, piece)) = self.piece_at(rank * 8 + file) {
                    if empty > 0 {
                        placement.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    let mut character = piece_char(piece);
                    if color == Color::White {
                        character = character.to_ascii_uppercase();
                    }
                    placement.push(character);
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                placement.push((b'0' + empty) as char);
            }
            if rank != 0 {
                placement.push('/');
            }
        }
        let mut castling = String::new();
        for (flag, character) in [
            (WHITE_KING_SIDE, 'K'),
            (WHITE_QUEEN_SIDE, 'Q'),
            (BLACK_KING_SIDE, 'k'),
            (BLACK_QUEEN_SIDE, 'q'),
        ] {
            if self.castling & flag != 0 {
                castling.push(character);
            }
        }
        if castling.is_empty() {
            castling.push('-');
        }
        let en_passant = self.en_passant.map_or_else(
            || "-".to_owned(),
            |square| {
                format!(
                    "{}{}",
                    (b'a' + square % 8) as char,
                    (b'1' + square / 8) as char
                )
            },
        );
        format!(
            "{} {} {} {} {} {}",
            placement,
            if self.side_to_move == Color::White {
                "w"
            } else {
                "b"
            },
            castling,
            en_passant,
            self.halfmove_clock,
            self.fullmove_number
        )
    }

    pub fn bitboard(&self, color: Color, piece: Piece) -> u64 {
        self.pieces[color.index()][piece.index()]
    }

    pub fn piece_at(&self, square: u8) -> Option<(Color, Piece)> {
        if square >= 64 {
            return None;
        }
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                if self.bitboard(color, piece) & bit(square) != 0 {
                    return Some((color, piece));
                }
            }
        }
        None
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = self.generate(GenMode::All);
        moves.retain(|mv| !self.apply_move(*mv, false).in_check(self.side_to_move));
        moves.to_vec()
    }

    /// Legal captures and all four promotions, without generating quiet moves.
    pub fn legal_tactical_moves(&self) -> Vec<Move> {
        let mut moves = self.generate(GenMode::Tactical);
        moves.retain(|mv| !self.apply_move(*mv, false).in_check(self.side_to_move));
        moves.to_vec()
    }

    /// Exact existence check; stop checking legality after the first legal move.
    pub fn has_legal_move(&self) -> bool {
        self.first_legal().is_some()
    }

    pub(crate) fn first_legal(&self) -> Option<(Move, Self)> {
        for mode in [GenMode::Tactical, GenMode::Quiet] {
            for &mv in self.generate(mode).iter() {
                if let Some(child) = self.generated_child(mv) {
                    return Some((mv, child));
                }
            }
        }
        None
    }

    pub(crate) fn legal_count_up_to(&self, limit: usize) -> usize {
        let mut count = 0;
        for &mv in self.generate(GenMode::All).iter() {
            if !self.apply_move(mv, false).in_check(self.side_to_move) {
                count += 1;
                if count >= limit {
                    break;
                }
            }
        }
        count
    }

    /// Geometry is guaranteed by generate(); one board copy also becomes the
    /// searched child. Includes discovered EP checks and king destination safety.
    pub(crate) fn generated_child(&self, mv: Move) -> Option<Self> {
        let child = self.make_move(mv);
        (!child.in_check(self.side_to_move)).then_some(child)
    }

    /// Validate an untrusted full-key TT move without generating a move list.
    pub(crate) fn hash_child(&self, mv: Move) -> Option<Self> {
        self.is_pseudo_legal(mv)
            .then(|| self.generated_child(mv))
            .flatten()
    }

    fn is_pseudo_legal(&self, mv: Move) -> bool {
        if mv.from >= 64 || mv.to >= 64 || mv.from == mv.to {
            return false;
        }
        let Some((color, piece)) = self.piece_at(mv.from) else {
            return false;
        };
        if color != self.side_to_move
            || self
                .piece_at(mv.to)
                .is_some_and(|(c, p)| c == color || p == Piece::King)
        {
            return false;
        }
        if piece != Piece::Pawn {
            if mv.promotion.is_some() {
                return false;
            }
            if piece == Piece::King && mv.from.abs_diff(mv.to) == 2 {
                return self.generate(GenMode::Quiet).contains(&mv);
            }
            return self.piece_attacks(piece, color, mv.from) & bit(mv.to) != 0;
        }
        if matches!(mv.to / 8, 0 | 7) {
            if !mv.promotion.is_some_and(|p| PROMOTIONS.contains(&p)) {
                return false;
            }
        } else if mv.promotion.is_some() {
            return false;
        }
        if PAWN_ATTACKS[color.index()][mv.from as usize] & bit(mv.to) != 0 {
            return self.piece_at(mv.to).is_some() || self.en_passant == Some(mv.to);
        }
        if self.piece_at(mv.to).is_some() {
            return false;
        }
        let forward = if color == Color::White { 8i16 } else { -8 };
        let difference = i16::from(mv.to) - i16::from(mv.from);
        difference == forward
            || (difference == 2 * forward
                && mv.from / 8 == if color == Color::White { 1 } else { 6 }
                && self
                    .piece_at((i16::from(mv.from) + forward) as u8)
                    .is_none())
    }

    /// Artificial search operation, never a legal game move. Counters and
    /// placement stay unchanged; repetition across this boundary is invalid.
    pub(crate) fn null_move(&self) -> Option<Self> {
        if self.in_check(self.side_to_move) {
            return None;
        }
        let mut next = *self;
        next.side_to_move = self.side_to_move.other();
        next.en_passant = None;
        Some(next)
    }

    /// Apply a generated move without mutating the original position.
    /// The caller must supply a legal or internally generated pseudo-legal move.
    pub fn make_move(&self, mv: Move) -> Self {
        self.apply_move(mv, true)
    }

    // Legality probes need piece placement only; avoid maintaining a key for
    // these private temporary positions, which must never escape or be hashed.
    fn apply_move(&self, mv: Move, update_key: bool) -> Self {
        let mut next = *self;
        let color = self.side_to_move;
        let enemy = color.other();
        let (owner, piece) = self
            .piece_at(mv.from)
            .expect("move origin must hold a piece");
        debug_assert_eq!(owner, color);
        debug_assert!(mv.from < 64 && mv.to < 64);
        let captured = self.piece_at(mv.to);
        next.pieces[color.index()][piece.index()] &= !bit(mv.from);
        if let Some((captured_color, captured_piece)) = captured {
            debug_assert_eq!(captured_color, enemy);
            debug_assert_ne!(captured_piece, Piece::King);
            next.pieces[captured_color.index()][captured_piece.index()] &= !bit(mv.to);
        }
        if piece == Piece::Pawn && self.en_passant == Some(mv.to) && captured.is_none() {
            let victim = if color == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            };
            next.pieces[enemy.index()][Piece::Pawn.index()] &= !bit(victim);
        }
        next.pieces[color.index()][mv.promotion.unwrap_or(piece).index()] |= bit(mv.to);
        if piece == Piece::King {
            next.castling &= match color {
                Color::White => !(WHITE_KING_SIDE | WHITE_QUEEN_SIDE),
                Color::Black => !(BLACK_KING_SIDE | BLACK_QUEEN_SIDE),
            };
            if mv.from.abs_diff(mv.to) == 2 {
                let (rook_from, rook_to) = if mv.to > mv.from {
                    (mv.from + 3, mv.from + 1)
                } else {
                    (mv.from - 4, mv.from - 1)
                };
                next.pieces[color.index()][Piece::Rook.index()] &= !bit(rook_from);
                next.pieces[color.index()][Piece::Rook.index()] |= bit(rook_to);
            }
        }
        // Moving from, or capturing on, a rook home square permanently loses its right.
        for square in [mv.from, mv.to] {
            next.castling &= match square {
                0 => !WHITE_QUEEN_SIDE,
                7 => !WHITE_KING_SIDE,
                56 => !BLACK_QUEEN_SIDE,
                63 => !BLACK_KING_SIDE,
                _ => u8::MAX,
            };
        }
        next.en_passant = if piece == Piece::Pawn && mv.from.abs_diff(mv.to) == 16 {
            Some((mv.from + mv.to) / 2)
        } else {
            None
        };
        next.halfmove_clock = if piece == Piece::Pawn || captured.is_some() {
            0
        } else {
            self.halfmove_clock.saturating_add(1)
        };
        if color == Color::Black {
            next.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        next.side_to_move = enemy;
        if update_key {
            let mut key = self.position_key
                ^ piece_key(color, piece, mv.from)
                ^ piece_key(color, mv.promotion.unwrap_or(piece), mv.to);
            if let Some((victim_color, victim_piece)) = captured {
                key ^= piece_key(victim_color, victim_piece, mv.to);
            } else if piece == Piece::Pawn && self.en_passant == Some(mv.to) {
                let square = if color == Color::White {
                    mv.to - 8
                } else {
                    mv.to + 8
                };
                key ^= piece_key(enemy, Piece::Pawn, square);
            }
            if piece == Piece::King && mv.from.abs_diff(mv.to) == 2 {
                let (from, to) = if mv.to > mv.from {
                    (mv.from + 3, mv.from + 1)
                } else {
                    (mv.from - 4, mv.from - 1)
                };
                key ^= piece_key(color, Piece::Rook, from) ^ piece_key(color, Piece::Rook, to);
            }
            for right in 0..4 {
                if (self.castling ^ next.castling) & (1 << right) != 0 {
                    key ^= ZOBRIST[769 + right];
                }
            }
            next.position_key = key;
        }
        next
    }

    pub fn in_check(&self, color: Color) -> bool {
        let king = self.bitboard(color, Piece::King);
        debug_assert_eq!(king.count_ones(), 1);
        self.is_attacked(king.trailing_zeros() as u8, color.other())
    }

    /// Deterministic position key, excluding move counters. An en passant target is
    /// included only when it changes the set of legal moves (including pin checks).
    pub fn hash(&self) -> u64 {
        let mut hash = self.position_key;
        if self.side_to_move == Color::Black {
            hash ^= ZOBRIST[768];
        }
        if let Some(square) = self.en_passant {
            let mut pawns = PAWN_ATTACKS[self.side_to_move.other().index()][square as usize]
                & self.bitboard(self.side_to_move, Piece::Pawn);
            while pawns != 0 {
                let mv = Move {
                    from: pop_lsb(&mut pawns),
                    to: square,
                    promotion: None,
                };
                if !self.apply_move(mv, false).in_check(self.side_to_move) {
                    hash ^= ZOBRIST[773 + (square % 8) as usize];
                    break;
                }
            }
        }
        hash
    }

    fn recompute_position_key(&self) -> u64 {
        let mut hash = 0;
        for color in [Color::White, Color::Black] {
            for piece in PIECES {
                let mut squares = self.bitboard(color, piece);
                while squares != 0 {
                    let square = pop_lsb(&mut squares) as usize;
                    hash ^= ZOBRIST[(color.index() * 6 + piece.index()) * 64 + square];
                }
            }
        }
        for right in 0..4 {
            if self.castling & (1 << right) != 0 {
                hash ^= ZOBRIST[769 + right];
            }
        }
        hash
    }

    pub fn parse_uci_move(&self, text: &str) -> Result<Move, String> {
        let bytes = text.as_bytes();
        if bytes.len() != 4 && bytes.len() != 5 {
            return Err("UCI move must contain two squares and an optional promotion".into());
        }
        let from = parse_square(&bytes[..2]).ok_or("invalid UCI origin square")?;
        let to = parse_square(&bytes[2..4]).ok_or("invalid UCI destination square")?;
        let promotion = if bytes.len() == 5 {
            Some(match bytes[4] {
                b'q' => Piece::Queen,
                b'r' => Piece::Rook,
                b'b' => Piece::Bishop,
                b'n' => Piece::Knight,
                _ => return Err("UCI promotion must be q, r, b, or n".into()),
            })
        } else {
            None
        };
        let mv = Move {
            from,
            to,
            promotion,
        };
        if self.legal_moves().contains(&mv) {
            Ok(mv)
        } else {
            Err(format!("illegal move: {text}"))
        }
    }

    /// Recognize only material configurations where checkmate is impossible.
    pub fn is_insufficient_material(&self) -> bool {
        for color in [Color::White, Color::Black] {
            if self.bitboard(color, Piece::Pawn)
                | self.bitboard(color, Piece::Rook)
                | self.bitboard(color, Piece::Queen)
                != 0
            {
                return false;
            }
        }
        let knights =
            self.bitboard(Color::White, Piece::Knight) | self.bitboard(Color::Black, Piece::Knight);
        let bishops =
            self.bitboard(Color::White, Piece::Bishop) | self.bitboard(Color::Black, Piece::Bishop);
        if (knights | bishops).count_ones() <= 1 {
            return true;
        }
        if knights != 0 {
            return false;
        }
        // Any number of bishops confined to one square color cannot mate.
        let dark = 0xaa55_aa55_aa55_aa55u64;
        bishops & dark == 0 || bishops & !dark == 0
    }

    pub(crate) fn occupied_by(&self, color: Color) -> u64 {
        self.pieces[color.index()]
            .iter()
            .fold(0, |all, &pieces| all | pieces)
    }

    pub(crate) fn occupied(&self) -> u64 {
        self.occupied_by(Color::White) | self.occupied_by(Color::Black)
    }

    pub(crate) fn has_castling_rights(&self) -> bool {
        self.castling != 0
    }

    fn is_attacked(&self, square: u8, by: Color) -> bool {
        if PAWN_ATTACKS[by.other().index()][square as usize] & self.bitboard(by, Piece::Pawn) != 0
            || KNIGHT_ATTACKS[square as usize] & self.bitboard(by, Piece::Knight) != 0
            || KING_ATTACKS[square as usize] & self.bitboard(by, Piece::King) != 0
        {
            return true;
        }
        let occupied = self.occupied();
        let queens = self.bitboard(by, Piece::Queen);
        sliding_attacks(square, occupied, &DIAGONALS) & (queens | self.bitboard(by, Piece::Bishop))
            != 0
            || sliding_attacks(square, occupied, &ORTHOGONALS)
                & (queens | self.bitboard(by, Piece::Rook))
                != 0
    }

    /// Geometric attacks including the first blocker; ignores pins and own men.
    /// Evaluation uses these as pressure, never as legal moves.
    pub(crate) fn piece_attacks(&self, piece: Piece, color: Color, square: u8) -> u64 {
        match piece {
            Piece::Pawn => PAWN_ATTACKS[color.index()][square as usize],
            Piece::Knight => KNIGHT_ATTACKS[square as usize],
            Piece::Bishop => sliding_attacks(square, self.occupied(), &DIAGONALS),
            Piece::Rook => sliding_attacks(square, self.occupied(), &ORTHOGONALS),
            Piece::Queen => {
                sliding_attacks(square, self.occupied(), &DIAGONALS)
                    | sliding_attacks(square, self.occupied(), &ORTHOGONALS)
            }
            Piece::King => KING_ATTACKS[square as usize],
        }
    }

    pub(crate) fn generate(&self, mode: GenMode) -> MoveList {
        let tactical_only = mode == GenMode::Tactical;
        let color = self.side_to_move;
        let enemy = color.other();
        let own = self.occupied_by(color);
        let opposing = self.occupied_by(enemy);
        let occupied = own | opposing;
        let enemy_king = self.bitboard(enemy, Piece::King);
        let destinations = !own
            & !enemy_king
            & match mode {
                GenMode::Tactical => opposing,
                GenMode::Quiet => !occupied,
                GenMode::All => u64::MAX,
            };
        let mut moves = MoveList::default();
        let mut pawns = self.bitboard(color, Piece::Pawn);
        let forward: i16 = if color == Color::White { 8 } else { -8 };
        let start_rank = if color == Color::White { 1 } else { 6 };
        while pawns != 0 {
            let from = pop_lsb(&mut pawns);
            let to = (from as i16 + forward) as u8;
            if occupied & bit(to) == 0 {
                if (!tactical_only || matches!(to / 8, 0 | 7))
                    && !(mode == GenMode::Quiet && matches!(to / 8, 0 | 7))
                {
                    add_pawn_move(&mut moves, from, to);
                }
                if !tactical_only && from / 8 == start_rank {
                    let double = (from as i16 + forward * 2) as u8;
                    if occupied & bit(double) == 0 {
                        moves.push(Move {
                            from,
                            to: double,
                            promotion: None,
                        });
                    }
                }
            }
            let mut captures = PAWN_ATTACKS[color.index()][from as usize]
                & ((opposing & !enemy_king) | self.en_passant.map_or(0, bit));
            while captures != 0 && mode != GenMode::Quiet {
                add_pawn_move(&mut moves, from, pop_lsb(&mut captures));
            }
        }
        for piece in [
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            let mut pieces = self.bitboard(color, piece);
            while pieces != 0 {
                let from = pop_lsb(&mut pieces);
                let attacks = match piece {
                    Piece::Knight => KNIGHT_ATTACKS[from as usize],
                    Piece::Bishop => sliding_attacks(from, occupied, &DIAGONALS),
                    Piece::Rook => sliding_attacks(from, occupied, &ORTHOGONALS),
                    Piece::Queen => {
                        sliding_attacks(from, occupied, &DIAGONALS)
                            | sliding_attacks(from, occupied, &ORTHOGONALS)
                    }
                    Piece::King => KING_ATTACKS[from as usize],
                    Piece::Pawn => unreachable!(),
                };
                let mut targets = attacks & destinations;
                while targets != 0 {
                    moves.push(Move {
                        from,
                        to: pop_lsb(&mut targets),
                        promotion: None,
                    });
                }
            }
        }
        if !tactical_only && !self.in_check(color) {
            let (from, king_side, queen_side) = match color {
                Color::White => (4, WHITE_KING_SIDE, WHITE_QUEEN_SIDE),
                Color::Black => (60, BLACK_KING_SIDE, BLACK_QUEEN_SIDE),
            };
            if self.castling & king_side != 0
                && occupied & (bit(from + 1) | bit(from + 2)) == 0
                && !self.is_attacked(from + 1, enemy)
                && !self.is_attacked(from + 2, enemy)
            {
                moves.push(Move {
                    from,
                    to: from + 2,
                    promotion: None,
                });
            }
            if self.castling & queen_side != 0
                && occupied & (bit(from - 1) | bit(from - 2) | bit(from - 3)) == 0
                && !self.is_attacked(from - 1, enemy)
                && !self.is_attacked(from - 2, enemy)
            {
                moves.push(Move {
                    from,
                    to: from - 2,
                    promotion: None,
                });
            }
        }
        moves
    }
}

pub fn perft(board: &Board, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = board.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .into_iter()
        .map(|mv| perft(&board.make_move(mv), depth - 1))
        .sum()
}

pub fn perft_divide(board: &Board, depth: u8) -> Vec<(Move, u64)> {
    if depth == 0 {
        return Vec::new();
    }
    board
        .legal_moves()
        .into_iter()
        .map(|mv| (mv, perft(&board.make_move(mv), depth - 1)))
        .collect()
}

fn parse_unsigned(text: &str, name: &str) -> Result<u32, String> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("FEN {name} must be an unsigned integer"));
    }
    text.parse()
        .map_err(|_| format!("FEN {name} exceeds the supported range"))
}

fn parse_square(text: &[u8]) -> Option<u8> {
    if text.len() == 2 && (b'a'..=b'h').contains(&text[0]) && (b'1'..=b'8').contains(&text[1]) {
        Some((text[1] - b'1') * 8 + text[0] - b'a')
    } else {
        None
    }
}

fn piece_char(piece: Piece) -> char {
    match piece {
        Piece::Pawn => 'p',
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::King => 'k',
    }
}

const fn bit(square: u8) -> u64 {
    1u64 << square
}

fn piece_key(color: Color, piece: Piece, square: u8) -> u64 {
    ZOBRIST[(color.index() * 6 + piece.index()) * 64 + square as usize]
}

fn pop_lsb(bits: &mut u64) -> u8 {
    let square = bits.trailing_zeros() as u8;
    *bits &= *bits - 1;
    square
}

fn add_pawn_move(moves: &mut MoveList, from: u8, to: u8) {
    if to / 8 == 0 || to / 8 == 7 {
        for piece in PROMOTIONS {
            moves.push(Move {
                from,
                to,
                promotion: Some(piece),
            });
        }
    } else {
        moves.push(Move {
            from,
            to,
            promotion: None,
        });
    }
}

const DIAGONALS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHOGONALS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

const fn generate_rays(directions: &[(i8, i8); 4]) -> [[u64; 64]; 4] {
    let mut rays = [[0; 64]; 4];
    let mut direction = 0;
    while direction < 4 {
        let mut square = 0;
        while square < 64 {
            let mut file = (square % 8) as i8 + directions[direction].0;
            let mut rank = (square / 8) as i8 + directions[direction].1;
            while file >= 0 && file < 8 && rank >= 0 && rank < 8 {
                rays[direction][square] |= bit((rank * 8 + file) as u8);
                file += directions[direction].0;
                rank += directions[direction].1;
            }
            square += 1;
        }
        direction += 1;
    }
    rays
}

static BISHOP_RAYS: [[u64; 64]; 4] = generate_rays(&DIAGONALS);
static ROOK_RAYS: [[u64; 64]; 4] = generate_rays(&ORTHOGONALS);

#[inline]
fn sliding_attacks(square: u8, occupied: u64, directions: &[(i8, i8); 4]) -> u64 {
    let rays = if directions == &DIAGONALS {
        &BISHOP_RAYS
    } else {
        &ROOK_RAYS
    };
    let mut attacks = 0;
    for (direction, &(df, dr)) in directions.iter().enumerate() {
        let ray = rays[direction][square as usize];
        let blockers = ray & occupied;
        attacks |= if blockers == 0 {
            ray
        } else {
            let nearest = if df + 8 * dr > 0 {
                blockers.trailing_zeros()
            } else {
                63 - blockers.leading_zeros()
            };
            ray & !rays[direction][nearest as usize]
        };
    }
    attacks
}

const fn leaper_attacks(deltas: &[(i8, i8)]) -> [u64; 64] {
    let mut table = [0; 64];
    let mut square = 0;
    while square < 64 {
        let mut direction = 0;
        while direction < deltas.len() {
            let file = (square % 8) as i8 + deltas[direction].0;
            let rank = (square / 8) as i8 + deltas[direction].1;
            if file >= 0 && file < 8 && rank >= 0 && rank < 8 {
                table[square] |= bit((rank * 8 + file) as u8);
            }
            direction += 1;
        }
        square += 1;
    }
    table
}

const KNIGHT_ATTACKS: [u64; 64] = leaper_attacks(&[
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]);
const KING_ATTACKS: [u64; 64] = leaper_attacks(&[
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
]);
const PAWN_ATTACKS: [[u64; 64]; 2] = [
    leaper_attacks(&[(1, 1), (-1, 1)]),
    leaper_attacks(&[(1, -1), (-1, -1)]),
];

const fn zobrist_keys() -> [u64; 781] {
    let mut keys = [0; 781];
    let mut state = 0x5a45_4e49_5448_2026u64;
    let mut index = 0;
    while index < keys.len() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        keys[index] = value ^ (value >> 31);
        index += 1;
    }
    keys
}

const ZOBRIST: [u64; 781] = zobrist_keys();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_move_clears_en_passant_and_preserves_real_counters_and_placement() {
        let board = Board::from_fen("7k/8/8/3pP3/8/8/8/K7 w - d6 0 12").unwrap();
        let next = board.null_move().unwrap();
        assert_eq!(next.to_fen(), "7k/8/8/3pP3/8/8/8/K7 b - - 0 12");
        assert_eq!(next.hash(), Board::from_fen(&next.to_fen()).unwrap().hash());
        assert_eq!(board.en_passant, Some(43));
        assert_eq!(next.en_passant, None);
        let checked = Board::from_fen("4r2k/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(checked.null_move().is_none());
        assert!(board.parse_uci_move("0000").is_err());
    }

    #[test]
    fn ray_attacks_match_walking_reference_for_every_blocker_subset() {
        fn walk(square: u8, occupied: u64, directions: &[(i8, i8); 4]) -> u64 {
            let mut result = 0;
            for &(df, dr) in directions {
                let mut file = (square % 8) as i8 + df;
                let mut rank = (square / 8) as i8 + dr;
                while (0..8).contains(&file) && (0..8).contains(&rank) {
                    let target = 1u64 << (8 * rank + file);
                    result |= target;
                    if occupied & target != 0 {
                        break;
                    }
                    file += df;
                    rank += dr;
                }
            }
            result
        }
        for directions in [&DIAGONALS, &ORTHOGONALS] {
            for square in 0..64 {
                let mask = walk(square, 0, directions);
                let mut subset = 0u64;
                loop {
                    assert_eq!(
                        sliding_attacks(square, subset, directions),
                        walk(square, subset, directions)
                    );
                    // Occupancy outside these rays must have no influence.
                    assert_eq!(
                        sliding_attacks(square, subset | !mask, directions),
                        walk(square, subset, directions)
                    );
                    subset = subset.wrapping_sub(mask) & mask;
                    if subset == 0 {
                        break;
                    }
                }
            }
        }
    }

    fn fen(text: &str) -> Board {
        Board::from_fen(text).unwrap()
    }

    #[test]
    fn start_position_perft() {
        let board = Board::startpos();
        for (depth, expected) in [(0, 1), (1, 20), (2, 400), (3, 8902), (4, 197281)] {
            assert_eq!(perft(&board, depth), expected, "depth {depth}");
        }
        assert_eq!(board.to_fen(), STARTPOS);
        assert_eq!(
            perft_divide(&board, 3)
                .iter()
                .map(|(_, nodes)| nodes)
                .sum::<u64>(),
            8902
        );
    }

    #[test]
    fn kiwipete_perft() {
        let board = fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        for (depth, expected) in [(1, 48), (2, 2039), (3, 97862)] {
            assert_eq!(perft(&board, depth), expected, "depth {depth}");
        }
    }

    #[test]
    fn endgame_perft_with_en_passant_and_checks() {
        let board = fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        for (depth, expected) in [(1, 14), (2, 191), (3, 2812), (4, 43238)] {
            assert_eq!(perft(&board, depth), expected, "depth {depth}");
        }
    }

    #[test]
    fn double_push_and_clocks_round_trip() {
        let board = Board::startpos();
        let next = board.make_move(board.parse_uci_move("e2e4").unwrap());
        assert_eq!(
            next.to_fen(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        );
        let next = next.make_move(next.parse_uci_move("g8f6").unwrap());
        assert_eq!(next.halfmove_clock, 1);
        assert_eq!(next.fullmove_number, 2);
        assert_eq!(Board::from_fen(&next.to_fen()).unwrap(), next);
        assert_eq!(board.to_fen(), STARTPOS);
    }

    #[test]
    fn castling_moves_rook_and_removes_rights() {
        let board = fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let next = board.make_move(board.parse_uci_move("e1g1").unwrap());
        assert_eq!(next.piece_at(5), Some((Color::White, Piece::Rook)));
        assert_eq!(next.piece_at(6), Some((Color::White, Piece::King)));
        assert_eq!(next.piece_at(7), None);
        assert_eq!(next.castling, BLACK_KING_SIDE | BLACK_QUEEN_SIDE);
        let capture = board.make_move(board.parse_uci_move("a1a8").unwrap());
        assert_eq!(capture.castling, WHITE_KING_SIDE | BLACK_KING_SIDE);
    }

    #[test]
    fn castling_cannot_cross_check_but_rook_may_be_attacked() {
        let crossing = fen("4kr2/8/8/8/8/8/8/4K2R w K - 0 1");
        assert!(crossing.parse_uci_move("e1g1").is_err());
        let rook_attacked = fen("4k2r/8/8/8/8/8/8/4K2R w K - 0 1");
        assert!(rook_attacked.parse_uci_move("e1g1").is_ok());
    }

    #[test]
    fn en_passant_removes_captured_pawn_and_rejects_discovered_check() {
        let board = fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2");
        let next = board.make_move(board.parse_uci_move("e5d6").unwrap());
        assert_eq!(next.piece_at(35), None);
        assert_eq!(next.piece_at(43), Some((Color::White, Piece::Pawn)));
        let pinned = fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 2");
        assert!(pinned.parse_uci_move("g5f6").is_err());
    }

    #[test]
    fn four_promotion_choices_and_required_suffix() {
        let board = fen("7k/P7/8/8/8/8/8/7K w - - 0 1");
        let promotions: Vec<_> = board
            .legal_moves()
            .into_iter()
            .filter(|mv| mv.from == 48)
            .collect();
        assert_eq!(promotions.len(), 4);
        assert!(board.parse_uci_move("a7a8").is_err());
        for suffix in ["q", "r", "b", "n"] {
            let mv = board.parse_uci_move(&format!("a7a8{suffix}")).unwrap();
            let next = board.make_move(mv);
            assert_eq!(next.bitboard(Color::White, Piece::Pawn), 0);
            assert_eq!(
                next.piece_at(56),
                Some((Color::White, mv.promotion.unwrap()))
            );
        }
    }

    #[test]
    fn hash_tracks_rights_and_only_legal_en_passant() {
        let with_ep = fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2");
        let no_ep = fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2");
        assert_ne!(with_ep.hash(), no_ep.hash());
        let pinned = fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 2");
        let pinned_no_ep = fen("4k3/8/8/r4pPK/8/8/8/8 w - - 0 2");
        assert_eq!(pinned.hash(), pinned_no_ep.hash());
        let unused = fen("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 2");
        let unused_no_ep = fen("4k3/8/8/3p4/8/8/8/4K3 w - - 23 7");
        assert_eq!(unused.hash(), unused_no_ep.hash());
        let rights = fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        let no_rights = fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1");
        assert_ne!(rights.hash(), no_rights.hash());
        let black_turn = fen("4k3/8/8/8/8/8/8/4K2R b - - 0 1");
        assert_ne!(black_turn.hash(), no_rights.hash());
    }

    #[test]
    fn checkmate_and_stalemate_have_no_legal_moves() {
        let mate = fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
        assert!(mate.in_check(Color::Black));
        assert!(mate.legal_moves().is_empty());
        let stalemate = fen("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1");
        assert!(!stalemate.in_check(Color::Black));
        assert!(stalemate.legal_moves().is_empty());
    }

    #[test]
    fn conservative_insufficient_material() {
        for text in [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2N1K3 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
        ] {
            assert!(fen(text).is_insufficient_material(), "{text}");
        }
        for text in [
            "4k3/8/8/8/8/8/8/2NNK3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2BBK3 w - - 0 1",
            "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",
        ] {
            assert!(!fen(text).is_insufficient_material(), "{text}");
        }
    }

    #[test]
    fn malformed_fen_and_moves_return_errors() {
        for text in [
            "",
            "8/8/8/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/8/8/4K2k w - - 0",
            "8/8/8/8/8/8/8/4K2k x - - 0 1",
            "8/8/8/8/8/8/8/4K2k w - - 0 0",
            "8/8/8/8/8/8/8/4K2k w - - -1 1",
            "8/8/8/8/8/8/8/4K2k w - - 65536 1",
            "8/8/8/8/8/8/8/4K2k w - - 0 4294967296",
            "8/8/8/8/8/8/8/4K2k w K - 0 1",
            "4k3/8/8/8/8/8/8/4K2R w KK - 0 1",
            "8/8/8/8/8/8/8/4K1kP w - - 0 1",
            "8/8/8/8/8/8/8/4Kk2 w - - 0 1",
            "8/8/8/8/8/8/8/4KK1k w - - 0 1",
            "8/8/8/8/8/8/8/4K2k w - e6 0 1",
            "4k3/8/8/3p4/8/8/8/4K3 w - d6 1 2",
            "44/8/8/8/8/8/8/4K2k w - - 0 1",
            "9/8/8/8/8/8/8/4K2k w - - 0 1",
            "8/8/8/8/8/8/8/4K2é w - - 0 1",
            "4k3/8/8/8/8/8/8/4R2K w - - 0 1",
        ] {
            assert!(
                Board::from_fen(text).is_err(),
                "accepted invalid FEN: {text}"
            );
        }
        let board = Board::startpos();
        for text in [
            "", "e2e5", "e2e4q", "e9e4", "i2e4", "éé", "e2e4z", "0000", "E2E4",
        ] {
            assert!(
                board.parse_uci_move(text).is_err(),
                "accepted invalid move: {text}"
            );
        }
    }
}
