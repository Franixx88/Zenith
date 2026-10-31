//! Process-isolated streaming adapter; never materializes an expanded dataset.
use serde::Serialize;
use sfbinpack::{
    ChunkReader, TrainingDataEntry,
    chess::{
        color::Color as SfColor,
        coords::Square,
        r#move::{Move as SfMove, MoveType},
        piece::Piece as SfPiece,
        piecetype::PieceType,
        position::Position,
    },
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufRead, BufWriter, Read, Seek, SeekFrom, Write},
    path::PathBuf,
};
use zenith::board::{Board, Color, Piece};
#[path = "../../../examples/nnue_data/mod.rs"]
mod encoding;

const MAGIC: &[u8; 8] = b"ZBSTRM1\0";
const MAGIC_RESULTS: &[u8; 8] = b"ZBSTRM2\0";
const MAX_CHUNK: u32 = 100 * 1024 * 1024;
const MAX_INDEX: usize = 1_000_000;

struct Config {
    files: Vec<PathBuf>,
    split: String,
    val_mod: u64,
    limit: u64,
    seed: u64,
    batch: usize,
    max_cp: i32,
    numerator: i64,
    denominator: i64,
    shuffle: bool,
    quiet: bool,
    inspect: bool,
    with_results: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            files: vec![],
            split: "train".into(),
            val_mod: 20,
            limit: 0,
            seed: 1,
            batch: 4096,
            max_cp: 3000,
            numerator: 1,
            denominator: 1,
            shuffle: true,
            quiet: true,
            inspect: false,
            with_results: false,
        }
    }
}
#[derive(Default, Serialize)]
struct Stats {
    chunks: u64,
    decoded: u64,
    emitted: u64,
    limited: bool,
    rejected: BTreeMap<&'static str, u64>,
}
impl Stats {
    fn reject(&mut self, why: &'static str) {
        *self.rejected.entry(why).or_default() += 1;
    }
}
#[derive(Clone, Copy)]
struct Chunk {
    file: usize,
    offset: u64,
    len: u32,
}

fn index(files: &[PathBuf]) -> Result<Vec<Chunk>, String> {
    let mut index = Vec::new();
    for (file_index, path) in files.iter().enumerate() {
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        let mut offset = 0u64;
        while offset < size {
            let mut header = [0; 8];
            file.read_exact(&mut header)
                .map_err(|_| format!("{}: truncated BINP header at {offset}", path.display()))?;
            let len = u32::from_le_bytes(header[4..].try_into().unwrap());
            if &header[..4] != b"BINP" || !(34..=MAX_CHUNK).contains(&len) {
                return Err(format!(
                    "{}: invalid BINP header/length at {offset}",
                    path.display()
                ));
            }
            let end = offset
                .checked_add(8 + u64::from(len))
                .ok_or("chunk offset overflow")?;
            if end > size {
                return Err(format!(
                    "{}: truncated BINP payload at {offset}",
                    path.display()
                ));
            }
            index.push(Chunk {
                file: file_index,
                offset: offset + 8,
                len,
            });
            if index.len() > MAX_INDEX {
                return Err("chunk index exceeds bounded limit".into());
            }
            file.seek(SeekFrom::Start(end)).map_err(|e| e.to_string())?;
            offset = end;
        }
    }
    if index.is_empty() {
        return Err("no BINP chunks".into());
    }
    Ok(index)
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn parse() -> Result<Config, String> {
    let mut args = std::env::args().skip(1);
    let mut cfg = Config::default();
    if let Some(command) = args.next() {
        cfg.inspect = match command.as_str() {
            "stream" => false,
            "inspect" => true,
            _ => return Err("expected stream or inspect".into()),
        };
    } else {
        return Err("usage: zenith-binpack stream|inspect --input FILE [--input FILE...] [--limit N] [--seed N] [--split train|val|all]".into());
    }
    while let Some(flag) = args.next() {
        if flag == "--no-shuffle" {
            cfg.shuffle = false;
            continue;
        }
        if flag == "--keep-tactical" {
            cfg.quiet = false;
            continue;
        }
        if flag == "--with-results" {
            cfg.with_results = true;
            continue;
        }
        let value = args.next().ok_or_else(|| format!("missing {flag} value"))?;
        let number = || {
            value
                .parse::<u64>()
                .map_err(|_| format!("{flag} requires a nonnegative integer"))
        };
        match flag.as_str() {
            "--input" => cfg.files.push(value.into()),
            "--split" => cfg.split = value,
            "--val-mod" => cfg.val_mod = number()?,
            "--seed" => cfg.seed = number()?,
            "--limit" => cfg.limit = number()?,
            "--batch-rows" => {
                let n = number()?;
                if !(1..=65536).contains(&n) {
                    return Err("batch-rows must be 1..65536".into());
                }
                cfg.batch = n as usize;
            }
            "--max-cp" => {
                let n = number()?;
                if !(1..=30000).contains(&n) {
                    return Err("max-cp must be 1..30000".into());
                }
                cfg.max_cp = n as i32;
            }
            "--score-num" | "--score-den" => {
                let n = number()?;
                if !(1..=1_000_000).contains(&n) {
                    return Err("score scale must be 1..1000000".into());
                }
                if flag == "--score-num" {
                    cfg.numerator = n as i64;
                } else {
                    cfg.denominator = n as i64;
                }
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if cfg.files.is_empty()
        || cfg.val_mod < 2
        || !["train", "val", "all"].contains(&cfg.split.as_str())
    {
        return Err("inputs, val-mod >=2 and a valid split are required".into());
    }
    Ok(cfg)
}

fn candidate(entry: &TrainingDataEntry, cfg: &Config, stats: &mut Stats) -> Option<[u8; 100]> {
    // VALUE_NONE is never turned into a valid label by rescaling.
    // Conservative raw mate/tablebase band, before any unit conversion.
    if entry.score.unsigned_abs() >= 30000 {
        stats.reject("sentinel_or_mate");
        return None;
    }
    let score = i64::from(entry.score) * cfg.numerator / cfg.denominator;
    if score.unsigned_abs() > cfg.max_cp as u64 {
        stats.reject("score");
        return None;
    }
    if !(-1..=1).contains(&entry.result) {
        stats.reject("result");
        return None;
    }
    let side = entry.pos.side_to_move();
    if cfg.quiet
        && (entry.mv.mtype() == MoveType::Promotion
            || entry.mv.mtype() == MoveType::EnPassant
            || (entry.mv.mtype() != MoveType::Castle
                && entry.pos.pieces_bb(!side).sq_set(entry.mv.to())))
    {
        stats.reject("teacher_tactical");
        return None;
    }
    // Fast canonical feature key before expensive FEN/legality checks. This is
    // exactly the input-based split used by the TSV preparer, across all files.
    let mut fast = [0u8; 100];
    for (half, color) in [side, !side].into_iter().enumerate() {
        for piece in 0..6u8 {
            let mut bits = entry
                .pos
                .pieces_bb_color(color, PieceType::from_ordinal(piece))
                .bits();
            if side == SfColor::Black {
                bits = bits.swap_bytes();
            }
            let i = (half * 6 + usize::from(piece)) * 8;
            fast[i..i + 8].copy_from_slice(&bits.to_le_bytes());
        }
    }
    let validation = encoding::split_key(&fast).is_multiple_of(cfg.val_mod);
    if (cfg.split == "train" && validation) || (cfg.split == "val" && !validation) {
        stats.reject("other_split");
        return None;
    }
    let board = match entry
        .pos
        .fen()
        .ok()
        .and_then(|fen| Board::from_fen(&fen).ok())
    {
        Some(board) => board,
        None => {
            stats.reject("nonstandard_or_invalid_fen");
            return None;
        }
    };
    if board.in_check(board.side_to_move) {
        stats.reject("check");
        return None;
    }
    if !board.has_legal_move() || board.is_insufficient_material() || board.halfmove_clock >= 100 {
        stats.reject("terminal_or_draw");
        return None;
    }
    if board.parse_uci_move(&entry.mv.as_uci()).is_err() {
        stats.reject("invalid_teacher_move");
        return None;
    }
    // Depth is absent in binpack. Zero means unknown, not game ply or a fake depth.
    let mut record = encoding::encode(&board, score as i16, 0);
    if cfg.with_results {
        if !(-1..=1).contains(&entry.result) {
            stats.reject("invalid_result");
            return None;
        }
        // Preserve 100-byte records. High bit means a known side-to-move
        // outcome; low bits encode loss=0, draw=1, win=2. Zero stays unknown.
        record[99] = 0x80 | (entry.result + 1) as u8;
    }
    debug_assert_eq!(&record[..96], &fast[..96]);
    Some(record)
}

fn frame(out: &mut impl Write, records: &mut Vec<[u8; 100]>) -> io::Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    out.write_all(&(records.len() as u32).to_le_bytes())?;
    for record in records.iter() {
        out.write_all(record)?;
    }
    records.clear();
    out.flush()
}
fn run(cfg: &Config, out: &mut impl Write) -> Result<Stats, String> {
    let mut chunks = index(&cfg.files)?;
    if cfg.shuffle {
        let mut seed = cfg.seed;
        for i in (1..chunks.len()).rev() {
            let j = (random(&mut seed) % (i as u64 + 1)) as usize;
            chunks.swap(i, j);
        }
    }
    if !cfg.inspect {
        out.write_all(if cfg.with_results {
            MAGIC_RESULTS
        } else {
            MAGIC
        })
        .map_err(|e| e.to_string())?;
    }
    let mut stats = Stats::default();
    let mut records = Vec::with_capacity(cfg.batch);
    let mut payload = Vec::new();
    let mut current: Option<(usize, File)> = None;
    'chunks: for chunk in chunks {
        if current.as_ref().is_none_or(|(i, _)| *i != chunk.file) {
            current = Some((
                chunk.file,
                File::open(&cfg.files[chunk.file]).map_err(|e| e.to_string())?,
            ));
        }
        let file = &mut current.as_mut().unwrap().1;
        file.seek(SeekFrom::Start(chunk.offset))
            .map_err(|e| e.to_string())?;
        payload.resize(chunk.len as usize, 0);
        file.read_exact(&mut payload).map_err(|e| e.to_string())?;
        stats.chunks += 1;
        let mut reader = ChunkReader::default();
        while reader.has_next(&payload) {
            // Malformed compressed payloads can panic inside the upstream crate.
            // Process isolation + no success footer makes this a failed epoch.
            let entry = reader.next(&payload);
            stats.decoded += 1;
            if cfg.inspect {
                let value = serde_json::json!({"fen": entry.pos.fen().map_err(|e| format!("{e:?}"))?,
                    "move": entry.mv.as_uci(), "score_stm": entry.score, "ply": entry.ply, "result_stm": entry.result});
                serde_json::to_writer(&mut *out, &value).map_err(|e| e.to_string())?;
                out.write_all(b"\n").map_err(|e| e.to_string())?;
                stats.emitted += 1;
            } else if let Some(record) = candidate(&entry, cfg, &mut stats) {
                records.push(record);
                stats.emitted += 1;
                if records.len() == cfg.batch {
                    frame(out, &mut records).map_err(|e| e.to_string())?;
                }
            }
            if cfg.limit != 0 && stats.emitted >= cfg.limit {
                stats.limited = true;
                break 'chunks;
            }
        }
    }
    if !cfg.inspect {
        frame(out, &mut records).map_err(|e| e.to_string())?;
        let footer = serde_json::to_vec(&stats).map_err(|e| e.to_string())?;
        out.write_all(&0u32.to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&(footer.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&footer).map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())?;
    Ok(stats)
}

// Bounded fixture packer for tests/cross-checks, not a bulk data converter.
fn pack_fixture(path: PathBuf) -> Result<(), String> {
    let mut writer =
        sfbinpack::CompressedTrainingDataEntryWriter::new_in_memory().map_err(|e| e.to_string())?;
    for (i, line) in io::stdin().lock().lines().enumerate() {
        if i >= 8192 {
            return Err("fixture limit is 8192 records".into());
        }
        let line = line.map_err(|e| e.to_string())?;
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 4 {
            return Err("fixture rows: FEN<TAB>UCI<TAB>score_stm<TAB>result_stm".into());
        }
        let board = Board::from_fen(fields[0])?;
        let mv = board.parse_uci_move(fields[1])?;
        let mut pos = Position::from_fen(fields[0]).map_err(|e| format!("{e:?}"))?;
        let from = Square::new(u32::from(mv.from));
        let mut to = Square::new(u32::from(mv.to));
        let mut kind = MoveType::Normal;
        let mut promoted = SfPiece::none();
        if let Some(p) = mv.promotion {
            kind = MoveType::Promotion;
            let piece = match p {
                Piece::Knight => PieceType::Knight,
                Piece::Bishop => PieceType::Bishop,
                Piece::Rook => PieceType::Rook,
                Piece::Queen => PieceType::Queen,
                _ => return Err("bad promotion".into()),
            };
            promoted = SfPiece::new(piece, pos.side_to_move());
        } else if board.piece_at(mv.from).unwrap().1 == Piece::King && mv.from.abs_diff(mv.to) == 2
        {
            kind = MoveType::Castle;
            to = Square::new(u32::from(if mv.to > mv.from {
                mv.from + 3
            } else {
                mv.from - 4
            }));
        } else if board.piece_at(mv.from).unwrap().1 == Piece::Pawn
            && mv.from % 8 != mv.to % 8
            && board.piece_at(mv.to).is_none()
        {
            kind = MoveType::EnPassant;
        }
        let score = fields[2].parse::<i16>().map_err(|e| e.to_string())?;
        let result = fields[3].parse::<i16>().map_err(|e| e.to_string())?;
        if !(-1..=1).contains(&result) {
            return Err("bad fixture result".into());
        }
        let ply = u16::try_from(
            (board.fullmove_number - 1) * 2 + u32::from(board.side_to_move == Color::Black),
        )
        .map_err(|_| "fixture ply overflow")?;
        pos.set_ply(ply);
        writer
            .write_entry(&TrainingDataEntry {
                pos,
                mv: SfMove::new(from, to, kind, promoted),
                score,
                ply,
                result,
            })
            .map_err(|e| e.to_string())?;
    }
    let bytes = writer.into_bytes().map_err(|e| e.to_string())?;
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let result = if args.get(1).is_some_and(|s| s == "pack-fixture") && args.len() == 3 {
        pack_fixture(args[2].clone().into())
    } else {
        parse().and_then(|cfg| {
            let stdout = io::stdout();
            let mut output = BufWriter::new(stdout.lock());
            run(&cfg, &mut output).map(|_| ())
        })
    };
    if let Err(error) = result {
        eprintln!("binpack: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sentinels_are_rejected_before_scaling() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut entry = TrainingDataEntry {
            pos,
            mv: SfMove::normal(Square::new(12), Square::new(28)),
            score: 32002,
            ply: 0,
            result: 0,
        };
        let cfg = Config {
            split: "all".into(),
            denominator: 100,
            ..Config::default()
        };
        let mut stats = Stats::default();
        for value in [32002, 32000, 31999, 31000, -31999, i16::MIN] {
            entry.score = value;
            assert!(candidate(&entry, &cfg, &mut stats).is_none());
        }
        entry.score = -1234;
        let record = candidate(&entry, &cfg, &mut stats).unwrap();
        assert_eq!(i16::from_le_bytes(record[96..98].try_into().unwrap()), -12);
        assert_eq!(record[98], 0);
    }
}
