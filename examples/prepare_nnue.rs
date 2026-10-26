//! Stream labelled positions into compact, side-to-move-relative NNUE records.
//!
//! Input (stdin or `--input FILE`), one row per line, tab separated:
//!     fen <TAB> depth <TAB> cp <TAB> mate
//! `cp` is in WHITE's perspective (Lichess eval-cache convention); empty cp or a
//! non-empty `mate` means "no usable score" and the row is dropped. Positions go
//! through Zenith's strict training-FEN parser, so the rules are exactly the
//! engine's rules.
//!
//! Output directory:
//!     train-000.bin ... train-NNN.bin   shuffled by hash into --shards files
//!     val.bin                           ~1/--val-mod of the data, disjoint
//!     manifest.txt                      settings, counts and reject reasons
//!
//! Record (100 bytes, little endian):
//!     12 x u64 bitboards: own P,N,B,R,Q,K then opponent P,N,B,R,Q,K, where "own"
//!     is the side to move and the board is flipped vertically when Black moves
//!     (bit = relative square, a1 = 0). This IS the side-to-move perspective
//!     input of docs/nnue-v1.md; the opponent perspective is a fixed permutation.
//!     i16 score  centipawns from the side to move; outliers are rejected
//!     u8  depth  teacher depth (saturating)
//!     u8  flags  reserved, 0
//!
//! The split key is a hash of exactly what the network sees, so a position and
//! its colour-mirrored twin can never land in different splits.
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use zenith::board::Color;
mod nnue_data;
use zenith::nnue::parse_training_fen;

const REASONS: [&str; 8] = [
    "bad_row",
    "no_score_or_mate",
    "depth_below_min",
    "cp_above_max",
    "bad_fen",
    "side_to_move_in_check",
    "no_legal_moves",
    "winning_capture_available",
];

struct Config {
    out: PathBuf,
    input: Option<PathBuf>,
    min_depth: u32,
    max_cp: i32,
    shards: usize,
    val_mod: u64,
    keep_tactical: bool,
    max_rows: u64,
    source: String,
    score_pov: ScorePov,
    require_complete: bool,
    data_license: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScorePov {
    White,
    Stm,
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = Config {
        out: PathBuf::new(),
        input: None,
        min_depth: 0,
        max_cp: 3000,
        shards: 16,
        val_mod: 20,
        keep_tactical: false,
        max_rows: u64::MAX,
        source: String::from("unspecified"),
        score_pov: ScorePov::White,
        require_complete: false,
        data_license: "unspecified".into(),
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--keep-tactical" {
            config.keep_tactical = true;
            index += 1;
            continue;
        }
        if flag == "--require-complete" {
            config.require_complete = true;
            index += 1;
            continue;
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {flag}"))?;
        let number = |text: &str| text.parse::<u64>().map_err(|e| format!("{flag}: {e}"));
        match flag {
            "--out" => config.out = PathBuf::from(value),
            "--input" => config.input = Some(PathBuf::from(value)),
            "--min-depth" => {
                config.min_depth =
                    u32::try_from(number(value)?).map_err(|_| "min-depth exceeds u32")?
            }
            "--max-cp" => {
                let n = number(value)?;
                if !(1..=30000).contains(&n) {
                    return Err("max-cp must be 1..30000".into());
                }
                config.max_cp = n as i32;
            }
            "--shards" => {
                let n = number(value)?;
                if !(1..=512).contains(&n) {
                    return Err("shards must be 1..512".into());
                }
                config.shards = n as usize;
            }
            "--val-mod" => {
                config.val_mod = number(value)?;
                if config.val_mod < 2 {
                    return Err("val-mod must be >=2".into());
                }
            }
            "--max-rows" => {
                config.max_rows = number(value)?;
                if config.max_rows == 0 {
                    return Err("max-rows must be positive".into());
                }
            }
            "--source" => config.source = value.clone(),
            "--license" => config.data_license = value.clone(),
            "--score-pov" => {
                config.score_pov = match value.as_str() {
                    "white" => ScorePov::White,
                    "stm" => ScorePov::Stm,
                    _ => return Err("score-pov must be white or stm".into()),
                }
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
        index += 2;
    }
    if config.out.as_os_str().is_empty() {
        return Err(
            "usage: prepare_nnue --out DIR [--input FILE] [--min-depth N] [--max-cp N] \
             [--shards N] [--val-mod N] [--keep-tactical] [--max-rows N] [--source TEXT] [--license TEXT] [--score-pov white|stm] [--require-complete]"
                .into(),
        );
    }
    if config.source.contains(['\n', '\r']) || config.data_license.contains(['\n', '\r']) {
        return Err("source/license must be one line".into());
    }
    if config.require_complete && config.max_rows != u64::MAX {
        return Err("require-complete cannot be combined with max-rows".into());
    }
    Ok(config)
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    // Validate/open the source before claiming a fresh output directory.
    let reader: Box<dyn BufRead> = match &config.input {
        Some(path) => Box::new(BufReader::new(File::open(path).map_err(|e| e.to_string())?)),
        None => Box::new(BufReader::new(io::stdin())),
    };
    if let Some(parent) = config.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(&config.out).map_err(|e| {
        format!("output must be a new directory; existing/partial datasets are preserved: {e}")
    })?;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    fs::write(
        config.out.join("INCOMPLETE"),
        b"Not committed: do not train on this directory\n",
    )
    .map_err(|e| e.to_string())?;
    let create = |name: String| {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(config.out.join(name))
            .map(|f| BufWriter::with_capacity(1 << 20, f))
            .map_err(|e| e.to_string())
    };
    let mut train: Vec<_> = (0..config.shards)
        .map(|i| create(format!("train-{i:03}.bin")))
        .collect::<Result<_, _>>()?;
    let mut validation = create("val.bin".into())?;

    let mut rejected = [0u64; REASONS.len()];
    let (mut rows, mut kept, mut kept_validation) = (0u64, 0u64, 0u64);
    let mut complete = false;
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        if complete {
            return Err("data after TSV completion marker".into());
        }
        if let Some(count) = line.strip_prefix("#zenith-tsv-complete\t") {
            if count
                .trim()
                .parse::<u64>()
                .map_err(|_| "bad TSV completion count")?
                != rows
            {
                return Err("TSV completion count mismatch".into());
            }
            complete = true;
            continue;
        }
        rows += 1;
        if rows % 5_000_000 == 0 {
            eprintln!("rows {rows} kept {kept}");
        }
        match convert(&line, &config) {
            Err(reason) => rejected[reason] += 1,
            Ok((record, key)) => {
                let validation_row = key % config.val_mod == 0;
                if validation_row {
                    validation.write_all(&record).map_err(|e| e.to_string())?;
                    kept_validation += 1;
                } else {
                    let shard = (splitmix(key) % config.shards as u64) as usize;
                    train[shard].write_all(&record).map_err(|e| e.to_string())?;
                }
                kept += 1;
                if kept >= config.max_rows {
                    break;
                }
            }
        }
    }
    if config.require_complete && !complete {
        return Err("TSV producer did not finish; dataset remains INCOMPLETE".into());
    }
    for writer in &mut train {
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
    }
    validation.flush().map_err(|e| e.to_string())?;
    validation.get_ref().sync_all().map_err(|e| e.to_string())?;
    if kept_validation == 0 || kept == kept_validation {
        return Err("train and validation must both be nonempty; increase input or adjust val-mod; partial output remains INCOMPLETE".into());
    }

    let mut manifest = String::new();
    manifest
        .push_str("format=zenith-nnue-records-v1\nrecord_bytes=100\nperspective=side_to_move\n");
    manifest
        .push_str("score=centipawns_side_to_move\nsplit=fnv1a(relative_bitboards)%val_mod==0\n");
    manifest.push_str(&format!("source={}\n", config.source));
    manifest.push_str(&format!("license={}\n", config.data_license));
    manifest.push_str(&format!(
        "dataset_id={created}-{}\ninput_score_pov={}\n",
        std::process::id(),
        if config.score_pov == ScorePov::White {
            "white"
        } else {
            "stm"
        }
    ));
    manifest.push_str(&format!(
        "min_depth={}\nmax_cp={}\nshards={}\nval_mod={}\nskip_tactical={}\n",
        config.min_depth, config.max_cp, config.shards, config.val_mod, !config.keep_tactical
    ));
    manifest.push_str(&format!(
        "rows_read={rows}\nrows_kept={kept}\nrows_validation={kept_validation}\n"
    ));
    for (name, count) in REASONS.iter().zip(rejected) {
        manifest.push_str(&format!("rejected_{name}={count}\n"));
    }
    let mut commit = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(config.out.join("manifest.txt.part"))
        .map_err(|e| e.to_string())?;
    commit
        .write_all(manifest.as_bytes())
        .map_err(|e| e.to_string())?;
    commit.sync_all().map_err(|e| e.to_string())?;
    drop(commit);
    // Removal before final atomic rename is safe: no manifest still means uncommitted.
    fs::remove_file(config.out.join("INCOMPLETE")).map_err(|e| e.to_string())?;
    fs::rename(
        config.out.join("manifest.txt.part"),
        config.out.join("manifest.txt"),
    )
    .map_err(|e| e.to_string())?;
    print!("{manifest}");
    Ok(())
}

/// Returns the 100-byte record and its split key, or the index of a reject reason.
fn convert(line: &str, config: &Config) -> Result<([u8; 100], u64), usize> {
    let mut fields = line.split('\t');
    let fen = fields.next().ok_or(0usize)?;
    let depth: u32 = fields
        .next()
        .ok_or(0usize)?
        .trim()
        .parse()
        .map_err(|_| 0usize)?;
    let cp_text = fields.next().ok_or(0usize)?.trim();
    let mate_text = fields.next().unwrap_or("").trim();
    if fields.next().is_some() {
        return Err(0);
    }
    if cp_text.is_empty() || !mate_text.is_empty() {
        return Err(1);
    }
    let cp_white: i32 = cp_text.parse().map_err(|_| 0usize)?;
    if depth < config.min_depth {
        return Err(2);
    }
    if cp_white.unsigned_abs() > config.max_cp as u32 {
        return Err(3);
    }
    let board = parse_training_fen(fen).map_err(|_| 4usize)?;
    let side = board.side_to_move;
    if board.in_check(side) {
        return Err(5);
    }
    if !board.has_legal_move() {
        return Err(6);
    }
    if !config.keep_tactical
        && board
            .legal_tactical_moves()
            .into_iter()
            .any(|mv| board.see(mv) > 0)
    {
        return Err(7);
    }
    let score = if side == Color::White || config.score_pov == ScorePov::Stm {
        cp_white
    } else {
        -cp_white
    };
    let record = nnue_data::encode(&board, score as i16, depth.min(255) as u8);
    let key = nnue_data::split_key(&record);
    Ok((record, key))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        Config {
            out: PathBuf::new(),
            input: None,
            min_depth: 0,
            max_cp: 3000,
            shards: 2,
            val_mod: 5,
            keep_tactical: true,
            max_rows: u64::MAX,
            source: "synthetic unit test".into(),
            score_pov: ScorePov::White,
            require_complete: false,
            data_license: "synthetic unit test".into(),
        }
    }
    #[test]
    fn original_white_black_labels_and_mirrored_inputs_match() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
        let cfg = config();
        let white = convert(&format!("{fen} w KQkq - 0 1\t24\t123\t"), &cfg).unwrap();
        let black = convert(&format!("{fen} b KQkq - 0 1\t24\t-123\t"), &cfg).unwrap();
        assert_eq!(white, black);
        assert_eq!(i16::from_le_bytes(black.0[96..98].try_into().unwrap()), 123);
        let mut cfg = cfg;
        cfg.score_pov = ScorePov::Stm;
        let black_stm = convert(&format!("{fen} b KQkq - 0 1\t24\t123\t"), &cfg).unwrap();
        assert_eq!(white, black_stm);
    }
    #[test]
    fn network_input_split_ignores_castling_and_extreme_labels_are_rejected() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w";
        let cfg = config();
        let with_rights = convert(&format!("{fen} KQkq - 0 1\t24\t123\t"), &cfg).unwrap();
        let without = convert(&format!("{fen} - - 0 1\t24\t123\t"), &cfg).unwrap();
        assert_eq!(with_rights, without);
        for score in [i32::MIN, 32002, -32002, 3001] {
            assert_eq!(
                convert(&format!("{fen} - - 0 1\t24\t{score}\t"), &cfg),
                Err(3)
            );
        }
        assert_eq!(convert(&format!("{fen} - - 0 1\t24\t\t1"), &cfg), Err(1));
    }
}
