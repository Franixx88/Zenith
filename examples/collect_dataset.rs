//! Stage 1 only: sampled game outcomes, not teacher evaluations or a trained NNUE.
use zenith::dataset::Dataset;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: collect_dataset OUTPUT_PREFIX LOG [LOG ...]".into());
    }
    let mut data = Dataset::default();
    let mut sources = String::new();
    for path in &args[1..] {
        let log = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        data.add_log(&log)?;
        let hash = log.bytes().fold(0xcbf29ce484222325u64, |h, c| {
            (h ^ u64::from(c)).wrapping_mul(0x100000001b3)
        });
        sources.push_str(&format!("source={path}\nsource_fnv64={hash:016x}\n"));
    }
    for (suffix, content) in [
        ("train.csv", data.csv(false)),
        ("validation.csv", data.csv(true)),
        (
            "manifest.txt",
            format!(
                "format=zenith-outcomes-v1\nengine={}\ngames={}\ntrain_positions={}\nvalidation_positions={}\nexcluded_ply_limit_games={}\nquarantined_cross_split_positions={}\nlabel=mean_white_game_outcome\nteacher=none\nsplit=FNV1a(opening_family)%5==0 validation\nsample=quiet non-check positions every 4 plies after max(16,opening+8)\n",
                env!("CARGO_PKG_VERSION"),
                data.games,
                data.count(false),
                data.count(true),
                data.excluded_adjudications,
                data.cross_split_positions
            ) + &sources,
        ),
    ] {
        std::fs::write(format!("{}-{suffix}", args[0]), content).map_err(|e| e.to_string())?;
    }
    println!(
        "{} games, {} training / {} validation positions; outcome labels only",
        data.games,
        data.count(false),
        data.count(true)
    );
    Ok(())
}
