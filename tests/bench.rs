//! A neural benchmark must exercise the search evaluator, not just load a file.
use std::process::Command;
use zenith::nnue::{Network, Parameters};

#[test]
fn neural_bench_uses_model_and_rejects_bad_input_before_search() {
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let path = std::env::temp_dir().join(format!("zenith bench model {unique}.znnue"));
    let mut parameters = Parameters::default();
    parameters.feature_bias.fill(128);
    Network::new(parameters).unwrap().save(&path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_zenith"))
        .args(["bench", "1", "--eval-file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("evaluation=nnue-v1 backend="));
    let evaluations: Vec<u64> = text
        .lines()
        .filter(|line| line.starts_with("nnue refreshes="))
        .map(|line| line.rsplit_once("evaluations=").unwrap().1.parse().unwrap())
        .collect();
    assert_eq!(evaluations.len(), zenith::bench::POSITIONS.len());
    assert!(evaluations.iter().all(|&count| count > 0));
    std::fs::write(&path, b"corrupt model").unwrap();
    let bad = Command::new(env!("CARGO_BIN_EXE_zenith"))
        .args(["bench", "1", "--eval-file"])
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(!bad.status.success());
    assert!(bad.stdout.is_empty());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("NNUE"));
    let missing = Command::new(env!("CARGO_BIN_EXE_zenith"))
        .args(["bench", "--eval-file"])
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(missing.stdout.is_empty());
}
