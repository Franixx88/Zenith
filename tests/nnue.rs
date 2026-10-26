//! Model arithmetic and state-machine regressions; weights are synthetic.
use zenith::{
    board::{Board, Color, Piece},
    nnue::*,
};

fn model() -> Network {
    let mut p = Parameters::default();
    let mut state = 0x42a1_783du32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for value in &mut p.feature_bias {
        *value = (next() % 601) as i16 - 200;
    }
    for value in &mut p.feature_weights {
        *value = (next() % 65) as i16 - 32;
    }
    for value in &mut p.output_weights {
        *value = (next() % 257) as i16 - 128;
    }
    p.output_bias = -12345;
    Network::new(p).unwrap()
}

#[test]
fn scalar_and_avx2_match_every_lane_on_legal_play_and_extreme_models() {
    if !Backend::Avx2.available() {
        return;
    }
    for extreme in [false, true] {
        let network = if extreme {
            let mut p = Parameters::default();
            p.feature_bias.fill(-MAX_FEATURE_BIAS);
            for (i, w) in p.feature_weights.iter_mut().enumerate() {
                *w = if i % 3 == 0 {
                    MAX_FEATURE_WEIGHT
                } else {
                    -MAX_FEATURE_WEIGHT
                };
            }
            p.output_weights.fill(-MAX_OUTPUT_WEIGHT);
            Network::new(p).unwrap()
        } else {
            model()
        };
        let scalar = Network::from_bytes(&network.to_bytes())
            .unwrap()
            .with_backend(Backend::Scalar)
            .unwrap();
        let avx2 = network.with_backend(Backend::Avx2).unwrap();
        let mut a = scalar.accumulator(&Board::startpos());
        let mut b = avx2.accumulator(&Board::startpos());
        let mut seed = 42u64;
        for ply in 0..2048 {
            for color in [Color::White, Color::Black] {
                assert_eq!(a.lanes(color), b.lanes(color));
            }
            assert_eq!(a.raw_score(), b.raw_score());
            assert_eq!(a.evaluate(), b.evaluate());
            if ply % 61 == 0 {
                assert_eq!(b.raw_score(), avx2.raw_reference(b.board()));
            }
            let moves = a.board().legal_moves();
            let next = if moves.is_empty() || ply % 256 == 255 {
                Board::startpos()
            } else {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                a.board()
                    .make_move(moves[(seed >> 32) as usize % moves.len()])
            };
            assert_eq!(a.update(&next), b.update(&next));
        }
    }
}

fn check(model: &Network, accumulator: &Accumulator<'_>, dense: bool) {
    let refreshed = model.accumulator(accumulator.board());
    for color in [Color::White, Color::Black] {
        assert_eq!(
            accumulator.lanes(color),
            refreshed.lanes(color),
            "{}",
            accumulator.board().to_fen()
        );
    }
    assert_eq!(accumulator.raw_score(), refreshed.raw_score());
    if dense {
        assert_eq!(
            accumulator.raw_score(),
            model.raw_reference(accumulator.board())
        );
        assert_eq!(
            accumulator.evaluate(),
            model.evaluate_reference(accumulator.board())
        );
    }
}

#[test]
fn feature_mapping_is_bijective_and_rank_color_relative() {
    for perspective in [Color::White, Color::Black] {
        let mut seen = [false; INPUTS];
        for color in [Color::White, Color::Black] {
            for piece in [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ] {
                for square in 0..64 {
                    let index = feature_index(perspective, color, piece, square);
                    assert!(!seen[index]);
                    seen[index] = true;
                    assert_eq!(
                        index,
                        feature_index(perspective.other(), color.other(), piece, square ^ 56)
                    );
                }
            }
        }
        assert!(seen.iter().all(|on| *on));
    }
    assert_eq!(feature_index(Color::White, Color::White, Piece::Pawn, 8), 8);
    assert_eq!(
        feature_index(Color::Black, Color::White, Piece::Pawn, 8),
        432
    );
    assert_eq!(
        feature_index(Color::Black, Color::Black, Piece::King, 60),
        324
    );
}

#[test]
fn four_field_training_fens_do_not_relax_the_playing_parser() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
    assert!(Board::from_fen(fen).is_err());
    assert_eq!(parse_training_fen(fen).unwrap(), Board::startpos());
    assert!(parse_training_fen("8/8/8/8/8/8/8/8 w - -").is_err());
    assert!(parse_training_fen(&format!("{fen} 0")).is_err());
    // Variant/editor positions with impossible piece counts remain rejected.
    assert!(
        parse_training_fen(
            "1B1B1B1B/1kB1B1B1/1B1B1B1B/P1B1B1B1/PB1B1B1B/P1B1BKB1/PB1B1B1B/B1B1B1B1 w - -"
        )
        .is_err()
    );
}

#[test]
fn fixed_goldens_from_independent_node_export_and_forward_match_exactly() {
    let network = model();
    let mut accumulator = network.accumulator(&Board::startpos());
    let mut count = 0;
    for line in include_str!("data/nnue-v1-golden.tsv")
        .trim_start_matches('\u{feff}')
        .lines()
    {
        let columns: Vec<_> = line.split('\t').collect();
        assert_eq!(columns.len(), 3);
        let board = Board::from_fen(columns[0]).unwrap();
        let raw = columns[1].parse::<i64>().unwrap();
        let cp = columns[2].parse::<i32>().unwrap();
        accumulator.update(&board);
        assert_eq!(accumulator.raw_score(), raw, "{}", columns[0]);
        assert_eq!(accumulator.evaluate(), cp, "{}", columns[0]);
        assert_eq!(network.raw_reference(&board), raw);
        count += 1;
    }
    assert_eq!(count, 22);
}

#[test]
fn hand_computed_forward_uses_stm_order_clipping_and_signed_division() {
    let mut p = Parameters::default();
    // White king a1: white lane0=300 ->255, black lane0=-20 ->0.
    p.feature_weights[feature_index(Color::White, Color::White, Piece::King, 0) * HIDDEN] = 300;
    p.feature_weights[feature_index(Color::Black, Color::White, Piece::King, 0) * HIDDEN] = -20;
    p.output_weights[0] = 64;
    p.output_weights[HIDDEN] = -32;
    p.output_bias = -1;
    let net = Network::new(p).unwrap();
    let mut board = Board::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let mut a = net.accumulator(&board);
    assert_eq!(a.raw_score(), 255 * 64 - 1);
    assert_eq!(a.evaluate(), 399);
    board.side_to_move = Color::Black;
    assert_eq!(a.update(&board), FeatureChanges::default());
    assert_eq!(a.raw_score(), -255 * 32 - 1);
    assert_eq!(a.evaluate(), -200);
    check(&net, &a, true);
}

#[test]
fn all_special_moves_and_undo_match_full_refresh() {
    let net = model();
    let mut cases: Vec<(String, String, u32, u32)> = vec![];
    for (fen, mv, removed, added) in [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 2, 2),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", 2, 2),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", 2, 2),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", 2, 2),
        ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", 2, 1),
        ("7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1", "e4d3", 2, 1),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1a8", 2, 1),
        ("7k/8/8/8/8/8/8/K7 w - - 0 1", "a1b1", 1, 1),
    ] {
        cases.push((fen.into(), mv.into(), removed, added));
    }
    for suffix in ['q', 'r', 'b', 'n'] {
        for (fen, mv, removed) in [
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", format!("a7a8{suffix}"), 1),
            ("7k/8/8/8/8/8/p7/7K b - - 0 1", format!("a2a1{suffix}"), 1),
            ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", format!("a7b8{suffix}"), 2),
            ("7k/8/8/8/8/8/p7/1R5K b - - 0 1", format!("a2b1{suffix}"), 2),
        ] {
            cases.push((fen.into(), mv, removed, 1));
        }
    }
    for (fen, mv, removed, added) in cases {
        let board = Board::from_fen(&fen).unwrap();
        let child = board.make_move(board.parse_uci_move(&mv).unwrap());
        let mut a = net.accumulator(&board);
        let parent = a.clone();
        assert_eq!(a.update(&child), FeatureChanges { removed, added }, "{mv}");
        check(&net, &a, true);
        assert_eq!(
            a.update(&board),
            FeatureChanges {
                removed: added,
                added: removed
            }
        );
        assert_eq!(a.lanes(Color::White), parent.lanes(Color::White));
        assert_eq!(a.lanes(Color::Black), parent.lanes(Color::Black));
        assert_eq!(a.raw_score(), parent.raw_score());
    }
}

#[test]
fn branching_random_play_repetition_and_fen_resets_are_exact() {
    let net = model();
    let start = Board::startpos();
    let mut a = net.accumulator(&start);
    for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        let next = a.board().make_move(a.board().parse_uci_move(text).unwrap());
        a.update(&next);
        check(&net, &a, true);
    }
    assert_eq!(a.raw_score(), net.accumulator(&start).raw_score());
    let mut seed = 239u64;
    for _ in 0..32 {
        a.update(&start);
        for ply in 0..256 {
            let board = *a.board();
            let moves = board.legal_moves();
            if moves.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let child = board.make_move(moves[(seed >> 32) as usize % moves.len()]);
            // Re-search / sibling / undo: the parent must survive every child.
            let mut branch = a.clone();
            branch.update(&child);
            check(&net, &branch, ply % 37 == 0);
            branch.update(&board);
            assert_eq!(branch.raw_score(), a.raw_score());
            a.update(&child);
            check(&net, &a, false);
        }
    }
}

#[test]
fn bounded_extreme_weights_do_not_overflow_or_enter_mate_band() {
    for sign in [-1, 1] {
        let mut p = Parameters::default();
        p.feature_bias.fill(sign * MAX_FEATURE_BIAS);
        p.feature_weights.fill(sign * MAX_FEATURE_WEIGHT);
        p.output_weights.fill(MAX_OUTPUT_WEIGHT);
        p.output_bias = MAX_OUTPUT_BIAS;
        let net = Network::new(p).unwrap();
        let start = Board::startpos();
        let mut a = net.accumulator(&start);
        check(&net, &a, true);
        assert!(a.evaluate().abs() <= MAX_SCORE);
        let bare = Board::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        a.update(&bare);
        check(&net, &a, true);
        a.update(&start);
        check(&net, &a, true);
    }
    for value in [i16::MIN, MAX_FEATURE_WEIGHT + 1] {
        let mut p = Parameters::default();
        p.feature_weights[0] = value;
        assert!(Network::new(p).is_err());
    }
    let p = Parameters {
        output_bias: i32::MIN,
        ..Parameters::default()
    };
    assert!(Network::new(p).is_err());
}

#[test]
fn versioned_format_roundtrip_rejects_corruption_truncation_and_wrong_architecture() {
    let net = model();
    let bytes = net.to_bytes();
    assert_eq!(bytes.len(), 394820);
    assert_eq!(bytes.len(), FILE_BYTES);
    let loaded = Network::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.to_bytes(), bytes);
    assert_eq!(
        loaded.raw_reference(&Board::startpos()),
        net.raw_reference(&Board::startpos())
    );
    for length in [0, 8, 63, 64, FILE_BYTES - 1] {
        assert!(Network::from_bytes(&bytes[..length]).is_err());
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(Network::from_bytes(&extra).is_err());
    for offset in [
        0,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        48,
        56,
        64,
        1024,
        FILE_BYTES - 1,
    ] {
        let mut bad = bytes.clone();
        bad[offset] ^= 1;
        assert!(Network::from_bytes(&bad).is_err(), "offset {offset}");
    }
    // A valid checksum must not bypass parameter range validation.
    let mut bad = bytes;
    bad[64..66].copy_from_slice(&i16::MIN.to_le_bytes());
    bad[48..56].fill(0);
    let hash = bad.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    });
    bad[48..56].copy_from_slice(&hash.to_le_bytes());
    assert!(Network::from_bytes(&bad).is_err());
}

#[test]
fn float_export_scales_rounding_and_rejection_are_explicit() {
    let mut f = FloatParameters {
        feature_bias: vec![0.; HIDDEN],
        feature_weights: vec![0.; INPUTS * HIDDEN],
        output_weights: vec![0.; 2 * HIDDEN],
        output_bias: 0.5,
    };
    f.feature_bias[0] = 0.5;
    f.feature_weights[0] = 0.5;
    f.output_weights[0] = -0.5;
    let net = f.quantize().unwrap();
    assert_eq!(net.parameters().feature_bias[0], 128);
    assert_eq!(net.parameters().feature_weights[0], 128);
    assert_eq!(net.parameters().output_weights[0], -32);
    assert_eq!(net.parameters().output_bias, 8160);
    assert_eq!(net.accumulator(&Board::startpos()).evaluate(), 99); // (8160-128*32)*400/16320 truncates
    f.feature_bias[0] = -0.5;
    assert_eq!(f.quantize().unwrap().parameters().feature_bias[0], -128);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1000.] {
        f.feature_weights[1] = invalid;
        assert!(f.quantize().is_err());
    }
    f.feature_weights.pop();
    assert!(f.quantize().is_err());
}
