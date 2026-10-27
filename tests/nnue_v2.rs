//! Four king banks: quadrant crossings must never mix weight-bank sums.
use zenith::{
    board::{Board, Color},
    nnue::*,
};

fn model(buckets: usize) -> Network {
    let mut p = Parameters::default();
    p.feature_weights.resize(buckets * INPUTS * HIDDEN, 0);
    p.feature_bias.fill(31);
    for (i, value) in p.feature_weights.iter_mut().enumerate() {
        *value = ((i.wrapping_mul(131).wrapping_add(i / HIDDEN * 17)) % 61) as i16 - 30;
    }
    for (i, value) in p.output_weights.iter_mut().enumerate() {
        *value = ((i * 37) % 251) as i16 - 125;
    }
    p.output_bias = -379;
    Network::new(p).unwrap()
}

fn check(net: &Network, a: &Accumulator<'_>, dense: bool) {
    let fresh = net.accumulator(a.board());
    for color in [Color::White, Color::Black] {
        assert_eq!(a.lanes(color), fresh.lanes(color), "{}", a.board().to_fen());
    }
    assert_eq!(a.raw_score(), fresh.raw_score());
    if dense {
        assert_eq!(a.raw_score(), net.raw_reference(a.board()));
    }
}

#[test]
fn quadrant_mapping_is_rank_relative_and_has_no_file_mirroring() {
    for (fen, white, black) in [
        ("7k/8/8/8/8/8/8/K7 w - - 0 1", 0, 1),
        ("k7/8/8/8/8/8/8/7K w - - 0 1", 1, 0),
        ("8/8/8/K7/7k/8/8/8 w - - 0 1", 2, 3),
        ("8/8/8/7K/k7/8/8/8 w - - 0 1", 3, 2),
    ] {
        let b = Board::from_fen(fen).unwrap();
        assert_eq!(king_bucket(&b, Color::White), white);
        assert_eq!(king_bucket(&b, Color::Black), black);
    }
}

#[test]
fn king_crossings_castles_ep_promotion_and_undo_match_dense() {
    let net = model(4);
    for (fen, mv, refreshes) in [
        ("4k3/8/8/8/8/8/8/3K4 w - - 0 1", "d1e1", 1),
        ("7k/8/8/8/3K4/8/8/8 w - - 0 1", "d4d5", 1),
        ("4k3/8/8/8/8/8/8/K7 b - - 0 1", "e8d8", 1),
        ("8/8/8/3k4/8/8/8/K7 b - - 0 1", "d5d4", 1),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", 1),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 0),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", 1),
        ("7k/8/8/8/8/8/4p3/3K4 w - - 0 1", "d1e2", 1),
        ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", 0),
        ("7k/1P6/8/8/8/8/8/K7 w - - 0 1", "b7b8q", 0),
    ] {
        let board = Board::from_fen(fen).unwrap();
        let child = board.make_move(board.parse_uci_move(mv).unwrap());
        let mut a = net.accumulator(&board);
        a.update(&child);
        assert_eq!(a.bucket_refreshes(), refreshes, "{mv}");
        check(&net, &a, true);
        a.update(&board);
        assert_eq!(a.bucket_refreshes(), refreshes, "undo {mv}");
        check(&net, &a, true);
    }
}

#[test]
fn scalar_simd_refresh_and_arbitrary_resets_agree_on_legal_walk() {
    let scalar = model(4).with_backend(Backend::Scalar).unwrap();
    let native = Network::from_bytes(&scalar.to_bytes()).unwrap();
    let mut a = scalar.accumulator(&Board::startpos());
    let mut b = native.accumulator(&Board::startpos());
    let mut seed = 97u64;
    let mut refreshed = 0;
    for ply in 0..2048 {
        let parent = *a.board();
        let moves = parent.legal_moves();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let next = if moves.is_empty() || ply % 128 == 127 {
            Board::startpos()
        } else {
            parent.make_move(moves[(seed >> 32) as usize % moves.len()])
        };
        assert_eq!(a.update(&next), b.update(&next));
        refreshed += a.bucket_refreshes();
        for color in [Color::White, Color::Black] {
            assert_eq!(a.lanes(color), b.lanes(color));
        }
        assert_eq!(a.raw_score(), b.raw_score());
        check(&native, &b, ply % 47 == 0);
        if ply % 17 == 0 {
            a.update(&parent);
            b.update(&parent);
            check(&scalar, &a, true);
            check(&native, &b, true);
            a.update(&next);
            b.update(&next);
        }
    }
    assert!(refreshed > 0);
}

#[test]
fn cloned_v1_banks_preserve_the_existing_network_exactly() {
    let v1 = model(1);
    let p = v1.parameters();
    let v2 = Network::new(Parameters {
        feature_bias: p.feature_bias,
        feature_weights: p.feature_weights.repeat(4),
        output_weights: p.output_weights,
        output_bias: p.output_bias,
    })
    .unwrap();
    for line in include_str!("data/nnue-v1-fens.txt").lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let board = parse_training_fen(line.trim_start_matches('\u{feff}')).unwrap();
        assert_eq!(
            v1.accumulator(&board).raw_score(),
            v2.accumulator(&board).raw_score()
        );
    }
    assert_eq!(v1.format_version(), 1);
    assert_eq!(v2.format_version(), 2);
}

fn reseal(bytes: &mut [u8]) {
    bytes[48..56].fill(0);
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    bytes[48..56].copy_from_slice(&hash.to_le_bytes());
}

#[test]
fn v2_format_is_bounded_strict_and_checks_all_four_banks() {
    let net = model(4);
    let bytes = net.to_bytes();
    assert_eq!(bytes.len(), V2_FILE_BYTES);
    assert_eq!(bytes.len(), 1_574_468);
    assert_eq!(Network::from_bytes(&bytes).unwrap().to_bytes(), bytes);
    for len in [0, 8, 63, 64, FILE_BYTES, bytes.len() - 1] {
        assert!(Network::from_bytes(&bytes[..len]).is_err());
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(Network::from_bytes(&extra).is_err());
    for offset in [8, 12, 16, 20, 24, 28, 32, 36, 40, 56] {
        let mut bad = bytes.clone();
        bad[offset] ^= 1;
        reseal(&mut bad);
        assert!(Network::from_bytes(&bad).is_err(), "field {offset}");
    }
    let mut bad = bytes.clone();
    let offset = 64 + HIDDEN * 2 + (4 * INPUTS * HIDDEN - 1) * 2;
    bad[offset..offset + 2].copy_from_slice(&i16::MIN.to_le_bytes());
    reseal(&mut bad);
    assert!(Network::from_bytes(&bad).is_err());
    let mut extreme = Parameters {
        feature_weights: vec![MAX_FEATURE_WEIGHT; 4 * INPUTS * HIDDEN],
        ..Parameters::default()
    };
    extreme.feature_bias.fill(MAX_FEATURE_BIAS);
    extreme.output_weights.fill(MAX_OUTPUT_WEIGHT);
    let extreme = Network::new(extreme).unwrap();
    check(&extreme, &extreme.accumulator(&Board::startpos()), true);
}
