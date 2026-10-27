//! Synthetic weights test arithmetic and state, not playing strength.
use zenith::{
    board::{Board, Color},
    nnue::{Backend, Network, v3},
};

fn model() -> Network {
    let mut p = v3::Parameters::default();
    p.feature_bias.fill(130);
    for (i, w) in p.feature_weights.iter_mut().enumerate() {
        *w = (((i.wrapping_mul(17) ^ (i >> 9)) % 17) as i16) - 8;
    }
    for (b, h) in p.heads.iter_mut().enumerate() {
        for (i, w) in h.l1_weights.iter_mut().enumerate() {
            *w = (i % 19) as i16 - 9;
        }
        h.l1_bias.fill(6000);
        for (i, w) in h.l2_weights.iter_mut().enumerate() {
            *w = (i % 29) as i16 - 14;
        }
        h.l2_bias.fill(5000);
        for (i, w) in h.output_weights.iter_mut().enumerate() {
            *w = (i % 23) as i16 - 11;
        }
        h.output_bias = b as i32 * 100 - 400;
    }
    Network::new_v3(p).unwrap()
}

#[test]
fn canonical_mapping_covers_all_banks_and_mirror_contexts() {
    let mut seen = [false; 32];
    for rank in 0..8 {
        for file in 0..8 {
            // King alone on each rank/file; the other king is always distant.
            let white_rank = format!(
                "{}K{}",
                if file == 0 {
                    String::new()
                } else {
                    file.to_string()
                },
                if file == 7 {
                    String::new()
                } else {
                    (7 - file).to_string()
                }
            );
            let mut ranks = vec!["8".to_string(); 8];
            ranks[7 - rank] = white_rank;
            ranks[if rank < 4 { 0 } else { 7 }] = if file < 4 { "7k" } else { "k7" }.into();
            let board = Board::from_fen(&format!("{} w - - 0 1", ranks.join("/"))).unwrap();
            let canonical_file = file.min(7 - file);
            let expected = (rank / 2) * 4 + canonical_file + if file >= 4 { 16 } else { 0 };
            assert_eq!(v3::context(&board, Color::White), expected);
            seen[expected] = true;
        }
    }
    assert!(seen.into_iter().all(|x| x));
    for n in 2..=32 {
        assert_eq!(v3::output_bucket(n), ((n - 2) / 4).min(7) as usize);
    }
}

#[test]
fn incremental_special_moves_mirror_crossing_undo_and_simd_match_refresh() {
    let net = model();
    let scalar = Network::from_bytes(&net.to_bytes())
        .unwrap()
        .with_backend(Backend::Scalar)
        .unwrap();
    for (fen, sequence) in [
        ("7k/8/8/8/8/8/8/3K4 w - - 0 1", vec!["d1e1"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", vec!["e1g1", "e8c8"]),
        ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", vec!["e5d6"]),
        ("7k/P7/8/8/8/8/8/K7 w - - 0 1", vec!["a7a8n"]),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", vec!["a7b8q"]),
        ("7k/8/8/8/8/8/p7/1R5K b - - 0 1", vec!["a2b1n"]),
    ] {
        let start = Board::from_fen(fen).unwrap();
        let mut a = net.accumulator(&start);
        for mv in sequence {
            let next = a.board().make_move(a.board().parse_uci_move(mv).unwrap());
            a.update(&next);
            if mv == "d1e1" {
                assert_eq!(a.bucket_refreshes(), 1);
            }
            let fresh = scalar.accumulator(&next);
            for color in [Color::White, Color::Black] {
                assert_eq!(a.lanes(color), fresh.lanes(color));
            }
            assert_eq!(a.raw_score(), scalar.raw_reference(&next));
        }
        a.update(&start);
        assert_eq!(a.raw_score(), scalar.raw_reference(&start));
    }
    let mut a = net.accumulator(&Board::startpos());
    let mut seed = 77u64;
    for i in 0..512 {
        let fresh = scalar.accumulator(a.board());
        for color in [Color::White, Color::Black] {
            assert_eq!(a.lanes(color), fresh.lanes(color));
        }
        assert_eq!(a.raw_score(), fresh.raw_score());
        if i % 31 == 0 {
            assert_eq!(a.raw_score(), scalar.raw_reference(a.board()));
        }
        let moves = a.board().legal_moves();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let next = if moves.is_empty() || i % 128 == 127 {
            Board::startpos()
        } else {
            a.board()
                .make_move(moves[(seed >> 32) as usize % moves.len()])
        };
        a.update(&next);
    }
}

fn resign(bytes: &mut [u8]) {
    bytes[48..56].fill(0);
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    bytes[48..56].copy_from_slice(&hash.to_le_bytes());
}

#[test]
fn format_roundtrip_and_hostile_headers_and_last_bank_bounds() {
    let net = model();
    let bytes = net.to_bytes();
    assert_eq!(bytes.len(), v3::FILE_BYTES);
    let loaded = Network::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.format_version(), 3);
    assert_eq!(loaded.hidden_size(), 512);
    assert_eq!(bytes, loaded.to_bytes());
    for offset in [8, 36, 40, 56, 64, 76, 127] {
        let mut bad = bytes.clone();
        bad[offset] ^= 1;
        resign(&mut bad);
        assert!(Network::from_bytes(&bad).is_err(), "offset {offset}");
    }
    let mut bad = bytes.clone();
    bad.pop();
    assert!(Network::from_bytes(&bad).is_err());
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Network::from_bytes(&bad).is_err());
    let mut bad = bytes.clone();
    bad[v3::HEADER + 512 * 2 + 16 * 768 * 512 * 2 - 2..v3::HEADER + 512 * 2 + 16 * 768 * 512 * 2]
        .copy_from_slice(&513i16.to_le_bytes());
    resign(&mut bad);
    assert!(Network::from_bytes(&bad).is_err());
    let mut bad = bytes.clone();
    let last = bad.len() - 4;
    bad[last..].copy_from_slice(&65537i32.to_le_bytes());
    resign(&mut bad);
    assert!(Network::from_bytes(&bad).is_err());
}

#[test]
fn exact_hand_computed_pair_and_layers_and_negative_truncation() {
    let mut p = v3::Parameters::default();
    p.feature_bias.fill(128);
    // Pair=floor(128*128/255)=64; L1=64, L2=64; raw=-65*64-1=-4161.
    for h in &mut p.heads {
        h.l1_weights[0] = 64;
        h.l2_weights[0] = 64;
        h.output_weights[0] = -65;
        h.output_bias = -1;
    }
    let net = Network::new_v3(p).unwrap();
    let a = net.accumulator(&Board::startpos());
    assert_eq!(a.raw_score(), -4161);
    assert_eq!(a.evaluate(), -101);
}

#[test]
fn maximum_parameters_keep_all_simd_and_scalar_sums_bounded() {
    for sign in [1, -1] {
        let mut p = v3::Parameters::default();
        p.feature_bias.fill(4096 * sign);
        p.feature_weights.fill(512 * sign);
        for h in &mut p.heads {
            h.l1_weights.fill(127 * sign);
            h.l2_weights.fill(127 * sign);
            h.output_weights.fill(127 * sign);
            h.l1_bias.fill(65536 * i32::from(sign));
            h.l2_bias.fill(65536 * i32::from(sign));
            h.output_bias = 65536 * i32::from(sign);
        }
        let net = Network::new_v3(p).unwrap();
        let scalar = Network::from_bytes(&net.to_bytes())
            .unwrap()
            .with_backend(Backend::Scalar)
            .unwrap();
        let a = net.accumulator(&Board::startpos());
        assert!(a.lanes(Color::White).iter().all(|&v| v == 20480 * sign));
        let expected = if sign > 0 { 1101856 } else { -65536 };
        assert_eq!(a.raw_score(), expected);
        assert_eq!(scalar.raw_reference(a.board()), expected);
        assert!(a.evaluate().abs() < 28000);
    }
}
