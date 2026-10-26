use zenith::board::{Board, Move, Piece};

const VALUE: [i32; 6] = [100, 320, 335, 500, 950, 20_000];

#[test]
fn exchange_fixtures_cover_pins_kings_xrays_and_special_moves() {
    for (fen, played, expected) in [
        ("4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", 100),
        ("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", -850),
        ("4k3/8/4p3/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0),
        ("4k3/4n3/8/3p4/2B5/8/8/4R1K1 w - - 0 1", "c4d5", 100),
        ("8/8/4k3/3p4/2B5/8/8/3R2K1 w - - 0 1", "c4d5", 100),
        ("8/8/4k3/3p4/2B5/8/8/6K1 w - - 0 1", "c4d5", -235),
        ("3rk3/8/8/3p4/3R4/8/8/3RK3 w - - 0 1", "d4d5", 100),
        ("3r3k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6", 0),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q", 850),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n", 220),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q", -100),
        ("1r5k/P7/8/8/8/8/8/7K w - - 0 1", "a7b8q", 1350),
        ("7k/8/8/8/8/7K/pR6/1b6 w - - 0 1", "b2b1", -1015),
        ("7k/8/8/8/8/7K/pR6/1r6 w - - 0 1", "b2b1", -850),
        ("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1", "d1d5", -950),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 0),
    ] {
        let board = Board::from_fen(fen).unwrap();
        let before = board;
        assert_eq!(
            board.see(board.parse_uci_move(played).unwrap()),
            expected,
            "{played}: {fen}"
        );
        assert_eq!(board, before);
        for threshold in [
            i32::MIN,
            -1000,
            -80,
            -30,
            0,
            1,
            1000,
            i32::MAX,
            expected - 1,
            expected,
            expected + 1,
        ] {
            assert_eq!(
                board.see_ge(board.parse_uci_move(played).unwrap(), threshold),
                expected >= threshold,
                "{played}, threshold={threshold}: {fen}"
            );
        }
        // Flip ranks and exchange colours to exercise Black's pawn direction,
        // promotions, en passant, and castling with the same material answer.
        let fields: Vec<_> = fen.split_whitespace().collect();
        let swap_case = |text: &str| {
            text.chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else {
                        c.to_ascii_uppercase()
                    }
                })
                .collect::<String>()
        };
        let placement = fields[0]
            .split('/')
            .rev()
            .map(swap_case)
            .collect::<Vec<_>>()
            .join("/");
        let ep = if fields[3] == "-" {
            "-".into()
        } else {
            format!(
                "{}{}",
                fields[3].as_bytes()[0] as char,
                (b'9' - (fields[3].as_bytes()[1] - b'0')) as char
            )
        };
        let mirrored = Board::from_fen(&format!(
            "{placement} b {} {ep} {} {}",
            swap_case(fields[2]),
            fields[4],
            fields[5]
        ))
        .unwrap();
        let original = board.parse_uci_move(played).unwrap();
        let mv = Move {
            from: original.from ^ 56,
            to: original.to ^ 56,
            promotion: original.promotion,
        };
        assert!(mirrored.legal_moves().contains(&mv));
        assert_eq!(mirrored.see(mv), expected, "mirrored {played}: {fen}");
        for threshold in [
            i32::MIN,
            -1000,
            -80,
            -30,
            0,
            1,
            1000,
            i32::MAX,
            expected - 1,
            expected,
            expected + 1,
        ] {
            assert_eq!(
                mirrored.see_ge(mv, threshold),
                expected >= threshold,
                "mirrored {played}, threshold={threshold}: {fen}"
            );
        }
    }
}

// Independent reference: generate all legal moves, then choose the legal LVA.
// Deliberately does not use attackers_to, SEE's bitboard walk, or swap folding.
fn gain(board: &Board, mv: Move) -> i32 {
    let mover = board.piece_at(mv.from).unwrap().1;
    board.piece_at(mv.to).map_or_else(
        || {
            if mover == Piece::Pawn && mv.from % 8 != mv.to % 8 {
                100
            } else {
                0
            }
        },
        |(_, piece)| VALUE[piece.index()],
    ) + mv.promotion.map_or(0, |piece| VALUE[piece.index()] - 100)
}

fn reply(board: &Board, target: u8) -> i32 {
    let next = board
        .legal_moves()
        .into_iter()
        .filter(|mv| {
            mv.to == target
                && board.piece_at(target).is_some()
                && (mv.promotion.is_none() || mv.promotion == Some(Piece::Queen))
        })
        .min_by_key(|mv| (VALUE[board.piece_at(mv.from).unwrap().1.index()], mv.from));
    next.map_or(0, |mv| {
        (gain(board, mv) - reply(&board.make_move(mv), target)).max(0)
    })
}

#[test]
fn bitboard_see_agrees_with_slow_legal_exchange_reference() {
    for (_, fen) in zenith::bench::POSITIONS {
        let board = Board::from_fen(fen).unwrap();
        for mv in board
            .legal_moves()
            .into_iter()
            .filter(|mv| board.piece_at(mv.to).is_some() || mv.promotion.is_some())
        {
            let expected = gain(&board, mv) - reply(&board.make_move(mv), mv.to);
            assert_eq!(board.see(mv), expected, "{mv}: {fen}");
        }
    }
}
