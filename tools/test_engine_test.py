"""Regression tests for the evidence gate, independent of engine strength."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import chess.pgn
from engine_test import opening_book, options, validate_pgn, sprt_decision, preflight


def mate_game(white, round_id="1", opening=None):
    game = chess.pgn.Game()
    game.headers.update(White=white, Black="baseline" if white == "candidate" else "candidate",
                        Round=round_id, Result="0-1", Termination="normal")
    node = game
    board = game.board()
    for token in opening or "a2a3 a7a6 h2h3 h7h6 f2f3 e7e5 g2g4 d8h4".split():
        move = board.parse_uci(token)
        board.push(move)
        node = node.add_variation(move)
    return str(game)


class EvidenceGate(unittest.TestCase):
    def test_external_engines_only_omit_safe_optional_shared_options(self):
        import chess.engine
        fake = MagicMock()
        fake.options = {"Hash": chess.engine.Option("Hash", "spin", 16, 1, 128, None)}
        fake.id = {"name": "single-thread-fixture"}
        with patch("chess.engine.SimpleEngine.popen_uci") as launch:
            launch.return_value.__enter__.return_value = fake
            settings = {"Hash":128,"Threads":1,"Move Overhead":50,"SyzygyPath":"<empty>"}
            result = preflight(Path("fixture.exe"),settings,external=True)
            self.assertEqual(result["options"],{"Hash":128})
            self.assertEqual(result["omitted_options"]["Threads"],1)
            fake.configure.assert_called_with({"Hash":128})
            for bad in [settings|{"Threads":2}, settings|{"EvalFiel":"typo"},settings|{"Hash":256},
                        settings|{"SyzygyPath":"tables"}]:
                with self.assertRaises((ValueError, chess.engine.EngineError)): preflight(Path("fixture.exe"),bad,external=True)
            with self.assertRaises(ValueError): preflight(Path("fixture.exe"),settings)

    def validate(self, games):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.pgn"
            path.write_text("\n\n".join(games) + "\n", encoding="utf-8")
            return validate_pgn(path)

    def test_pair_score_and_arrival_order(self):
        result = self.validate([mate_game("baseline"), mate_game("candidate")])
        self.assertTrue(result["valid"])
        self.assertEqual(result["pentanomial"], [0, 0, 1, 0, 0])
        self.assertEqual(result["wdl"], {"wins": 1, "losses": 1})

    def test_fault_and_partial_pair_are_rejected(self):
        self.assertFalse(self.validate([mate_game("candidate")])["valid"])
        faulty = mate_game("baseline").replace('[Result "0-1"]', '[Result "1-0"]')
        self.assertFalse(self.validate([mate_game("candidate"), faulty])["valid"])

    def test_duplicate_colors_and_mismatched_openings_rejected(self):
        self.assertFalse(self.validate([mate_game("candidate")] * 2)["valid"])
        other = "h2h3 h7h6 a2a3 a7a6 f2f3 e7e5 g2g4 d8h4".split()
        self.assertFalse(self.validate([mate_game("candidate"), mate_game("baseline", opening=other)])["valid"])

    def test_book_splits_are_legal_and_disjoint(self):
        source = Path(__file__).resolve().parents[1] / "data/openings.txt"
        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for split in ["calibration", "holdout"]:
                path = Path(directory) / f"{split}.pgn"
                self.assertEqual(opening_book(source, path, split), 64)
                positions = set()
                with path.open() as stream:
                    while (game := chess.pgn.read_game(stream)) is not None:
                        positions.add(game.end().board().fen().rsplit(" ", 2)[0])
                roots.append(positions)
            self.assertFalse(roots[0] & roots[1])

    def test_options_do_not_override_shared_resources(self):
        for item in ["Threads=8", "Hash=256", "SyzygyPath=local", "a", "=2"]:
            with self.assertRaises(ValueError):
                options([item])
        self.assertEqual(options(["Singular Extensions=false"]), {"Singular Extensions": "false"})

    def test_game_cap_does_not_accept_a_hypothesis(self):
        self.assertEqual(sprt_decision("Finished match", True), "inconclusive")
        self.assertEqual(sprt_decision("Finished match", False), "fixed_length")
        self.assertEqual(sprt_decision("SPRT completed - H1 was accepted", True), "H1")
        self.assertEqual(sprt_decision("H0 was accepted\nH1 was accepted", True), "H0")

    def test_only_normal_sprt_stop_may_preserve_a_singleton_outside_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.pgn"
            path.write_text("\n\n".join([mate_game("candidate"), mate_game("baseline"),
                mate_game("candidate", "2")]) + "\n", encoding="utf-8")
            self.assertFalse(validate_pgn(path)["valid"])
            result = validate_pgn(path, allow_singletons=True)
            self.assertTrue(result["valid"])
            self.assertEqual(result["pairs"], 1)
            self.assertEqual(len(result["unpaired_games"]), 1)
            self.assertEqual(sum(result["paired_wdl"].values()), 2)


if __name__ == "__main__":
    unittest.main()
