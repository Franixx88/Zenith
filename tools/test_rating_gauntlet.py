"""Rating input semantics: exclude scouts, preserve colors/families, parse Ordo."""
from pathlib import Path
import tempfile
import unittest
import hashlib
from unittest.mock import patch

import chess.pgn
import run_rating_gauntlet as R
from test_engine_test import mate_game


class RatingEvidenceTests(unittest.TestCase):
    def test_adapter_preserves_original_and_pins_both_launcher_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/"binaries").mkdir()
            original=folder/"binaries/minimalchess-0.6.exe"
            original.write_bytes(b"original engine fixture")
            proxy=folder/"proxy.exe";proxy.write_bytes(b"adapter fixture")
            digest=hashlib.sha256(original.read_bytes()).hexdigest()
            plan={"frozen_sha256":{"binaries/minimalchess-0.6.exe":digest}}
            with patch.object(R,"MINIMALCHESS_06_SHA",digest):
                R.install_minimalchess_adapter(folder,plan,proxy)
                R.install_minimalchess_adapter(folder,plan,proxy)
                self.assertEqual(original.read_bytes(),b"original engine fixture")
                for name,sha in plan["frozen_sha256"].items():
                    self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha)
                self.assertEqual(R.opponent_binary(folder,plan,{"id":"minimalchess-0.6"}),
                                 folder/"binaries/minimalchess-0.6-uci.exe")
                self.assertEqual(R.opponent_binary(folder,plan,{"id":"weiss-1.0"}),
                                 folder/"binaries/weiss-1.0.exe")
                proxy.write_bytes(b"changed adapter")
                with self.assertRaisesRegex(ValueError,"refusing to overwrite"):
                    R.install_minimalchess_adapter(folder,plan,proxy)

    def test_adapter_rejects_unreviewed_engine_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/"binaries").mkdir()
            (folder/"binaries/minimalchess-0.6.exe").write_bytes(b"different engine")
            with self.assertRaisesRegex(ValueError,"pinned MinimalChess"):
                R.install_minimalchess_adapter(folder,{"frozen_sha256":{}},folder/"missing.exe")
            self.assertFalse((folder/"binaries/minimalchess-0.6-uci.exe").exists())

    def test_combine_relabels_players_and_preserves_pair_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);match=root/"main-fixture";match.mkdir()
            (match/"games.pgn").write_text(mate_game("candidate")+"\n\n"+mate_game("baseline")+"\n")
            moves=tuple("a2a3 a7a6 h2h3 h7h6 f2f3 e7e5 g2g4 d8h4".split())
            clusters,count=R.combine([(match,R.ZENITH,"Anchor")],root/"combined.pgn",{moves:"F17"})
            self.assertEqual(count,2)
            self.assertEqual(set(clusters),{"F17"})
            self.assertEqual(len(clusters["F17"]),2)
            text=(root/"combined.pgn").read_text()
            self.assertIn('[OpeningFamily "F17"]',text)
            self.assertNotIn('[White "candidate"]',text)
            self.assertIn(f'[White "{R.ZENITH}"]',text)
            self.assertIn('[White "Anchor"]',text)

    def test_ordo_csv_uses_player_header_and_preserves_quoted_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"ratings.csv"
            file.write_text('"#","PLAYER","RATING","ERROR"\n1,"Alpha, beta",2499.25,20\n')
            self.assertEqual(R.read_ratings(file),{"Alpha, beta":2499.25})

    def test_bootstrap_resamples_whole_families_and_both_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def records(name,result):
                return [f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n\n{result}\n\n'
                        for white,black in [(R.ZENITH,name),(name,R.ZENITH)]]
            clusters={"F17":records("Anchor","1-0"),"F18":records("Anchor","1/2-1/2")}
            def fit(binary,pgn,*args):
                text=pgn.read_text()
                self.assertEqual(text.count(f'[White "{R.ZENITH}"]'),2)
                self.assertEqual(text.count('[White "Anchor"]'),2)
                # Each family contributes an entire two-color pair (or copies).
                self.assertEqual(text.count('[Result "1-0"]')%2,0)
                return {R.ZENITH:2500+text.count('[Result "1-0"]')}
            with patch.object(R,"run_ordo",side_effect=fit):
                a=R.bootstrap(Path("unused"),clusters,folder,"Anchor",2500,40)
                b=R.bootstrap(Path("unused"),clusters,folder,"Anchor",2500,40)
            self.assertEqual(a["ratings"],b["ratings"])
            self.assertEqual(a["valid_samples"],40)
            self.assertEqual(a["percentile95"],[2500,2504])

    def test_failed_cluster_fits_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(R,"run_ordo",side_effect=ValueError("disconnected")):
                with self.assertRaises(ValueError):
                    R.bootstrap(Path("unused"),{"F17":["fixture"]},Path(tmp),"Anchor",2500,20)


if __name__ == "__main__":
    unittest.main()
