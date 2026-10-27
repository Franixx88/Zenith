"""Rating input semantics: exclude scouts, preserve colors/families, parse Ordo."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess.pgn
import run_rating_gauntlet as R
from test_engine_test import mate_game


class RatingEvidenceTests(unittest.TestCase):
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
