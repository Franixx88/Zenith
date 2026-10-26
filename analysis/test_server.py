import copy
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import time
import unittest
from unittest import mock

import chess
import chess.engine
from server import AnalysisSession, Conflict, Server, Study, checked_game, ROOT, main


class DocumentTests(unittest.TestCase):
    def test_branches_keep_mainline_and_full_history(self):
        study = Study()
        study.add_moves(["e2e4", "e7e5", "g1f3"])
        main = study.current
        study.navigate("first")
        study.add_moves(["d2d4", "d7d5"])
        branch = study.current
        self.assertEqual(study.game.variations[0].move.uci(), "e2e4")
        self.assertEqual([m.uci() for m in study.board().move_stack], ["d2d4", "d7d5"])
        self.assertIn("( 1. d4 d5 )", study.pgn())
        study.install(checked_game(study.pgn(), "pgn"))
        self.assertEqual(len(study.nodes), 6)
        self.assertNotEqual(main, branch)

    def test_illegal_pv_is_atomic_and_null_moves_are_rejected(self):
        study = Study()
        for moves in [["e2e4", "e7e4"], ["0000"], [None]]:
            with self.assertRaises((ValueError, TypeError)):
                study.add_moves(moves)
            self.assertEqual(len(study.nodes), 1)
            self.assertEqual(study.current, 0)

    def test_special_moves_and_underpromotion(self):
        for fen, moves in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"]),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", ["a7a8n"]),
        ]:
            study = Study()
            study.install(checked_game(fen, "fen"))
            study.add_moves(moves)
            expected = chess.Board(fen)
            for move in moves:
                expected.push_uci(move)
            self.assertEqual(study.board().fen(), expected.fen())
            self.assertEqual(study.board().root().fen(), chess.Board(fen).fen())
            if moves == ["a7a8n"]:
                self.assertEqual(study.game.headers["Result"], "1/2-1/2")

    def test_bad_import_does_not_replace_document_and_stale_actions_conflict(self):
        session = AnalysisSession(Path("unused"), start_engine=False)
        session.action({"revision": 0, "action": "move", "move": "e2e4"})
        with self.assertRaises(Conflict):
            session.action({"revision": 0, "action": "new"})
        before = session.study.pgn()
        for text in ["8/8/8/8/8/8/8/8 w - - 0 1", "invalid"]:
            with self.assertRaises(ValueError):
                session.action({"revision": 1, "action": "fen", "text": text})
        self.assertEqual(session.study.pgn(), before)
        with self.assertRaises(ValueError):
            checked_game("1. e4 e5 2. Ke3 *", "pgn")
        with self.assertRaises(ValueError):
            checked_game("this is not a chess game", "pgn")

    def test_saved_variation_and_cursor_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            session = AnalysisSession(Path("unused"), path, start_engine=False)
            session.action({"revision": 0, "action": "move", "move": "e2e4"})
            session.action({"revision": 1, "action": "navigate", "to": "first"})
            session.action({"revision": 2, "action": "move", "move": "d2d4"})
            restored = AnalysisSession(Path("unused"), path, start_engine=False)
            self.assertEqual(restored.study.board().fen(), session.study.board().fen())
            self.assertEqual(restored.study.pgn(), session.study.pgn())

    def test_stale_engine_info_cannot_replace_new_position(self):
        session = AnalysisSession(Path("unused"), start_engine=False)
        session.action({"revision": 0, "action": "move", "move": "e2e4"})
        before = copy.deepcopy(session.analysis)
        session._publish({"score": chess.engine.PovScore(chess.engine.Cp(123), chess.WHITE)}, chess.Board(), 0, 0)
        self.assertEqual(session.analysis, before)

    def test_score_is_normalized_to_white_and_pv_to_san(self):
        session = AnalysisSession(Path("unused"), start_engine=False)
        board = chess.Board()
        board.push_uci("e2e4")
        move = board.parse_uci("e7e5")
        session._publish({"score": chess.engine.PovScore(chess.engine.Cp(75), chess.BLACK),
                          "pv": [move], "depth": 5}, board, 0, 0)
        self.assertEqual(session.analysis["cp"], -75)
        self.assertEqual(session.analysis["pv"][0]["san"], "e5")
        self.assertEqual(session.analysis["pv"][0]["turn"], "b")

    def test_multipv_publishes_only_a_complete_same_depth_batch(self):
        session = AnalysisSession(Path("unused"), start_engine=False)
        session.settings["MultiPV"] = 3
        board = chess.Board()
        def feed(rank, depth, uci, score):
            session._publish({"multipv": rank, "depth": depth, "nodes": 1000,
                "score": chess.engine.PovScore(chess.engine.Cp(score), chess.WHITE),
                "pv": [board.parse_uci(uci)]}, board, session.revision, 0)
        feed(1, 4, "e2e4", 50)
        feed(2, 4, "d2d4", 30)
        self.assertFalse(session.analysis["pv"])
        feed(3, 4, "g1f3", 10)
        self.assertEqual(len(session.analysis["variations"]), 3)
        self.assertEqual(session.analysis["cp"], 50)
        feed(1, 5, "e2e4", 45)
        feed(2, 4, "d2d4", 29)
        feed(3, 5, "g1f3", 10)
        self.assertEqual(session.analysis["depth"], 4)
        feed(2, 5, "d2d4", 28)
        self.assertEqual(session.analysis["depth"], 5)
        self.assertEqual([line["cp"] for line in session.analysis["variations"]], [45, 28, 10])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.session = AnalysisSession(Path("unused"), start_engine=False)
        self.server = Server(0, self.session)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.session.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request(method, path, json.dumps(data) if data is not None else None, headers or {})
        response = connection.getresponse()
        status, body = response.status, response.read()
        connection.close()
        return status, body

    def test_real_state_move_and_static_page(self):
        status, body = self.request("GET", "/api/state")
        state = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(state["legal"]), 20)
        status, body = self.request("POST", "/api/action", {"action": "move", "revision": 0, "move": "e2e4"}, {"X-Zenith-Token": state["token"]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["turn"], "b")
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/pieces/wN.svg")[0], 200)

    def test_reject_foreign_origin_csrf_and_path_traversal(self):
        self.assertEqual(self.request("POST", "/api/action", {"action": "new", "revision": 0})[0], 403)
        headers = {"X-Zenith-Token": self.session.token, "Origin": "https://example.org"}
        self.assertEqual(self.request("POST", "/api/action", {"action": "new", "revision": 0}, headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/state", headers={"Host": "evil.invalid"})[0], 403)
        self.assertEqual(self.request("GET", "/../server.py")[0], 404)

    def test_second_launch_opens_existing_board_without_another_engine(self):
        argv = ["server.py", "--port", str(self.server.server_port), "--open"]
        with mock.patch.object(sys, "argv", argv), mock.patch("server.webbrowser.open") as browser, mock.patch("server.AnalysisSession") as constructor:
            main()
            constructor.assert_not_called()
            browser.assert_called_once_with(f"http://127.0.0.1:{self.server.server_port}")


class NativeEngineTests(unittest.TestCase):
    def test_real_native_multipv_batches_and_rapid_position_change(self):
        binary = Path(os.environ.get("ZENITH_TEST_ENGINE", str(ROOT / "target/release/zenith.exe")))
        if not binary.exists():
            self.skipTest("native release binary missing")
        session = AnalysisSession(binary)
        def wait(predicate):
            end = time.monotonic() + 15
            while time.monotonic() < end:
                with session.lock:
                    self.assertFalse(session.error, session.error)
                    if predicate():
                        return
                time.sleep(0.01)
            self.fail("timed out waiting for native MultiPV")
        try:
            wait(lambda: bool(session.options))
            if "MultiPV" not in session.options:
                self.skipTest("engine predates MultiPV")
            session.action({"revision": session.revision, "action": "settings",
                "settings": {"Threads": 2, "Hash": 16, "MultiPV": 3, "SyzygyPath": "<empty>"}})
            wait(lambda: len(session.analysis.get("variations", [])) == 3)
            for move in ["e2e4", "e7e5"]:
                session.action({"revision": session.revision, "action": "move", "move": move})
            wait(lambda: session.analysis.get("node") == session.study.current and len(session.analysis.get("variations", [])) == 3)
            lines = session.analysis["variations"]
            self.assertEqual(len({line["pv"][0]["uci"] for line in lines}), 3)
            self.assertEqual(len({line["depth"] for line in lines}), 1)
            for line in lines:
                board = session.study.board()
                for move in line["pv"]:
                    parsed = board.parse_uci(move["uci"])
                    self.assertIn(parsed, board.legal_moves)
                    board.push(parsed)
        finally:
            session.close()

    def test_live_analysis_cancellation_settings_and_repetition(self):
        binary = Path(os.environ.get("ZENITH_TEST_ENGINE", str(ROOT / "target/release/zenith.exe")))
        if not binary.exists():
            self.skipTest("native release binary missing")
        session = AnalysisSession(binary)
        def wait(predicate, label):
            end = time.monotonic() + 12
            while time.monotonic() < end:
                with session.lock:
                    if session.error:
                        self.fail(session.error)
                    if predicate():
                        return
                time.sleep(0.01)
            self.fail(f"timed out: {label}")
        def act(**payload):
            return session.action({"revision": session.revision, **payload})
        try:
            wait(lambda: bool(session.options), "engine handshake")
            act(action="settings", settings={"Threads": 2, "Hash": 16, "SyzygyPath": "<empty>"})
            wait(lambda: session.analysis.get("depth", 0) >= 3, "initial depth")
            with session.lock:
                session.last_seen = time.monotonic() - 20
            wait(lambda: session.current_analysis is None, "idle browser lease cancellation")
            session.touch()
            wait(lambda: session.current_analysis is not None, "resume after browser returns")
            act(action="move", move="e2e4")
            act(action="move", move="e7e5")
            wait(lambda: session.analysis.get("node") == session.study.current and bool(session.analysis.get("pv")), "new PV")
            board = session.study.board()
            for move in session.analysis["pv"]:
                parsed = board.parse_uci(move["uci"])
                self.assertIn(parsed, board.legal_moves)
                board.push(parsed)
            act(action="analysis", enabled=False)
            wait(lambda: session.current_analysis is None, "stop acknowledgement")
            self.assertEqual(session.analysis["status"], "paused")
            act(action="new")
            for token in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
                act(action="move", move=token)
            act(action="analysis", enabled=True)
            wait(lambda: session.analysis.get("status") == "terminal", "threefold draw")
            self.assertEqual(session.snapshot()["termination"], "THREEFOLD_REPETITION")
            self.assertEqual(session.analysis["cp"], 0)
        finally:
            session.close()
        self.assertFalse(session.worker.is_alive())

    @unittest.skipUnless(os.name == "nt", "Windows Job Objects")
    def test_killed_bridge_does_not_orphan_engine_process(self):
        import ctypes as c
        from ctypes import wintypes as w
        code = ("from process_guard import EngineJob; import subprocess,sys,time; "
                "job=EngineJob(); child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                "job.assign(child.pid); print(child.pid,flush=True); time.sleep(60)")
        parent = subprocess.Popen([sys._base_executable, "-u", "-c", code],
                                  cwd=Path(__file__).parent, stdout=subprocess.PIPE, text=True)
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        kernel.WaitForSingleObject.restype = w.DWORD
        kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        child = None
        try:
            pid = int(parent.stdout.readline().strip())
            child = kernel.OpenProcess(0x101001, False, pid)
            self.assertTrue(child)
            parent.kill()
            parent.wait(timeout=3)
            self.assertEqual(kernel.WaitForSingleObject(child, 5000), 0)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=3)
            parent.stdout.close()
            if child:
                kernel.TerminateProcess(child, 1)
                kernel.CloseHandle(child)


if __name__ == "__main__":
    unittest.main()
