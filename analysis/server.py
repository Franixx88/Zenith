"""Local, single-session analysis board backed by the native Zenith UCI engine."""
from __future__ import annotations

import argparse
import copy
import io
import http.client
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import webbrowser

import chess
import chess.engine
import chess.pgn
from process_guard import EngineJob

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
MAX_NODES = 4096
MAX_PLIES = 2048
MAX_BODY = 512 * 1024
OPTION_NAMES = ("Threads", "Hash", "MultiPV", "Null Move Pruning", "Selective Pruning",
                "Search Enhancements", "Modern Search", "Singular Extensions", "SMP Voting",
                "RFP Margin", "Futility Margin", "NMP Base Reduction", "LMR Divisor",
                "SyzygyPath", "SyzygyProbeLimit", "SyzygyProbeDepth")


class Conflict(ValueError):
    pass


def checked_game(text: str, kind: str) -> chess.pgn.Game:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Вставь FEN или PGN.")
    if len(text.encode("utf-8")) > MAX_BODY:
        raise ValueError("Слишком большой файл; предел — 512 КБ.")
    if kind == "fen":
        board = chess.Board(text.strip())
        if not board.is_valid():
            raise ValueError("Некорректная позиция: проверь королей, шах, пешки и рокировки.")
        game = chess.pgn.Game()
        game.setup(board)
        return game
    stream = io.StringIO(text)
    game = chess.pgn.read_game(stream)
    if game is None or game.errors:
        raise ValueError("PGN содержит некорректные или нелегальные ходы.")
    if not game.variations and "[" not in text and text.strip() not in {"*", "1-0", "0-1", "1/2-1/2"}:
        raise ValueError("Не удалось распознать PGN.")
    if chess.pgn.read_game(stream) is not None:
        raise ValueError("Импортируй одну партию за раз.")
    board = game.board()
    if board.chess960 or not board.is_valid():
        raise ValueError("Поддерживаются корректные позиции стандартных шахмат.")
    stack = [(game, board, 0)]
    count = 0
    while stack:
        node, board, ply = stack.pop()
        count += 1
        if count > MAX_NODES or ply > MAX_PLIES:
            raise ValueError("Слишком большое дерево вариантов.")
        for child in node.variations:
            if child.move not in board.legal_moves:
                raise ValueError("В PGN найден нелегальный ход.")
            following = board.copy(stack=False)
            following.push(child.move)
            stack.append((child, following, ply + 1))
    return game


class Study:
    """All document changes are atomic under the enclosing session lock."""
    def __init__(self):
        self.install(chess.pgn.Game())

    def install(self, game):
        self.game = game
        self.nodes = {}
        self.ids = {}
        stack = [game]
        while stack:
            node = stack.pop()
            index = len(self.nodes)
            self.nodes[index] = node
            self.ids[id(node)] = index
            stack.extend(reversed(node.variations))
        self.current = 0
        self.tree_cache = None

    @property
    def node(self):
        return self.nodes[self.current]

    def board(self):
        # Retains the entire move stack, including imported FEN and variations.
        return self.node.board()

    def line(self):
        result = []
        node = self.node
        while node.parent is not None:
            result.append(self.ids[id(node)])
            node = node.parent
        return [0, *reversed(result)]

    def add_moves(self, tokens):
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 128:
            raise ValueError("Ожидалась последовательность ходов UCI.")
        board = self.board()
        if len(board.move_stack) + len(tokens) > MAX_PLIES or len(self.nodes) + len(tokens) > MAX_NODES:
            raise ValueError("Достигнут предел дерева вариантов.")
        moves = []
        for token in tokens:
            move = board.parse_uci(token)
            if move not in board.legal_moves:
                raise ValueError("Нелегальный ход.")
            board.push(move)
            moves.append(move)
        # Validate the whole PV BEFORE publishing any part of it.
        node = self.node
        for move in moves:
            found = next((child for child in node.variations if child.move == move), None)
            node = found if found is not None else node.add_variation(move)
            if id(node) not in self.ids:
                index = len(self.nodes)
                self.ids[id(node)] = index
                self.nodes[index] = node
        self.current = self.ids[id(node)]
        self.tree_cache = None
        if node.is_mainline():
            self.sync_result()

    def sync_result(self):
        outcome = self.game.end().board().outcome(claim_draw=True)
        self.game.headers["Result"] = outcome.result() if outcome else "*"

    def navigate(self, destination):
        node = self.node
        if isinstance(destination, int) and not isinstance(destination, bool):
            if destination not in self.nodes:
                raise ValueError("Этот ход больше не существует.")
            self.current = destination
        elif destination == "first":
            self.current = 0
        elif destination == "previous":
            self.current = self.ids[id(node.parent)] if node.parent else 0
        elif destination == "next":
            if node.variations:
                self.current = self.ids[id(node.variations[0])]
        elif destination == "last":
            while node.variations:
                node = node.variations[0]
            self.current = self.ids[id(node)]
        else:
            raise ValueError("Неизвестное направление.")

    def tree(self):
        if self.tree_cache is not None:
            return self.tree_cache
        result = []
        stack = [(self.game, self.game.board())]
        while stack:
            node, board = stack.pop()
            index = self.ids[id(node)]
            result.append({"id": index, "parent": self.ids.get(id(node.parent)),
                           "san": node.san() if node.parent else "Начальная позиция",
                           "uci": node.move.uci() if node.move else None,
                           "ply": board.ply(), "children": [self.ids[id(n)] for n in node.variations]})
            for child in reversed(node.variations):
                following = board.copy(stack=False)
                following.push(child.move)
                stack.append((child, following))
        self.tree_cache = result
        return result

    def pgn(self):
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        return self.game.accept(exporter)


class AnalysisSession:
    def __init__(self, binary: Path, state_path: Path | None = None, start_engine=True):
        self.binary = binary
        self.state_path = state_path
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.token = secrets.token_urlsafe(32)
        self.study = Study()
        self.revision = 0
        self.sequence = 0
        self.game_id = 0
        self.enabled = True
        self.last_seen = 0.0
        self.current_analysis = None
        self.analysis = {"status": "waiting", "pv": [], "nodes": 0}
        self.pending_lines = {}
        self.pending_depth = None
        self.evaluations = {}
        self.options = {}
        self.settings = {"Threads": min(8, os.cpu_count() or 1), "Hash": 128}
        self.engine_name = "Zenith"
        self.engine_pid = None
        self.error = None
        self._restore()
        self.worker = None
        self.watchdog = None
        if start_engine:
            self.worker = threading.Thread(target=self._worker, name="zenith-uci-bridge", daemon=True)
            self.watchdog = threading.Thread(target=self._watchdog, name="zenith-browser-lease", daemon=True)
            self.worker.start()
            self.watchdog.start()

    def _restore(self):
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            if self.state_path.stat().st_size > MAX_BODY * 2:
                raise ValueError("saved session is too large")
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.study.install(checked_game(saved["pgn"], "pgn"))
            for index in saved.get("path", []):
                if not isinstance(index, int) or index < 0:
                    raise ValueError("bad cursor")
                node = self.study.node.variations[index]
                self.study.current = self.study.ids[id(node)]
            # Untrusted saved options are checked against the real UCI schema later.
            self.saved_settings = saved.get("settings", {})
            self.enabled = saved.get("enabled", True) if type(saved.get("enabled", True)) is bool else True
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            logging.warning("Cannot restore analysis session: %s", error)
            self.study = Study()

    def _save(self):
        if self.state_path is None:
            return
        path = []
        node = self.study.node
        while node.parent is not None:
            path.append(node.parent.variations.index(node))
            node = node.parent
        payload = {"pgn": self.study.pgn(), "path": list(reversed(path)), "settings": self.settings,
                   "enabled": self.enabled}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError:
            logging.exception("Cannot save analysis session")

    def _notify(self):
        self.sequence += 1
        self.changed.notify_all()

    def touch(self):
        with self.lock:
            inactive = time.monotonic() - self.last_seen > 15
            self.last_seen = time.monotonic()
            if inactive:
                self.wake.set()

    def snapshot(self):
        with self.lock:
            board = self.study.board()
            outcome = board.outcome(claim_draw=True)
            pieces = {chess.square_name(sq): piece.symbol() for sq, piece in board.piece_map().items()}
            return {"revision": self.revision, "sequence": self.sequence, "token": self.token,
                    "node": self.study.current, "fen": board.fen(en_passant="fen"), "pieces": pieces,
                    "turn": "w" if board.turn else "b", "check": board.is_check(),
                    "king": chess.square_name(board.king(board.turn)) if board.is_check() else None,
                    "legal": [move.uci() for move in board.legal_moves],
                    "lastMove": board.peek().uci() if board.move_stack else None,
                    "result": outcome.result() if outcome else None,
                    "termination": outcome.termination.name if outcome else None,
                    "headers": dict(self.study.game.headers), "tree": self.study.tree(),
                    "path": self.study.line(), "analysis": copy.deepcopy(self.analysis),
                    "evaluations": copy.deepcopy(self.evaluations), "enabled": self.enabled,
                    "settings": self.settings.copy(), "options": copy.deepcopy(self.options),
                    "engine": self.engine_name, "error": self.error}

    def _invalidate(self):
        self.revision += 1
        self.error = None
        self.pending_lines = {}
        self.pending_depth = None
        self.analysis = copy.deepcopy(self.evaluations.get(self.study.current, {"pv": [], "nodes": 0}))
        self.analysis["status"] = "starting" if self.enabled else "paused"
        if self.current_analysis is not None:
            self.current_analysis.stop()
        self.wake.set()
        self._notify()

    def _parse_settings(self, supplied):
        if not isinstance(supplied, dict) or len(supplied) > len(OPTION_NAMES):
            raise ValueError("Некорректные настройки.")
        result = {}
        for name, value in supplied.items():
            if name not in self.options:
                raise ValueError(f"Недоступная опция: {name}")
            option = self.options[name]
            if option["type"] == "check":
                if type(value) is not bool:
                    raise ValueError(f"{name}: требуется переключатель.")
            elif option["type"] == "spin":
                if type(value) is not int or not option["min"] <= value <= option["max"]:
                    raise ValueError(f"{name}: допустимо {option['min']}–{option['max']}.")
            elif not isinstance(value, str) or len(value) > 4096 or any(c in value for c in "\r\n\0"):
                raise ValueError(f"{name}: некорректная строка.")
            if name == "SyzygyPath" and value.strip() not in {"", "<empty>"}:
                paths = value.strip().strip('"').split(os.pathsep)
                if any(not Path(path).is_dir() for path in paths):
                    raise ValueError("Папка Syzygy не найдена.")
                if not any(any(Path(path).glob("*.rtbw")) or any(Path(path).glob("*.rtbz")) for path in paths):
                    raise ValueError("В папках Syzygy не найдены .rtbw/.rtbz таблицы.")
            result[name] = value
        return result

    def action(self, request):
        if not isinstance(request, dict):
            raise ValueError("Ожидался JSON-объект.")
        with self.lock:
            # Mutations from a stale tab cannot overwrite a newer position.
            if request.get("revision") != self.revision:
                raise Conflict("Позиция уже изменилась. Попробуй ещё раз.")
            action = request.get("action")
            if action == "move":
                self.study.add_moves([request.get("move")])
            elif action == "pv":
                self.study.add_moves(request.get("moves"))
            elif action == "navigate":
                self.study.navigate(request.get("to"))
            elif action in {"fen", "pgn", "new"}:
                game = chess.pgn.Game() if action == "new" else checked_game(request.get("text"), action)
                self.study.install(game)
                if action == "pgn":
                    self.study.navigate("last")
                self.evaluations.clear()
                self.game_id += 1
            elif action == "promote":
                if self.study.node.parent:
                    self.study.node.parent.promote_to_main(self.study.node)
                    self.study.tree_cache = None
                    self.study.sync_result()
            elif action == "settings":
                self.settings.update(self._parse_settings(request.get("settings")))
                self.evaluations.clear()
            elif action == "clear":
                self.evaluations.clear()
                self.game_id += 1
            elif action == "analysis":
                if type(request.get("enabled")) is not bool:
                    raise ValueError("Требуется enabled=true/false.")
                self.enabled = request["enabled"]
            else:
                raise ValueError("Неизвестное действие.")
            self.last_seen = time.monotonic()
            self._invalidate()
            self._save()
            return self.snapshot()

    def _watchdog(self):
        while not self.closed.wait(1):
            with self.lock:
                if self.current_analysis is not None and time.monotonic() - self.last_seen > 15:
                    self.current_analysis.stop()
                    self.analysis["status"] = "waiting"
                    self._notify()

    def _publish(self, info, board, revision, node):
        with self.lock:
            if revision != self.revision:
                return
            if info.get("string", "").startswith("search complete"):
                self.analysis["status"] = "complete"
                self._notify()
                return
            if "score" not in info:
                return
            rank = info.get("multipv", 1)
            expected = min(self.settings.get("MultiPV", 1), board.legal_moves.count())
            if not 1 <= rank <= expected:
                return
            depth = info.get("depth", 0)
            if rank == 1:
                self.pending_lines = {}
                self.pending_depth = depth
            elif self.pending_depth != depth:
                return
            update = {}
            for name in ("depth", "seldepth", "nodes", "nps", "time", "tbhits"):
                if name in info:
                    update[name] = info[name]
            if "score" in info:
                white = info["score"].white()
                update["cp"] = white.score()
                update["mate"] = white.mate()
                update["bound"] = "lower" if info.get("lowerbound") else "upper" if info.get("upperbound") else None
            if "pv" in info:
                position = board.copy()
                pv = []
                for move in info["pv"]:
                    if move not in position.legal_moves:
                        break
                    pv.append({"uci": move.uci(), "san": position.san(move),
                               "number": position.fullmove_number, "turn": "w" if position.turn else "b"})
                    position.push(move)
                update["pv"] = pv
            update["status"] = "running"
            update["node"] = node
            if not update.get("pv"):
                return
            self.pending_lines[rank] = update
            if len(self.pending_lines) != expected:
                return
            lines = [self.pending_lines[index] for index in range(1, expected + 1)]
            if len({line["pv"][0]["uci"] for line in lines}) != expected:
                return
            self.analysis = {**lines[0], "variations": copy.deepcopy(lines)}
            self.evaluations[node] = copy.deepcopy(self.analysis)
            self._notify()

    def _worker(self):
        engine = None
        job = None
        try:
            job = EngineJob()
            engine = chess.engine.SimpleEngine.popen_uci(str(self.binary), timeout=10,
                                                        cwd=str(ROOT), creationflags=0x08000000 if os.name == "nt" else 0)
            self.engine_pid = engine.transport.get_pid()
            job.assign(self.engine_pid)
            with self.lock:
                self.engine_name = engine.id.get("name", "Zenith")
                for name in OPTION_NAMES:
                    if name in engine.options:
                        option = engine.options[name]
                        self.options[name] = {"type": option.type, "default": option.default,
                                              "min": option.min, "max": option.max}
                supplied = getattr(self, "saved_settings", {})
                try:
                    self.settings.update(self._parse_settings(supplied))
                except ValueError:
                    logging.warning("Ignoring invalid saved engine settings")
                for name, option in self.options.items():
                    self.settings.setdefault(name, option["default"])
                self._notify()
            self.wake.set()
            while not self.closed.is_set():
                self.wake.wait()
                self.wake.clear()
                with self.lock:
                    if self.closed.is_set():
                        break
                    if not self.enabled or time.monotonic() - self.last_seen > 15:
                        continue
                    revision, node, game_id = self.revision, self.study.current, self.game_id
                    board = self.study.board()
                    settings = self.settings.copy()
                    if board.is_game_over(claim_draw=True):
                        outcome = board.outcome(claim_draw=True)
                        self.analysis = {"status": "terminal", "pv": [], "nodes": 0,
                                         "cp": 0 if outcome.winner is None else None,
                                         "mate": 0 if board.is_checkmate() else None,
                                         "winner": "w" if outcome.winner else "b" if outcome.winner is not None else None}
                        self._notify()
                        continue
                # python-chess manages MultiPV via analysis(), not configure().
                engine.configure({name: value for name, value in settings.items() if name != "MultiPV"})
                count = settings.get("MultiPV", 1) if "MultiPV" in engine.options else None
                analysis = engine.analysis(board, limit=None, multipv=count, game=game_id, info=chess.engine.INFO_ALL)
                with self.lock:
                    self.current_analysis = analysis
                    if revision != self.revision or not self.enabled or self.closed.is_set():
                        analysis.stop()
                try:
                    for info in analysis:
                        self._publish(info, board, revision, node)
                    analysis.wait()
                finally:
                    with self.lock:
                        self.current_analysis = None
        except Exception as error:
            logging.exception("Zenith analysis failed")
            with self.lock:
                self.error = f"Не удалось запустить анализ: {error}"
                self.analysis["status"] = "error"
                self._notify()
        finally:
            if engine:
                try:
                    engine.quit()
                except (chess.engine.EngineError, TimeoutError):
                    engine.close()
            if job:
                job.close()

    def close(self):
        self.closed.set()
        with self.lock:
            if self.current_analysis is not None:
                self.current_analysis.stop()
            self.wake.set()
            self._notify()
        if self.worker:
            self.worker.join(timeout=12)
        if self.watchdog:
            self.watchdog.join(timeout=2)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, session):
        self.session = session
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        super().__init__(("127.0.0.1", port), Handler)
        if port == 0:
            self.allowed_hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            # Closing/reloading the board is a normal stream disconnect.
            pass

    def log_message(self, *args):
        pass

    def _headers(self, status, content_type, size=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; script-src 'self'; frame-ancestors 'none'")
        if size is not None:
            self.send_header("Content-Length", str(size))
        self.end_headers()

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _local(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            self._json({"error": "Local connections only"}, 403)
            self.close_connection = True
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site" and self.headers.get("Sec-Fetch-Mode") != "navigate":
            self._json({"error": "Foreign request rejected"}, 403)
            self.close_connection = True
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in self.server.allowed_hosts}:
            self._json({"error": "Foreign origin rejected"}, 403)
            self.close_connection = True
            return False
        return True

    def do_GET(self):
        if not self._local():
            return
        path = urlsplit(self.path).path
        session = self.server.session
        try:
            if path == "/api/state":
                session.touch()
                self._json(session.snapshot())
            elif path == "/api/health":
                self._json({"app": "zenith-analysis", "api": 1, "engine": session.engine_name,
                            "engine_pid": session.engine_pid, "pid": os.getpid(),
                            "persistent": session.state_path is not None})
            elif path == "/api/events":
                self._events()
            elif path == "/api/pgn":
                with session.lock:
                    pgn = session.study.pgn().encode()
                self._headers(200, "application/x-chess-pgn; charset=utf-8", len(pgn))
                self.wfile.write(pgn)
            else:
                relative = "index.html" if path == "/" else path.lstrip("/")
                file = (STATIC / relative).resolve()
                if not file.is_relative_to(STATIC.resolve()) or not file.is_file():
                    self._json({"error": "Not found"}, 404)
                    return
                types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
                body = file.read_bytes()
                self._headers(200, types.get(file.suffix, "application/octet-stream"), len(body))
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if not self._local():
            return
        session = self.server.session
        if not secrets.compare_digest(self.headers.get("X-Zenith-Token", ""), session.token):
            self._json({"error": "Invalid session token"}, 403)
            self.close_connection = True
            return
        if urlsplit(self.path).path != "/api/action":
            self._json({"error": "Not found"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                raise ValueError("Слишком большой или пустой запрос.")
            request = json.loads(self.rfile.read(size))
            self._json(session.action(request))
        except Conflict as error:
            self._json({"error": str(error)}, 409)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            self._json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _events(self):
        session = self.server.session
        self._headers(200, "text/event-stream; charset=utf-8")
        sequence = revision = -1
        self.connection.settimeout(15)
        try:
            while not session.closed.is_set():
                session.touch()
                with session.changed:
                    if session.sequence == sequence:
                        session.changed.wait(timeout=5)
                    next_sequence, next_revision = session.sequence, session.revision
                    if next_revision != revision:
                        event, payload = "state", session.snapshot()
                    elif next_sequence != sequence:
                        event, payload = "analysis", {"revision": session.revision,
                            "sequence": session.sequence,
                            "analysis": copy.deepcopy(session.analysis), "error": session.error,
                            "options": copy.deepcopy(session.options), "settings": session.settings.copy(),
                            "engine": session.engine_name}
                    else:
                        event = None
                if event:
                    message = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    self.wfile.write(message.encode())
                else:
                    self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                sequence, revision = next_sequence, next_revision
                # UI updates are capped; this never slows the native search.
                session.closed.wait(0.15)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--engine", type=Path, default=ROOT / "target/release" / ("zenith.exe" if os.name == "nt" else "zenith"))
    parser.add_argument("--open", action="store_true", help="open the board in the default browser")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if not args.engine.is_file():
        parser.error("Build the engine first: cargo build --release --bin zenith")
    url = f"http://127.0.0.1:{args.port}"
    if args.open:
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1)
        try:
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            health = json.loads(response.read())
            if response.status == 200 and health.get("app") == "zenith-analysis" and health.get("api") == 1:
                print(f"Already running: {url}", flush=True)
                webbrowser.open(url)
                return
        except (OSError, ValueError):
            pass
        finally:
            connection.close()
    logging.basicConfig(level=logging.WARNING)
    session = AnalysisSession(args.engine.resolve(), None if args.no_save else ROOT / "runs/analysis/session.json")
    try:
        server = Server(args.port, session)
    except OSError:
        session.close()
        raise
    print(f"Zenith analysis board: {url}", flush=True)
    print("Native UCI, 8 threads by default. Ctrl+C stops the server and engine.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


if __name__ == "__main__":
    main()
