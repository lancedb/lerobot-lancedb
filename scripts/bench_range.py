"""Read-only path-style S3 object GET/HEAD and HTTP byte ranges on loopback."""

import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

CATEGORIES = ("frames", "videos_descriptor", "video_payload", "parquet", "other")
BUCKET = "loader-bench"


def category(key):
    path = Path(key)
    if "frames.lance" in path.parts:
        return "frames"
    if "videos.lance" in path.parts:
        return "videos_descriptor"
    return {".mp4": "video_payload", ".parquet": "parquet"}.get(path.suffix, "other")


class Traffic:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {key: {"gets": 0, "bytes": 0} for key in CATEGORIES}

    def add(self, kind, gets=0, size=0):
        with self.lock:
            self.counts[kind]["gets"] += gets
            self.counts[kind]["bytes"] += size

    def snapshot(self):
        with self.lock:
            rows = {key: dict(counts) for key, counts in self.counts.items()}
        rows["total"] = {k: sum(v[k] for v in rows.values()) for k in ("gets", "bytes")}
        return rows


def byte_range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError(value)
    start, end = match.groups()
    lo = int(start) if start else max(0, size - int(end))
    hi = min(size, int(end) + 1) if start and end else size
    if lo >= hi:
        raise ValueError(value)
    return lo, hi


@contextmanager
def range_server(root, rtt_ms):
    # Cache snapshots contain symlinks into the Hub blob directory. Only serve named objects.
    objects = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    traffic = Traffic()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def serve(self):
            time.sleep(rtt_ms / 1000)
            prefix = f"/{BUCKET}/"
            path = unquote(urlsplit(self.path).path)
            key = path.removeprefix(prefix)
            kind = category(key)
            if self.command == "GET":
                traffic.add(kind, gets=1)
            if not path.startswith(prefix) or key not in objects:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            file = objects[key]
            stat = file.stat()
            size = stat.st_size
            value = self.headers.get("Range")
            try:
                lo, hi = (0, size) if value is None else byte_range(value, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200 if value is None else 206)
            self.send_header("Content-Length", str(hi - lo))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", f'"{stat.st_mtime_ns:x}-{size:x}"')
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            if value is not None:
                self.send_header("Content-Range", f"bytes {lo}-{hi - 1}/{size}")
            self.end_headers()
            if self.command == "GET":
                with file.open("rb") as stream:
                    stream.seek(lo)
                    remaining = hi - lo
                    try:
                        while remaining:
                            chunk = stream.read(min(256 * 1024, remaining))
                            if not chunk:
                                raise OSError(f"Object truncated during GET: {key}")
                            self.wfile.write(chunk)
                            traffic.add(kind, size=len(chunk))
                            remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        do_GET = do_HEAD = serve

    class Server(ThreadingHTTPServer):
        # Concurrent episode prefetches should see the requested RTT without listen-backlog retries.
        request_queue_size = 128

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", traffic
        finally:
            server.shutdown()
            thread.join()
