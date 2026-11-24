"""Tiny JSON-lines + console logger."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: str | Path | None = None, verbose: bool = True):
        self.run_dir = Path(run_dir) if run_dir else None
        self.verbose = verbose
        self.t0 = time.time()
        self.files: dict = {}

    def log(self, stream: str, record: dict) -> None:
        record = dict(record, wall=round(time.time() - self.t0, 3))
        if self.run_dir is not None:
            f = self.files.get(stream)
            if f is None:
                f = open(self.run_dir / f"{stream}.jsonl", "a")
                self.files[stream] = f
            f.write(json.dumps(record, default=float) + "\n")
            f.flush()

    def info(self, msg: str) -> None:
        if self.verbose:
            print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        self.files = {}
