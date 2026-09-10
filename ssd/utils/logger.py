"""训练日志：同时输出到控制台与 txt 文件（UTF-8，逐行 flush，崩溃不丢日志）。"""
from __future__ import annotations

import os
import time


class RunLogger:
    """极简 logger：log(msg) 打印并追加写文件。"""

    def __init__(self, log_path: str | None = None, prefix_time: bool = True):
        self.log_path = log_path
        self.prefix_time = prefix_time
        self._fh = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    def log(self, msg: str, also_print: bool = True) -> str:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}" if self.prefix_time else msg
        if also_print:
            print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        return line

    def section(self, title: str) -> None:
        self.log("=" * 60)
        self.log(title)
        self.log("=" * 60)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
