# core/app_logger.py — 중앙 로그 관리
"""
LoRaScape 로그 관리 모듈
────────────────────────────────────────────────────────────
기존 코드 전반에 흩어진 print() 호출을 그대로 유지하면서,
그 출력을 가로채 다음 세 곳에 동시에 보냅니다.

  1. 원래 콘솔(stdout) — 기존 동작 그대로 유지
  2. 로그 파일 (logs/lorascape_YYYYMMDD.log) — 일자별 회전
  3. 메모리 버퍼 + Qt 시그널 — LogViewerWindow가 실시간 구독

사용법 (main.py 최상단, 다른 import보다 먼저):
    from core.app_logger import install_log_capture
    install_log_capture()

이후 코드 전체에서 기존처럼 print(...)만 쓰면 자동으로 캡처됩니다.
별도로 core.app_logger.log(msg, level="INFO"/"WARN"/"ERROR")를
호출하면 레벨을 직접 지정할 수도 있습니다.
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import sys, os, io
from datetime import datetime
from collections import deque

try:
    from PyQt5.QtCore import QObject, pyqtSignal
    _HAS_QT = True
except Exception:
    _HAS_QT = False

if getattr(sys, 'frozen', False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOG_DIR = os.path.join(_APP_DIR, "logs")
_MAX_BUFFER = 2000          # 메모리에 유지할 최대 라인 수
_MAX_LOG_FILES = 14         # 로그 파일 보관 일수 (오래된 파일 자동 삭제)


def _classify(line: str) -> str:
    """기존 print() 메시지 패턴으로 레벨을 추정."""
    s = line.upper()
    if '[ERROR]' in s or '오류' in line or 'TRACEBACK' in s or 'EXCEPTION' in s:
        return 'ERROR'
    if '⚠' in line or 'WARN' in s or '경고' in line:
        return 'WARN'
    return 'INFO'


class _LogSignal(QObject if _HAS_QT else object):
    if _HAS_QT:
        new_line = pyqtSignal(str, str, str)  # (timestamp, level, message)


class _TeeStream(io.TextIOBase):
    """기존 stdout/stderr을 가로채 파일·버퍼·시그널로 동시에 내보내는 스트림."""

    def __init__(self, original_stream, file_handle, manager: "LogManager"):
        self._orig = original_stream
        self._file = file_handle
        self._mgr  = manager
        self._partial = ""

    def write(self, s: str):
        # 원래 콘솔에는 항상 그대로 출력 (기존 동작 보존)
        try:
            self._orig.write(s)
        except Exception:
            pass

        self._partial += s
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                self._mgr._record(line)
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


class LogManager:
    """로그 캡처·저장·조회를 담당하는 싱글톤 매니저."""

    def __init__(self):
        self.buffer: deque = deque(maxlen=_MAX_BUFFER)
        self.signal = _LogSignal() if _HAS_QT else None
        self._file_handle = None
        self._installed = False

    def install(self):
        if self._installed:
            return
        os.makedirs(_LOG_DIR, exist_ok=True)
        self._cleanup_old_logs()

        fname = datetime.now().strftime("lorascape_%Y%m%d.log")
        fpath = os.path.join(_LOG_DIR, fname)
        self._file_handle = open(fpath, 'a', encoding='utf-8', buffering=1)

        sys.stdout = _TeeStream(sys.stdout, self._file_handle, self)
        sys.stderr = _TeeStream(sys.stderr, self._file_handle, self)
        self._installed = True
        self._record(f"[INFO] 로그 시작: {fpath}")

    def _cleanup_old_logs(self):
        """오래된 로그 파일을 자동 삭제 (보관 일수 초과분)."""
        try:
            if not os.path.isdir(_LOG_DIR):
                return
            files = sorted(
                f for f in os.listdir(_LOG_DIR)
                if f.startswith("lorascape_") and f.endswith(".log"))
            excess = len(files) - _MAX_LOG_FILES
            for f in files[:max(0, excess)]:
                try:
                    os.remove(os.path.join(_LOG_DIR, f))
                except Exception:
                    pass
        except Exception:
            pass

    def _record(self, line: str):
        ts    = datetime.now().strftime("%H:%M:%S")
        level = _classify(line)
        self.buffer.append((ts, level, line))
        if self.signal is not None:
            try:
                self.signal.new_line.emit(ts, level, line)
            except Exception:
                pass
        # 파일에는 TeeStream.write가 이미 orig를 거치므로 별도 기록 불필요
        # (orig가 콘솔이고, 콘솔 출력은 그대로 두되 파일 기록만 별도 수행)
        if self._file_handle:
            try:
                self._file_handle.write(
                    f"[{ts}] [{level}] {line}\n")
            except Exception:
                pass

    def get_lines(self):
        return list(self.buffer)

    def log_dir(self) -> str:
        return _LOG_DIR

    def log(self, msg: str, level: str = "INFO"):
        """기존 print() 대신 레벨을 직접 지정하고 싶을 때 사용."""
        tag = f"[{level}] {msg}"
        print(tag)


# ── 모듈 전역 싱글톤 ─────────────────────────────────────────
_manager = LogManager()


def install_log_capture():
    """main.py 최상단에서 한 번 호출."""
    _manager.install()


def get_manager() -> LogManager:
    return _manager


def log(msg: str, level: str = "INFO"):
    _manager.log(msg, level)
