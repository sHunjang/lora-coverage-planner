# core/app_config.py — 앱 전역 설정 (회사/사용자/작업폴더)
from __future__ import annotations
import os, json

if getattr(__import__('sys'), 'frozen', False):
    import sys
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONFIG_PATH = os.path.join(_APP_DIR, "app_config.json")

DEFAULT_CONFIG = {
    "company"  : "",
    "user"     : "",
    "work_dir" : os.path.join(_APP_DIR, "projects"),
}


def load_config() -> dict:
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, encoding='utf-8') as f:
                data = json.load(f)
            merged = dict(DEFAULT_CONFIG)
            merged.update(data)
            return merged
    except Exception:
        pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    try:
        with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[CONFIG] 저장 실패: {e}")


def is_first_run() -> bool:
    """회사/사용자 정보가 비어있으면 최초 실행으로 간주."""
    cfg = load_config()
    return not (cfg.get('company') and cfg.get('user'))