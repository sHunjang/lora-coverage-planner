# core/license.py — HMAC 기반 로컬 라이선스 검증
from __future__ import annotations
import os, json, hashlib, hmac, base64

if getattr(__import__('sys'), 'frozen', False):
    import sys
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LICENSE_PATH = os.path.join(_APP_DIR, "license.dat")

# 배포 코드에는 검증용 비밀키만 포함됩니다.
# 변경 후
def _load_secret() -> bytes:
    # 1순위: exe 옆 license.key 파일 (배포 시 고객사별로 다르게 전달)
    key_path = os.path.join(_APP_DIR, "license.key")
    if os.path.exists(key_path):
        try:
            with open(key_path, 'rb') as f:
                raw = f.read().strip()
            # 간단한 난독화 해제 (base64)
            import base64
            return base64.b64decode(raw)
        except Exception:
            pass

    # # 2순위: 빌드에 포함된 _secret_key.py (기존 방식 호환)
    # try:
    #     from core._secret_key import COMPANY_SECRET
    #     return COMPANY_SECRET.encode('utf-8') if isinstance(
    #         COMPANY_SECRET, str) else COMPANY_SECRET
    # except Exception:
    #     pass

    # 3순위: 환경변수 (개발/테스트용)
    env_key = os.environ.get('LORASCAPE_SECRET')
    if env_key:
        return env_key.encode('utf-8')

    # 4순위: 개발 폴백 (배포 빌드에는 절대 도달하면 안 됨)
    return b"DEV-ONLY-FALLBACK-DO-NOT-SHIP"


_SECRET = _load_secret()
    


def _normalize(s: str) -> str:
    return s.strip().upper().replace(" ", "")


def _expected_code(company: str, user: str) -> str:
    msg = f"{_normalize(company)}|{_normalize(user)}".encode('utf-8')
    digest = hmac.new(_SECRET, msg, hashlib.sha256).digest()
    raw = base64.b32encode(digest[:15]).decode('ascii').rstrip('=')
    return '-'.join(raw[i:i+5] for i in range(0, len(raw), 5))


def verify_code(company: str, user: str, code: str) -> bool:
    """입력된 인증 코드가 유효한지 검증."""
    if not (company and user and code):
        return False
    expected = _expected_code(company, user)
    norm = lambda c: c.strip().upper().replace("-", "").replace(" ", "")
    return hmac.compare_digest(norm(expected), norm(code))


def save_license(company: str, user: str, code: str):
    """검증 통과한 라이선스를 로컬에 저장 (난독화)."""
    payload = json.dumps(
        {'company': company, 'user': user, 'code': code},
        ensure_ascii=False)
    # 단순 난독화 (base64) — 평문 노출 방지용
    enc = base64.b64encode(payload.encode('utf-8')).decode('ascii')
    try:
        with open(_LICENSE_PATH, 'w', encoding='utf-8') as f:
            f.write(enc)
    except Exception as e:
        print(f"[LICENSE] 저장 실패: {e}")


def load_license() -> dict | None:
    """저장된 라이선스를 읽어 검증. 유효하면 dict, 아니면 None."""
    try:
        if not os.path.exists(_LICENSE_PATH):
            return None
        with open(_LICENSE_PATH, encoding='utf-8') as f:
            enc = f.read()
        payload = base64.b64decode(enc.encode('ascii')).decode('utf-8')
        data = json.loads(payload)
        if verify_code(data.get('company', ''),
                       data.get('user', ''),
                       data.get('code', '')):
            return data
        return None
    except Exception:
        return None


def is_licensed() -> bool:
    return load_license() is not None