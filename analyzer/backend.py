from __future__ import annotations

import requests

from shared.constants import BITNETD_BASE_URL

HEALTH_URL = f"{BITNETD_BASE_URL}/health"
REQUEST_TIMEOUT_SECONDS = 1.0


def check_bitnetd_health() -> tuple[bool, str]:
    """Check bitnetd health endpoint and return a user-facing connection summary."""
    try:
        response = requests.get(HEALTH_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return True, "연결 성공"
        return False, "연결 실패"
    except requests.RequestException:
        return False, "연결 실패"
