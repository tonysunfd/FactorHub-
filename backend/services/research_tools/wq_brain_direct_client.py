from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.core.settings import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.worldquantbrain.com"
_POLL_INTERVAL = 10
_POLL_MAX_ATTEMPTS = 36
_CONCURRENT_BACKOFF = 30
_MAX_RETRIES = 5

SUBMIT_THRESHOLDS = {
    "sharpe": 1.25,
    "fitness": 1.0,
    "turnover_max": 0.7,
    "turnover_min": 0.01,
}

_ACCOUNT_ENV = {
    "primary": ("WQ_BRAIN_EMAIL", "WQ_BRAIN_PASSWORD"),
    "alt": ("WQ_BRAIN_ALT_EMAIL", "WQ_BRAIN_ALT_PASSWORD"),
}


def _get_credential(name: str, fallback: str = "") -> str:
    return (os.getenv(name) or getattr(settings, name, fallback) or "").strip()


def is_configured(account: str | None = None) -> bool:
    if account:
        env_email, env_pwd = _ACCOUNT_ENV.get(account, _ACCOUNT_ENV["primary"])
        return bool(_get_credential(env_email) and _get_credential(env_pwd))
    return any(bool(_get_credential(e) and _get_credential(p)) for e, p in _ACCOUNT_ENV.values())


def configured_accounts() -> list[str]:
    return [name for name, (e, p) in _ACCOUNT_ENV.items() if _get_credential(e) and _get_credential(p)]


class WQBrainDirectClient:
    def __init__(self, email: str | None = None, password: str | None = None):
        self.email = email or _get_credential("WQ_BRAIN_EMAIL")
        self.password = password or _get_credential("WQ_BRAIN_PASSWORD")
        self._session: requests.Session | None = None

    @classmethod
    def for_account(cls, account: str = "primary") -> "WQBrainDirectClient":
        env_email, env_pwd = _ACCOUNT_ENV.get(account, _ACCOUNT_ENV["primary"])
        return cls(email=_get_credential(env_email), password=_get_credential(env_pwd))

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.trust_env = False
            retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        return self._session

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def authenticate(self) -> bool:
        s = self._get_session()
        r = s.post(f"{API_BASE}/authentication", auth=(self.email, self.password))
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", "60"))
            logger.info("WQ auth rate-limited, waiting %ss", retry)
            time.sleep(retry + 1)
            return self.authenticate()
        if r.status_code not in (200, 201):
            logger.error("WQ auth failed: HTTP %s", r.status_code)
            return False
        data = r.json()
        if "inquiry" in data:
            logger.error("WQ auth requires biometric verification — log in via browser first")
            return False
        return True

    def get_user_info(self) -> dict[str, Any]:
        r = self._get_session().get(f"{API_BASE}/users/self")
        return r.json() if r.status_code == 200 else {}

    def simulate(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 0,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        s = self._get_session()
        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expression,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                r = s.post(f"{API_BASE}/simulations", json=payload)
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = _CONCURRENT_BACKOFF * (attempt + 1)
                logger.warning("WQ simulation connection error attempt %s: %s", attempt + 1, exc)
                if progress_callback:
                    progress_callback(0, f"连接异常，等待 {wait}s")
                time.sleep(wait)
                continue

            if r.status_code in (200, 201, 202):
                break

            if r.status_code == 429:
                detail = ""
                try:
                    detail = r.json().get("detail", "")
                except Exception:
                    detail = ""
                if "CONCURRENT_SIMULATION_LIMIT" in detail:
                    wait = _CONCURRENT_BACKOFF * (attempt + 1)
                    if progress_callback:
                        progress_callback(0, f"并发限制，等待 {wait}s")
                    time.sleep(wait)
                    continue

                retry = int(r.headers.get("Retry-After", "60"))
                if progress_callback:
                    progress_callback(0, f"速率限制，等待 {retry}s")
                time.sleep(retry + 1)
                continue

            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        else:
            return {"ok": False, "error": "WQ 模拟重试次数超限"}

        location = r.headers.get("Location", "")
        if not location:
            return {"ok": False, "error": "WQ 模拟未返回 Location"}

        url = location if location.startswith("http") else f"{API_BASE}{location}"
        for attempt in range(_POLL_MAX_ATTEMPTS):
            try:
                poll = s.get(url)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(_POLL_INTERVAL)
                continue
            if poll.status_code != 200:
                time.sleep(_POLL_INTERVAL)
                continue
            try:
                data = poll.json()
            except Exception:
                time.sleep(_POLL_INTERVAL)
                continue

            status = (data.get("status") or "").upper()
            progress = data.get("progress", 0)
            if progress_callback:
                pct = int(progress * 100) if isinstance(progress, float) and progress <= 1 else int(progress or 0)
                progress_callback(min(pct, 99), f"模拟进行中 ({pct}%)")

            if status in ("DONE", "COMPLETE"):
                alpha_raw = data.get("alpha", "")
                alpha_id = alpha_raw.split("/")[-1] if alpha_raw else None
                is_data = data.get("is", {})
                oos_data = data.get("oos", {})
                if alpha_id and not is_data:
                    alpha_detail = self._fetch_alpha(alpha_id)
                    is_data = alpha_detail.get("is", {})
                    oos_data = alpha_detail.get("oos", {})
                if progress_callback:
                    progress_callback(100, "模拟完成")
                return {
                    "ok": True,
                    "expression": expression,
                    "alpha_id": alpha_id,
                    "simulation_id": data.get("id", ""),
                    "is": is_data,
                    "oos": oos_data,
                    "settings": data.get("settings", {}),
                }
            if status in ("ERROR", "FAILED"):
                return {"ok": False, "error": f"WQ 模拟失败：{data.get('message', status)}"}

            time.sleep(_POLL_INTERVAL)

        return {"ok": False, "error": "WQ 模拟轮询超时"}

    def _fetch_alpha(self, alpha_id: str) -> dict[str, Any]:
        r = self._get_session().get(f"{API_BASE}/alphas/{alpha_id}")
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                logger.warning("WQ alpha detail json invalid: %s", alpha_id)
                return {}
        return {}

    def list_platform_alphas(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        r = self._get_session().get(
            f"{API_BASE}/users/self/alphas",
            params={"limit": limit, "offset": offset, "order": "-dateCreated"},
        )
        if r.status_code != 200:
            raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:500]}", response=r)
        data = r.json()
        alphas = data if isinstance(data, list) else data.get("results", [])
        result = []
        for a in alphas:
            code = a.get("regular", {})
            expr = code.get("code", "") if isinstance(code, dict) else str(code)
            settings_data = a.get("settings", {})
            is_data = a.get("is", {})
            result.append({
                "alpha_id": a.get("id"),
                "expression": expr,
                "status": a.get("status"),
                "dateCreated": a.get("dateCreated"),
                "neutralization": settings_data.get("neutralization"),
                "sharpe": is_data.get("sharpe"),
                "fitness": is_data.get("fitness"),
                "returns": is_data.get("returns"),
                "turnover": is_data.get("turnover"),
            })
        return {"total": len(result), "alphas": result}

    def check_alpha_status(self, alpha_id: str) -> dict[str, Any]:
        data = self._fetch_alpha(alpha_id)
        if not data:
            return {"ok": False, "error": f"Alpha {alpha_id} not found or unavailable"}
        return {
            "ok": True,
            "alpha_id": alpha_id,
            "status": data.get("status"),
            "dateSubmitted": data.get("dateSubmitted"),
            "dateCreated": data.get("dateCreated"),
            "grade": data.get("grade"),
            "color": data.get("color"),
            "hidden": data.get("hidden"),
            "is": data.get("is", {}),
            "checks": data.get("checks", {}),
        }

    def submit_alpha(self, alpha_id: str) -> dict[str, Any]:
        r = self._get_session().post(f"{API_BASE}/alphas/{alpha_id}/submit")
        if r.status_code in (200, 201, 202):
            return {
                "ok": True,
                "alpha_id": alpha_id,
                "status_code": r.status_code,
                "detail": "submitted",
            }
        detail = r.text[:500]
        try:
            data = r.json()
            detail = data.get("detail") or detail
        except Exception:
            pass
        return {
            "ok": False,
            "alpha_id": alpha_id,
            "status_code": r.status_code,
            "detail": detail,
        }
