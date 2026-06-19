"""
LLM 配置服务。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Any

from backend.core.settings import settings


class LLMConfigService:
    """管理本地 LLM 配置文件。"""

    def __init__(self) -> None:
        self.config_path = settings.CONFIG_DIR / "llm_config.json"

    def _default_config(self) -> dict[str, Any]:
        return {
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        }

    @staticmethod
    def _running_in_container() -> bool:
        return Path("/.dockerenv").exists() or os.getenv("FACTORHUB_RUNNING_IN_CONTAINER") == "1"

    def _normalize_runtime_base_url(self, base_url: str) -> str:
        normalized = str(base_url or "").strip()
        if not normalized:
            return normalized
        if not self._running_in_container():
            return normalized

        try:
            parsed = urlsplit(normalized)
        except Exception:
            return normalized

        hostname = (parsed.hostname or "").strip().lower()
        if hostname not in {"127.0.0.1", "localhost"}:
            return normalized

        replacement_host = os.getenv("FACTORHUB_CONTAINER_HOST_GATEWAY", "host.docker.internal").strip() or "host.docker.internal"
        netloc = replacement_host
        if parsed.port:
            netloc = f"{replacement_host}:{parsed.port}"
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo = f"{userinfo}:{parsed.password}"
            netloc = f"{userinfo}@{netloc}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    def load_raw_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return self._default_config()

        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_config()

        config = self._default_config()
        config.update(
            {
                "api_key": str(raw.get("api_key") or ""),
                "base_url": str(raw.get("base_url") or config["base_url"]),
                "model": str(raw.get("model") or config["model"]),
            }
        )
        return config

    def save_config(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        config = self.load_raw_config()
        if api_key is not None:
            config["api_key"] = api_key.strip()
        if base_url is not None and base_url.strip():
            config["base_url"] = base_url.strip()
        if model is not None and model.strip():
            config["model"] = model.strip()

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.get_public_config()

    def get_runtime_config(self) -> dict[str, Any]:
        config = self.load_raw_config()
        config["base_url"] = self._normalize_runtime_base_url(str(config.get("base_url") or ""))
        return config

    def get_public_config(self) -> dict[str, Any]:
        config = self.load_raw_config()
        api_key = config.get("api_key", "")
        masked = ""
        if api_key:
            if len(api_key) <= 8:
                masked = "*" * len(api_key)
            else:
                masked = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"

        return {
            "has_api_key": bool(api_key),
            "api_key_masked": masked,
            "base_url": config.get("base_url"),
            "model": config.get("model"),
        }


llm_config_service = LLMConfigService()
