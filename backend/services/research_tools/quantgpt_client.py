from __future__ import annotations

import json
import os
from typing import Any

import httpx

from backend.core.settings import settings


class QuantGPTClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        resolved_base_url = base_url or os.getenv("QUANTGPT_BASE_URL") or getattr(settings, "QUANTGPT_BASE_URL", "http://localhost:8003")
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout
        self.bearer_token = (os.getenv("QUANTGPT_BEARER_TOKEN") or getattr(settings, "QUANTGPT_BEARER_TOKEN", "") or "").strip()

    def _headers(self, accept: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if accept:
            headers["Accept"] = accept
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}{path}", params=params, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}{path}", json=payload, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def _extract_mcp_payload(self, data: Any) -> Any:
        if isinstance(data, dict):
            if "result" in data:
                return self._extract_mcp_payload(data["result"])
            structured = data.get("structuredContent")
            if isinstance(structured, dict):
                result_value = structured.get("result") if set(structured.keys()) == {"result"} else None
                if isinstance(result_value, str):
                    try:
                        return json.loads(result_value)
                    except json.JSONDecodeError:
                        return {"message": result_value}
                return structured
            content = data.get("content")
            if isinstance(content, list) and content:
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                return {"message": text}
        return data

    async def mcp_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/mcp/",
                json=payload,
                headers=self._headers("application/json, text/event-stream"),
            )
            resp.raise_for_status()
            text = resp.text.strip()

        if "data:" in text:
            lines = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
            for line in reversed(lines):
                if not line or line == "[DONE]":
                    continue
                try:
                    msg = json.loads(line)
                    return self._extract_mcp_payload(msg)
                except json.JSONDecodeError:
                    continue

        try:
            return self._extract_mcp_payload(resp.json())
        except Exception:
            return {"raw": text}

    async def validate_expression(self, expression: str, mode: str = "local") -> str:
        data = await self.mcp_call("validate_expression", {
            "expression": expression,
            "mode": mode,
        })
        if isinstance(data, dict):
            return data.get("message") or data.get("result") or data.get("raw") or json.dumps(data, ensure_ascii=False)
        return str(data)

    async def score_factor(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self.mcp_call("score_factor", payload)
        return data if isinstance(data, dict) else {"message": str(data)}

    async def diagnose_factor(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self.mcp_call("diagnose_factor", payload)
        return data if isinstance(data, dict) else {"message": str(data)}

    async def run_anti_overfit(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self.mcp_call("run_anti_overfit", payload)
        return data if isinstance(data, dict) else {"message": str(data)}

    async def run_rolling_validation(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self.mcp_call("run_rolling_validation", payload)
        return data if isinstance(data, dict) else {"message": str(data)}

    async def wq_status(self, account: str = "primary") -> dict[str, Any]:
        return await self._get_json("/api/v1/wq-brain/status", {"account": account})

    async def wq_user_info(self, account: str = "primary") -> dict[str, Any]:
        return await self._get_json("/api/v1/wq-brain/user-info", {"account": account})

    async def wq_platform_alphas(self, account: str = "primary", limit: int = 50) -> dict[str, Any]:
        return await self._get_json("/api/v1/wq-brain/platform-alphas", {"account": account, "limit": limit})

    async def wq_submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/api/v1/wq-brain/submit", payload)

    async def wq_batch_submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/api/v1/wq-brain/batch-submit", payload)

    async def wq_check_alphas(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/api/v1/wq-brain/batch-alpha-status", payload)

    async def wq_finalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/api/v1/wq-brain/batch-finalize", payload)
