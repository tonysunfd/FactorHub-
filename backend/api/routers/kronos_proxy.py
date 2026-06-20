"""
Kronos WebUI 反向代理
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from backend.core.settings import settings

router = APIRouter()


async def _proxy(request: Request, path: str = "") -> Response:
    upstream = settings.KRONOS_UI_SERVICE_URL.rstrip("/")
    suffix = f"/{path}" if path else ""
    target_url = f"{upstream}{suffix}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    body = await request.body()

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        try:
            upstream_response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Kronos WebUI 不可用：{exc}") from exc

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "content-encoding", "connection"}
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


@router.api_route("", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_root(request: Request) -> Response:
    return await _proxy(request, "")


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_path(path: str, request: Request) -> Response:
    return await _proxy(request, path)
