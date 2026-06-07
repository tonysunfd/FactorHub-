"""
LLM 配置 API 路由。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.auto_factor_mining_service import auto_factor_mining_service
from backend.services.llm_config_service import llm_config_service

router = APIRouter()


class LLMConfigUpdateRequest(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


@router.get("/config")
async def get_llm_config():
    """获取 LLM 配置状态。"""
    return llm_config_service.get_public_config()


@router.post("/config")
async def save_llm_config(request: LLMConfigUpdateRequest):
    """保存 LLM 配置。"""
    try:
        data = llm_config_service.save_config(
            api_key=request.api_key,
            base_url=request.base_url,
            model=request.model,
        )
        return {
            **data,
            "message": "LLM 配置已保存",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/restart")
async def restart_llm_service():
    """检查 LLM 服务配置是否可用。"""
    try:
        data = auto_factor_mining_service.get_llm_status()
        if not data.get("has_api_key"):
            return {
                **data,
                "message": "LLM 未配置 API Key，当前仅可使用非 LLM 兜底候选生成。",
            }
        return {
            **data,
            "message": "LLM 配置已加载，可继续自动挖掘。",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
