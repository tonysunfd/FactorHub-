"""WQ BRAIN API。"""
from fastapi import APIRouter, Query

from backend.services.research_tools.schemas import (
    WQBrainAlphaIdsRequest,
    WQBrainBatchSubmitRequest,
    WQBrainCandidateSyncRequest,
    WQBrainConfigUpdateRequest,
    WQBrainSubmitRequest,
)
from backend.services.research_tools.wqbrain_service import wqbrain_service

router = APIRouter()


@router.get("/config")
async def get_config():
    return await wqbrain_service.get_config()


@router.post("/config")
async def update_config(req: WQBrainConfigUpdateRequest):
    return await wqbrain_service.update_config(req.model_dump())


@router.get("/status")
async def get_status(account: str = Query("primary")):
    return await wqbrain_service.status(account)


@router.get("/user-info")
async def get_user_info(account: str = Query("primary")):
    return await wqbrain_service.user_info(account)


@router.get("/platform-alphas")
async def get_platform_alphas(account: str = Query("primary"), limit: int = Query(50)):
    return await wqbrain_service.platform_alphas(account, limit)


@router.get("/candidates")
async def get_candidates(limit: int = Query(100)):
    return await wqbrain_service.candidates(limit)


@router.post("/submit")
async def submit_alpha(req: WQBrainSubmitRequest):
    return await wqbrain_service.submit(req.model_dump())


@router.post("/batch-submit")
async def batch_submit(req: WQBrainBatchSubmitRequest):
    return await wqbrain_service.batch_submit(req.model_dump())


@router.post("/check-alphas")
async def check_alphas(req: WQBrainAlphaIdsRequest):
    return await wqbrain_service.check_alphas(req.model_dump())


@router.post("/finalize")
async def finalize(req: WQBrainAlphaIdsRequest):
    return await wqbrain_service.finalize(req.model_dump())


@router.post("/sync-candidates")
async def sync_candidates(req: WQBrainCandidateSyncRequest):
    return await wqbrain_service.sync_candidates(req.model_dump())
