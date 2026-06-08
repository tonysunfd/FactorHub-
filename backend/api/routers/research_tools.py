"""统一研究工具 API。"""
from fastapi import APIRouter

from backend.services.research_tools.schemas import ResearchToolBaseRequest, ValidationRequest
from backend.services.research_tools.validation_service import validation_service
from backend.services.research_tools.scoring_service import scoring_service
from backend.services.research_tools.diagnosis_service import diagnosis_service
from backend.services.research_tools.robustness_service import robustness_service

router = APIRouter()


@router.post("/validate")
async def validate_expression(req: ValidationRequest):
    return await validation_service.validate_expression(req.expression, req.mode)


@router.post("/score")
async def score_factor(req: ResearchToolBaseRequest):
    return await scoring_service.score_factor(req)


@router.post("/diagnose")
async def diagnose_factor(req: ResearchToolBaseRequest):
    return await diagnosis_service.diagnose_factor(req)


@router.post("/anti-overfit")
async def anti_overfit(req: ResearchToolBaseRequest):
    return await robustness_service.anti_overfit(req)


@router.post("/rolling-validation")
async def rolling_validation(req: ResearchToolBaseRequest):
    return await robustness_service.rolling_validation(req)
