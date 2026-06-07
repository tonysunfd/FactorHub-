from __future__ import annotations

from .expression_adapter import ExpressionAdapter
from .quantgpt_client import QuantGPTClient
from .schemas import ResearchToolBaseRequest, RobustnessResponse


class RobustnessService:
    def __init__(self, client: QuantGPTClient | None = None):
        self.client = client or QuantGPTClient()

    async def anti_overfit(self, req: ResearchToolBaseRequest) -> RobustnessResponse:
        try:
            payload = req.model_dump()
            payload["expression"] = ExpressionAdapter.adapt(req.expression)
            raw = await self.client.run_anti_overfit(payload)
            if raw.get("error"):
                return RobustnessResponse(success=False, error=str(raw.get("error")), raw=raw)

            summary = {
                "score": raw.get("score"),
                "recommendation": raw.get("recommendation"),
                "passed_count": raw.get("passed_count"),
                "total_count": raw.get("total_count"),
            }
            details = {
                "tests": raw.get("tests") or [],
            }
            return RobustnessResponse(
                success=True,
                summary=summary,
                details=details,
                raw={**raw, "input_expression": req.expression, "adapted_expression": payload["expression"]},
            )
        except Exception as e:
            return RobustnessResponse(success=False, error=str(e), raw={"input_expression": req.expression})

    async def rolling_validation(self, req: ResearchToolBaseRequest) -> RobustnessResponse:
        try:
            payload = req.model_dump()
            payload["expression"] = ExpressionAdapter.adapt(req.expression)
            raw = await self.client.run_rolling_validation(payload)
            if raw.get("error"):
                return RobustnessResponse(success=False, error=str(raw.get("error")), raw=raw)

            summary = {
                "score": raw.get("score"),
                "summary": raw.get("summary"),
                "decay_analysis": raw.get("decay_analysis"),
            }
            details = {
                "windows": raw.get("windows") or [],
            }
            return RobustnessResponse(
                success=True,
                summary=summary,
                details=details,
                raw={**raw, "input_expression": req.expression, "adapted_expression": payload["expression"]},
            )
        except Exception as e:
            return RobustnessResponse(success=False, error=str(e), raw={"input_expression": req.expression})


robustness_service = RobustnessService()
