from __future__ import annotations

from .expression_adapter import ExpressionAdapter
from .quantgpt_client import QuantGPTClient
from .schemas import ResearchToolBaseRequest, ScoreResponse


class ScoringService:
    def __init__(self, client: QuantGPTClient | None = None):
        self.client = client or QuantGPTClient()

    async def score_factor(self, req: ResearchToolBaseRequest) -> ScoreResponse:
        try:
            payload = req.model_dump()
            payload["expression"] = ExpressionAdapter.adapt(req.expression)
            raw = await self.client.score_factor(payload)
            if raw.get("error"):
                return ScoreResponse(success=False, error=str(raw.get("error")), raw=raw)

            score = raw.get("score")
            grade = raw.get("grade")
            component_scores = raw.get("component_scores")
            metrics = raw.get("metrics") or raw.get("key_metrics")
            summary = raw.get("summary")
            if not summary:
                summary = f"因子综合评分 {score if score is not None else '--'} / 100，等级 {grade or '--'}"

            return ScoreResponse(
                success=True,
                score=score,
                grade=grade,
                summary=summary,
                component_scores=component_scores,
                metrics=metrics,
                raw={**raw, "input_expression": req.expression, "adapted_expression": payload["expression"]},
            )
        except Exception as e:
            return ScoreResponse(success=False, error=str(e), raw={"input_expression": req.expression})


scoring_service = ScoringService()
