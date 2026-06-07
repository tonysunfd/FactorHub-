from __future__ import annotations

from .expression_adapter import ExpressionAdapter
from .quantgpt_client import QuantGPTClient
from .schemas import DiagnosisResponse, ResearchToolBaseRequest


class DiagnosisService:
    def __init__(self, client: QuantGPTClient | None = None):
        self.client = client or QuantGPTClient()

    async def diagnose_factor(self, req: ResearchToolBaseRequest) -> DiagnosisResponse:
        try:
            adapted_expression = ExpressionAdapter.adapt(req.expression)
            score_payload = req.model_dump()
            score_payload["expression"] = adapted_expression
            score_raw = await self.client.score_factor(score_payload)
            if score_raw.get("error"):
                return DiagnosisResponse(success=False, error=str(score_raw.get("error")), raw={"score": score_raw})

            metrics = score_raw.get("key_metrics") or score_raw.get("metrics") or {}
            diagnose_payload = {
                "expression": adapted_expression,
                "ic_mean": metrics.get("ic_mean", 0) or 0,
                "ic_ir": metrics.get("ic_ir", 0) or 0,
                "monotonicity_score": metrics.get("monotonicity", 0) or metrics.get("monotonicity_score", 0) or 0,
                "score": score_raw.get("score", 50) or 50,
            }
            raw = await self.client.diagnose_factor(diagnose_payload)
            if raw.get("error"):
                return DiagnosisResponse(success=False, error=str(raw.get("error")), raw={"score": score_raw, "diagnosis": raw})

            strategy = raw.get("strategy")
            reason = raw.get("reason")
            details = raw.get("details")
            mutation_prompt = raw.get("mutation_prompt") or {}

            report_parts = []
            if strategy:
                report_parts.append(f"策略：{strategy}")
            if reason:
                report_parts.append(f"原因：{reason}")
            if isinstance(details, dict) and details:
                report_parts.append(f"细节：{details}")
            report = "\n\n".join(report_parts) if report_parts else (raw.get("report") or raw.get("message"))

            improvement_suggestions: list[str] = []
            if reason:
                improvement_suggestions.append(reason)
            if isinstance(details, dict) and details.get("suggested_replacements"):
                improvement_suggestions.append(f"建议替换：{details['suggested_replacements']}")
            user_prompt = mutation_prompt.get("user")
            if user_prompt:
                improvement_suggestions.append(user_prompt)
            elif details:
                improvement_suggestions.append(str(details))

            key_findings = [item for item in [strategy, reason] if item]
            return DiagnosisResponse(
                success=True,
                report=report,
                key_findings=key_findings,
                improvement_suggestions=improvement_suggestions,
                raw={
                    "score": score_raw,
                    "diagnosis": raw,
                    "input_expression": req.expression,
                    "adapted_expression": adapted_expression,
                },
            )
        except Exception as e:
            return DiagnosisResponse(success=False, error=str(e), raw={})


diagnosis_service = DiagnosisService()
