from __future__ import annotations

from .expression_adapter import ExpressionAdapter
from .quantgpt_client import QuantGPTClient
from .schemas import ValidationResponse


class ValidationService:
    def __init__(self, client: QuantGPTClient | None = None):
        self.client = client or QuantGPTClient()

    async def validate_expression(self, expression: str, mode: str = "local") -> ValidationResponse:
        adapted_expression = ExpressionAdapter.adapt(expression)
        if mode == "local" and not self.client.is_configured():
            return ValidationResponse(
                success=True,
                valid=True,
                mode=mode,
                message="OK: 本地模式未配置 QuantGPT 上游，直接允许本地执行预检。",
                raw={
                    "input_expression": expression,
                    "adapted_expression": adapted_expression,
                    "bypassed_remote_validation": True,
                },
            )
        message = await self.client.validate_expression(adapted_expression, mode)
        valid = message.startswith("OK")
        return ValidationResponse(
            success=valid,
            valid=valid,
            mode=mode,
            message=message,
            raw={"upstream_message": message, "input_expression": expression, "adapted_expression": adapted_expression},
        )


validation_service = ValidationService()
