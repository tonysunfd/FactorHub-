from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from backend.services.expression_schema import EngineExecutionResult
from backend.services.factor_service import FactorCalculator

logger = logging.getLogger(__name__)


_NATIVE_CODE_SYSTEM_PROMPT = """你是一个量化研究员兼 Python 因子实现工程师。

你必须遵守以下规则：
1. 只返回 JSON 对象，不要输出 Markdown 或额外解释。
2. 输出字段必须包含 factor_name、implementation_code、implementation_notes。
3. implementation_code 必须是可执行的 Python 代码，且定义 def calculate_factor(df):。
4. 只能使用 df、pd、np，以及 df 中已有列；返回值必须是与 df.index 对齐的 pd.Series。
5. 不要读写文件，不要访问网络，不要 import 额外模块，不要包含测试代码。
6. 优先生成简单、稳健、可解释的实现，避免过深嵌套和过长中间链。
"""


class RDAgentNativeCodeExecutor:
    """使用 reference 风格的代码产物，但接入 FactorHub 本地数据与评估。"""

    def __init__(self) -> None:
        self._factor_calculator = FactorCalculator()

    @property
    def system_prompt(self) -> str:
        return _NATIVE_CODE_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        *,
        objective: str,
        hypothesis: dict[str, Any],
        candidate_fields: list[str],
        base_factors: list[str],
        known_implementations: list[str],
        accepted_code_examples: list[str],
    ) -> str:
        return (
            f"研究目标：{objective}\n"
            f"研究假设：{json.dumps(hypothesis, ensure_ascii=False)}\n"
            f"候选字段：{json.dumps(candidate_fields, ensure_ascii=False)}\n"
            f"基础因子：{json.dumps(base_factors, ensure_ascii=False)}\n"
            f"历史实现（避免重复）：{json.dumps(known_implementations[-10:], ensure_ascii=False)}\n"
            f"当前 SOTA 代码参考：{json.dumps(accepted_code_examples[-5:], ensure_ascii=False)}\n"
            "请输出 1 个可执行因子实现。必须返回 JSON，对应字段中的 implementation_code 需要定义 calculate_factor(df)。"
        )

    def extract_code_from_llm_response(self, payload: dict[str, Any]) -> tuple[str, str]:
        factor_name = str(payload.get("factor_name") or "RDAgentNativeFactor").strip() or "RDAgentNativeFactor"
        implementation_code = str(payload.get("implementation_code") or "").strip()
        if not implementation_code:
            raise ValueError("LLM 未返回 implementation_code")
        return factor_name, implementation_code

    def fallback_code(
        self,
        *,
        candidate_fields: list[str],
    ) -> tuple[str, str]:
        primary = candidate_fields[0] if candidate_fields else "close"
        secondary = candidate_fields[1] if len(candidate_fields) > 1 else primary
        return (
            "RDAgentFallbackCodeFactor",
            "\n".join(
                [
                    "def calculate_factor(df):",
                    f"    primary = pd.to_numeric(df['{primary}'], errors='coerce')",
                    f"    secondary = pd.to_numeric(df['{secondary}'], errors='coerce')",
                    "    signal = (primary.pct_change(5) - secondary.pct_change(10)).rolling(5, min_periods=1).mean()",
                    "    return pd.Series(signal, index=df.index, dtype=float)",
                ]
            ),
        )

    def execute_on_frame(self, stock_df: pd.DataFrame, code: str) -> pd.Series:
        return self._factor_calculator.calculate(stock_df, code)

    def execute_on_panel(
        self,
        *,
        stock_df: pd.DataFrame,
        code: str,
        factor_name: str,
    ) -> EngineExecutionResult:
        factor_series = self.execute_on_frame(stock_df, code)
        if not isinstance(factor_series, pd.Series):
            factor_series = pd.Series(factor_series, index=stock_df.index)
        factor_series = pd.to_numeric(factor_series, errors="coerce")
        return EngineExecutionResult(
            raw_expression=code,
            engine_type="rdagent_native_code",
            dialect="python_factor_function",
            factor_series=factor_series,
            metrics_source="rdagent_native_code_executor",
            diagnostics=[],
            canonical_expression=factor_name,
            canonical_ast=None,
            execution_meta={
                "implementation_code": code,
                "factor_name": factor_name,
            },
        )
