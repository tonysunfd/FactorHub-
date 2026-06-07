from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class EngineExecutionResult:
    raw_expression: str
    engine_type: str
    dialect: str
    factor_series: pd.Series | None
    metrics_source: str
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    canonical_expression: str | None = None
    canonical_ast: dict[str, Any] | None = None
    execution_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorEvaluationResult:
    expression: str
    raw_expression: str
    engine_type: str
    dialect: str
    canonical_expression: str | None
    canonical_ast: dict[str, Any] | None
    score: float
    grade: str
    report_metrics: dict[str, Any]
    backtest_summary: dict[str, Any]
    wq_brain: dict[str, Any]
    component_scores: dict[str, Any]
    anti_overfit: dict[str, Any]
    interpretation: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    report_url: str | None
    execution_meta: dict[str, Any] = field(default_factory=dict)
