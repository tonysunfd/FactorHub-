"""
自动因子挖掘服务。
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.repositories.factor_repository import FactorRepository
from backend.services.expression_schema import FactorEvaluationResult
from backend.services.factor_evaluation_service import FactorEvaluationService
from backend.services.factor_generator_service import factor_generator_service
from backend.services.quantgpt_expression_engine import QuantGPTExpressionEngine, normalize_expression as normalize_canonical_expression
from backend.services.research_tools.diagnosis_service import diagnosis_service
from backend.services.research_tools.expression_adapter import ExpressionAdapter
from backend.services.research_tools.factor_selection_service import (
    build_llm_factor_selector_prompt,
    factor_selection_service,
    infer_primary_problem_from_metrics,
)
from backend.services.research_tools.quantgpt_client import QuantGPTClient
from backend.services.research_tools.schemas import ResearchToolBaseRequest
from backend.services.research_tools.validation_service import validation_service
from backend.services.report_service import generate_report
from backend.services.llm_config_service import llm_config_service

logger = logging.getLogger(__name__)

AUTO_MINING_REPORT_DIR = settings.REPORTS_DIR / "auto_mining"
AUTO_MINING_REPORT_DIR.mkdir(parents=True, exist_ok=True)

_LLM_SYSTEM_PROMPT = """你是一个量化因子表达式生成器。

你必须遵守以下规则：
1. 只返回 JSON 数组，每个元素是一条因子表达式字符串。
2. 表达式只使用当前系统已经支持的行情字段、基础因子和常见技术指标函数。
3. 优先生成可解释、可执行、具有多样性的复合因子表达式，不要只改一个窗口参数。
4. 避免与已存在候选重复；优先提升 rankIC、L/S Sharpe、收益稳定性和 WQ Fitness。
5. 不要输出解释、Markdown、注释或额外字段。
"""

_FACTOR_SELECTOR_SYSTEM_PROMPT = """你是一个量化研究员。

你必须遵守以下规则：
1. 只允许从给定候选列表中选择基础因子。
2. 只返回 JSON 对象，不要输出 Markdown、解释性前缀或额外文本。
3. 输出字段必须包含 selected_factors、selection_rationale、per_factor_reason。
4. selected_factors 中的名字必须与候选列表完全一致。
5. 优先保证因子语义互补、与目标匹配，并符合场景可计算性约束。
"""

_VALID_SELECTION_DIRECTIONS = {
    "score",
    "ls_sharpe",
    "ls_return",
    "wq_rating",
    "wq_fitness",
    "wq_return",
    "report_sharpe",
}

_SELECTION_DIRECTION_ALIASES = {
    "stability": "report_sharpe",
    "stable": "report_sharpe",
    "sharpe": "report_sharpe",
    "return": "ls_return",
    "fitness": "wq_fitness",
}

_ROUND_PROBLEM_KEYWORDS = {
    "rank_ic": ["rankic", "rank ic", "横截面", "区分度", "排序能力"],
    "ls_sharpe": ["sharpe", "收益质量", "风险调整后收益", "l/s sharpe"],
    "turnover": ["换手", "可交易性", "turnover"],
    "ls_return": ["收益偏弱", "超额收益", "收益释放", "return", "收益不足"],
    "drawdown": ["回撤", "max drawdown", "drawdown"],
    "volatility": ["波动", "volatility", "净值稳定性"],
    "score": ["综合评分", "score", "可用性"],
    "wq_fitness": ["wq", "fitness"],
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _tokenize_text(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", value.lower()) if token]


def _normalize_expression(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _extract_text_from_llm_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    text_parts.append(normalized)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "text":
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
                continue
            if item_type == "output_text":
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
        return "\n".join(text_parts).strip()
    return str(content or "").strip()


def _classify_round_problem(text: str) -> str:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return ""
    for label, keywords in _ROUND_PROBLEM_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return label
    return ""


def _rank_round_problem_candidates(
    *,
    direction: str,
    report_metrics: dict[str, Any],
    backtest_summary: dict[str, Any],
    score: float,
) -> list[str]:
    report_sharpe = _safe_float(report_metrics.get("sharpe"))
    report_drawdown = abs(_safe_float(report_metrics.get("max_drawdown")))
    report_volatility = _safe_float(report_metrics.get("volatility"))
    ls_sharpe = _safe_float(backtest_summary.get("long_short_sharpe"))
    ls_return = _safe_float(backtest_summary.get("long_short_annual"))
    rank_ic = abs(_safe_float(backtest_summary.get("rank_ic_mean") or backtest_summary.get("ic_mean")))
    turnover = _safe_float(backtest_summary.get("turnover"))
    normalized_direction = str(direction or "").strip().lower()

    priorities: list[tuple[str, float]] = [
        ("ls_sharpe", max(1.0 - ls_sharpe, 0.0) + max(1.0 - report_sharpe, 0.0)),
        ("rank_ic", max(0.02 - rank_ic, 0.0) * 50.0),
        ("turnover", max(turnover - 0.35, 0.0) * 5.0),
        ("drawdown", max(report_drawdown - 0.2, 0.0) * 5.0),
        ("volatility", max(report_volatility - 0.3, 0.0) * 5.0),
        ("ls_return", max(0.12 - ls_return, 0.0) * 8.0),
        ("score", max(70.0 - score, 0.0) / 20.0),
    ]
    if normalized_direction == "ls_sharpe":
        priorities.append(("ls_sharpe", 1.5))
    elif normalized_direction == "ls_return":
        priorities.append(("ls_return", 1.5))
    elif normalized_direction == "report_sharpe":
        priorities.append(("ls_sharpe", 1.2))
    elif normalized_direction == "wq_fitness":
        priorities.append(("wq_fitness", 1.2))

    merged: dict[str, float] = {}
    for label, weight in priorities:
        merged[label] = merged.get(label, 0.0) + weight
    return [
        label
        for label, weight in sorted(merged.items(), key=lambda item: item[1], reverse=True)
        if weight > 0
    ]


def _build_metric_specific_problem_text(
    *,
    label: str,
    report_metrics: dict[str, Any],
    backtest_summary: dict[str, Any],
    score: float,
) -> str:
    report_sharpe = _safe_float(report_metrics.get("sharpe"))
    report_drawdown = abs(_safe_float(report_metrics.get("max_drawdown")))
    report_volatility = _safe_float(report_metrics.get("volatility"))
    ls_sharpe = _safe_float(backtest_summary.get("long_short_sharpe"))
    ls_return = _safe_float(backtest_summary.get("long_short_annual"))
    rank_ic = _safe_float(backtest_summary.get("rank_ic_mean") or backtest_summary.get("ic_mean"))
    turnover = _safe_float(backtest_summary.get("turnover"))
    abs_rank_ic = abs(rank_ic)

    if label == "ls_sharpe":
        if ls_sharpe < 0:
            return f"L/S Sharpe 已转负（{ls_sharpe:.2f}），收益质量明显恶化，当前结构需要先止损并恢复稳定性。"
        if report_sharpe < 0.5:
            return f"Report Sharpe 偏低（{report_sharpe:.2f}），风险调整后收益不足，说明当前收益质量还不够稳定。"
        return f"L/S Sharpe 偏低（{ls_sharpe:.2f}），收益质量仍需改善。"
    if label == "rank_ic":
        if rank_ic < 0:
            return f"横截面 rankIC 为负（{rank_ic:.4f}），说明当前表达式的排序方向已经失真，选股区分度不足。"
        return f"横截面 rankIC 偏弱（{rank_ic:.4f}），说明选股区分度不足。"
    if label == "turnover":
        return f"换手率偏高（{turnover:.2f}），真实可交易性承压，需要优先压低噪声和频繁切换。"
    if label == "drawdown":
        return f"最大回撤偏大（{report_drawdown:.2f}），回撤约束已经成为当前主要短板。"
    if label == "volatility":
        return f"波动偏高（{report_volatility:.2f}），净值稳定性不足，需要优先收敛波动暴露。"
    if label == "ls_return":
        if ls_return < 0:
            return f"L/S 年化收益为负（{ls_return:.2%}），收益端已经明显失效，需要先恢复正向收益。"
        return f"L/S 年化收益偏弱（{ls_return:.2%}），当前结构的超额收益释放不足。"
    if label == "score":
        return f"综合评分偏低（{score:.2f}），当前表达式质量和可用性仍然不足。"
    if label == "wq_fitness":
        return "WQ Fitness 偏弱，当前表达式在平台视角下的可提交性和质量仍需改善。"
    return infer_primary_problem_from_metrics(report_metrics, backtest_summary, score)


def _is_parent_smoothing_expression(expression: str, parent_expression: str) -> bool:
    normalized_expression = _normalize_semantic_expression(expression)
    normalized_parent = _normalize_semantic_expression(parent_expression)
    if not normalized_expression or not normalized_parent:
        return False
    if normalized_expression == normalized_parent:
        return False
    if normalized_parent not in normalized_expression:
        return False
    smoothing_markers = ("ts_mean(", "ts_zscore(", "rank(", "ts_std(")
    return any(marker in normalized_expression for marker in smoothing_markers)


def _grade_from_score(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _normalize_expression(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(str(value).strip())
    return result


def _normalize_selection_direction(direction: str | None, selection_mode: str = "auto") -> str:
    normalized = str(direction or "").strip().lower()
    if not normalized:
        return "report_sharpe" if str(selection_mode or "").strip().lower() == "manual_genetic" else "score"
    normalized = _SELECTION_DIRECTION_ALIASES.get(normalized, normalized)
    if normalized in _VALID_SELECTION_DIRECTIONS:
        return normalized
    return "report_sharpe" if str(selection_mode or "").strip().lower() == "manual_genetic" else "score"


def _campaign_metric_key(direction: str | None) -> str:
    normalized_direction = str(direction or "score").strip().lower()
    return {
        "ls_sharpe": "ls_sharpe",
        "ls_return": "ls_return",
        "report_sharpe": "report_sharpe",
        "wq_fitness": "wq_fitness",
        "wq_return": "wq_return",
    }.get(normalized_direction, "score")


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _component_tokens(values: Any) -> list[str]:
    if isinstance(values, (list, tuple, set)):
        tokens: list[str] = []
        for value in values:
            normalized = str(value or "").strip().lower()
            if normalized:
                tokens.append(normalized)
        return _dedupe_preserve_order(tokens)
    return []


def _reorder_base_factor_codes_for_continuation(
    base_factor_codes: list[str],
    continuation_context: dict[str, Any] | None,
) -> list[str]:
    deduped = [code for code in _dedupe_preserve_order(base_factor_codes) if code]
    if not continuation_context or not deduped:
        return deduped

    previous_codes = {
        _normalize_expression(code)
        for code in (
            continuation_context.get("previous_base_factor_codes")
            or continuation_context.get("previous_base_factors")
            or []
        )
        if _normalize_expression(code)
    }
    if not previous_codes:
        return deduped

    appended = [code for code in deduped if _normalize_expression(code) not in previous_codes]
    retained = [code for code in deduped if _normalize_expression(code) in previous_codes]
    return appended + retained


def _get_new_base_factor_codes(
    base_factor_codes: list[str],
    continuation_context: dict[str, Any] | None,
) -> list[str]:
    ordered_codes = _reorder_base_factor_codes_for_continuation(base_factor_codes, continuation_context)
    if not continuation_context:
        return []
    previous_codes = {
        _normalize_expression(code)
        for code in (continuation_context.get("previous_base_factor_codes") or [])
        if _normalize_expression(code)
    }
    return [
        code for code in ordered_codes
        if _normalize_expression(code) and _normalize_expression(code) not in previous_codes
    ]


def _expression_uses_code(expression: str, code: str) -> bool:
    normalized_expression = _normalize_semantic_expression(expression)
    normalized_code = _normalize_semantic_expression(code)
    return bool(normalized_expression and normalized_code and normalized_code in normalized_expression)


def _expression_contains_parent_anchor(expression: str, parent_expression: str) -> bool:
    normalized_expression = _normalize_semantic_expression(expression)
    normalized_parent = _normalize_semantic_expression(parent_expression)
    return bool(normalized_expression and normalized_parent and normalized_parent in normalized_expression)


def _normalize_semantic_expression(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    adapted = ExpressionAdapter.adapt(raw)
    return normalize_canonical_expression(adapted)


class AutoFactorMiningService:
    """基于 QuantGPT 风格的自动因子挖掘服务。"""

    def __init__(self) -> None:
        self._data_service = None
        self._factor_evaluation_service = FactorEvaluationService()
        self._quantgpt_engine = QuantGPTExpressionEngine()

    @property
    def data_service(self):
        if self._data_service is None:
            from backend.data.service import get_data_service

            self._data_service = get_data_service()
        return self._data_service

    def get_llm_status(self) -> dict[str, Any]:
        return llm_config_service.get_public_config()

    def get_report_path(self, filename: str) -> Path:
        safe_name = Path(filename).name
        return AUTO_MINING_REPORT_DIR / safe_name

    def select_factors(
        self,
        *,
        prompt: str,
        max_factor_count: int = 12,
        candidate_limit: int = 80,
        selection_mode: str = "auto",
        direction: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        universe: str | None = None,
        benchmark: str | None = None,
        extra_context: str | None = None,
        exclude_factors: list[str] | None = None,
        continuation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_selection_mode = str(selection_mode or "auto").strip().lower()
        normalized_direction = _normalize_selection_direction(direction, normalized_selection_mode)
        candidates = factor_selection_service.load_factor_candidates_for_llm(
            limit=max(int(candidate_limit or 0), 1),
            selection_mode=normalized_selection_mode,
        )
        if not candidates:
            return {
                "selected_factors": [],
                "selection_rationale": "当前因子库为空，无法筛选基础因子。",
                "per_factor_reason": {},
            }

        exclude = {str(name).strip() for name in (exclude_factors or []) if str(name or "").strip()}
        filtered_candidates = [item for item in candidates if item.get("name") not in exclude]
        if not filtered_candidates:
            return {
                "selected_factors": [],
                "selection_rationale": "候选因子已被排除列表过滤完毕，无法继续筛选。",
                "per_factor_reason": {},
            }

        filtered_candidates = self._prioritize_continuation_candidates(
            candidates=filtered_candidates,
            continuation_context=continuation_context,
            max_factor_count=max_factor_count,
        )

        llm_config = llm_config_service.get_runtime_config()
        if not llm_config.get("api_key"):
            raise ValueError("LLM 未配置 API Key，无法执行真实因子筛选。")

        payload = type(
            "FactorSelectionRequest",
            (),
            {
                "prompt": prompt,
                "direction": normalized_direction,
                "start_date": start_date or "",
                "end_date": end_date or "",
                "universe": universe or "",
                "benchmark": benchmark or "",
                "max_factor_count": max_factor_count,
                "selection_mode": normalized_selection_mode,
            },
        )()
        llm_prompt = build_llm_factor_selector_prompt(payload, filtered_candidates)
        continuation_instructions = ""
        if continuation_context:
            continuation_instructions = str(continuation_context.get("selection_instructions") or "").strip()
        if continuation_instructions:
            llm_prompt = f"{llm_prompt}\n\n连续探索约束：\n{continuation_instructions}"
        if normalized_selection_mode == "manual_genetic":
            llm_prompt = (
                f"{llm_prompt}\n\n手动遗传挖掘硬约束：\n"
                "只选择可在单股票 OHLCV 时间序列上直接计算的 seed/base factors；"
                "不要选择横截面、股票池依赖、目标股票绑定，或自动/RDAgent 已挖掘出的复合表达式。"
            )
        if extra_context:
            llm_prompt = f"{llm_prompt}\n\n补充上下文：\n{extra_context.strip()}"

        llm_result = self._select_factors_with_llm(
            prompt=llm_prompt,
            max_factor_count=max_factor_count,
            candidates=filtered_candidates,
            llm_config=llm_config,
        )
        selected_names = llm_result["selected_factors"]
        return {
            "selected_factors": selected_names,
            "selection_rationale": llm_result["selection_rationale"],
            "per_factor_reason": llm_result["per_factor_reason"],
            "llm_used": True,
            "llm_call_mode": "live_api",
            "llm_model": llm_config.get("model") or "deepseek-chat",
            "llm_base_url": llm_config.get("base_url") or "https://api.deepseek.com/v1",
            "llm_response_id": llm_result.get("llm_response_id"),
            "llm_provider": llm_result.get("llm_provider") or "openai_compatible",
            "llm_evidence": {
                "call_mode": "live_api",
                "provider": llm_result.get("llm_provider") or "openai_compatible",
                "model": llm_config.get("model") or "deepseek-chat",
                "base_url": llm_config.get("base_url") or "https://api.deepseek.com/v1",
                "response_id": llm_result.get("llm_response_id"),
            },
            "selection_mode": normalized_selection_mode,
            "candidate_count": len(filtered_candidates),
        }

    def _select_factors_with_llm(
        self,
        *,
        prompt: str,
        max_factor_count: int,
        candidates: list[dict[str, Any]],
        llm_config: dict[str, Any],
    ) -> dict[str, Any]:
        client = QuantGPTClient(base_url=llm_config.get("base_url") or "https://api.deepseek.com/v1")
        llm_response = self._run_async_tool(
            client.chat_json(
                api_key=llm_config["api_key"],
                model=llm_config.get("model") or "deepseek-chat",
                base_url=llm_config.get("base_url") or "https://api.deepseek.com/v1",
                system_prompt=_FACTOR_SELECTOR_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=1800,
            )
        )
        parsed = llm_response.get("content") or {}

        candidate_name_map = {str(item.get('name')): item for item in candidates if item.get("name")}
        selected_names = factor_selection_service.dedupe_factor_names(parsed.get("selected_factors") or [])
        selected_names = [name for name in selected_names if name in candidate_name_map][: max(int(max_factor_count or 0), 1)]
        if not selected_names:
            raise ValueError("LLM 未返回任何有效候选因子，请检查提示词或模型输出。")

        raw_reason_map = parsed.get("per_factor_reason") if isinstance(parsed.get("per_factor_reason"), dict) else {}
        per_factor_reason = {
            name: str(raw_reason_map.get(name) or "LLM 认为该因子与当前研究目标匹配。").strip()
            for name in selected_names
        }
        selection_rationale = str(parsed.get("selection_rationale") or "").strip() or "LLM 已根据研究目标完成基础因子筛选。"
        return {
            "selected_factors": selected_names,
            "selection_rationale": selection_rationale,
            "per_factor_reason": per_factor_reason,
            "llm_response_id": str(llm_response.get("response_id") or "").strip(),
            "llm_provider": str(llm_response.get("provider") or "openai_compatible").strip() or "openai_compatible",
        }

    def resolve_base_factor_codes(self, base_factors: list[str]) -> list[str]:
        if not base_factors:
            return []

        db = get_db_session()
        try:
            repo = FactorRepository(db)
            codes: list[str] = []
            for factor_name in base_factors:
                factor = repo.get_by_name(factor_name)
                if factor and factor.code:
                    codes.append(factor.code)
            return codes
        finally:
            db.close()

    def build_continuation_context(
        self,
        *,
        result: dict[str, Any] | None,
        request_payload: dict[str, Any] | None = None,
        prompt: str | None = None,
        factor_update_mode: str = "append",
        additional_factor_count: int = 5,
    ) -> dict[str, Any]:
        result = result or {}
        request_payload = request_payload or {}
        best_factor = (result.get("factors") or [{}])[0] if result.get("factors") else {}
        round_evaluation = result.get("round_evaluation") or best_factor.get("task_details", {}).get("round_evaluation") or {}
        interpretation = best_factor.get("interpretation") or best_factor.get("task_details", {}).get("interpretation") or {}
        report_metrics = best_factor.get("report_metrics") or {}
        backtest_summary = best_factor.get("backtest_summary") or {}
        base_factors = (
            round_evaluation.get("base_factors")
            or result.get("round_evaluation", {}).get("base_factors")
            or request_payload.get("base_factors")
            or []
        )
        previous_base_factors = request_payload.get("base_factors") or []
        if previous_base_factors and base_factors and not set(previous_base_factors).intersection(base_factors):
            previous_base_factors = list(base_factors)

        weaknesses: list[str] = []
        for value in (
            interpretation.get("weaknesses"),
            interpretation.get("risks"),
            interpretation.get("limitations"),
            round_evaluation.get("primary_problem"),
            round_evaluation.get("secondary_problem"),
        ):
            if isinstance(value, list):
                weaknesses.extend(str(item) for item in value if item)
            elif value:
                weaknesses.append(str(value))

        suggestions: list[str] = []
        for value in (
            interpretation.get("next_steps"),
            interpretation.get("improvement_ideas"),
            round_evaluation.get("suggested_actions"),
        ):
            if isinstance(value, list):
                suggestions.extend(str(item) for item in value if item)
            elif value:
                suggestions.append(str(value))

        primary_problem = round_evaluation.get("primary_problem") or (weaknesses[0] if weaknesses else "需要提升综合稳定性")
        secondary_problem = round_evaluation.get("secondary_problem") or ""
        if not secondary_problem:
            for weakness in weaknesses:
                normalized = str(weakness or "").strip()
                if normalized and normalized != primary_problem:
                    secondary_problem = normalized
                    break
        recommended_goal = round_evaluation.get("recommended_goal") or prompt or "提升综合分数与稳定性"
        metric_snapshot = {
            "score": best_factor.get("score"),
            "report_sharpe": report_metrics.get("sharpe"),
            "report_max_drawdown": report_metrics.get("max_drawdown"),
            "ls_sharpe": backtest_summary.get("long_short_sharpe"),
            "ls_return": backtest_summary.get("long_short_annual"),
            "rank_ic": backtest_summary.get("rank_ic_mean"),
            "turnover": backtest_summary.get("turnover"),
            "wq_fitness": backtest_summary.get("wq_fitness"),
        }
        canonical_ast = (
            best_factor.get("canonical_ast")
            or best_factor.get("task_details", {}).get("canonical_ast")
            or {}
        )
        parent_expression = (
            best_factor.get("expression")
            or best_factor.get("canonical_expression")
            or best_factor.get("task_details", {}).get("expression")
            or ""
        )
        parent_raw_expression = (
            best_factor.get("raw_expression")
            or best_factor.get("task_details", {}).get("raw_expression")
            or parent_expression
        )
        base_factor_components = self._collect_base_factor_component_tokens(base_factors)
        recent_expression_components = {
            "fields": _component_tokens(canonical_ast.get("fields")),
            "operators": _component_tokens(canonical_ast.get("operators")),
        }
        factor_usage = self._build_factor_usage_summary(
            current_base_factors=list(base_factors),
            previous_base_factors=list(previous_base_factors),
            best_factor=best_factor,
        )

        summary_text = "；".join(_dedupe_preserve_order([primary_problem, *weaknesses[:2], *suggestions[:2]]))
        selection_hints = self._build_continuation_selection_hints(
            primary_problem=primary_problem,
            secondary_problem=secondary_problem,
            recommended_goal=recommended_goal,
            suggested_actions=_dedupe_preserve_order(suggestions),
            metric_snapshot=metric_snapshot,
            used_new_factors=list((factor_usage or {}).get("used_new_factors") or []),
            unused_new_factors=list((factor_usage or {}).get("unused_new_factors") or []),
        )
        return {
            "base_factors": list(base_factors),
            "previous_base_factors": list(previous_base_factors),
            "primary_problem": primary_problem,
            "secondary_problem": secondary_problem,
            "recommended_goal": recommended_goal,
            "suggested_actions": _dedupe_preserve_order(suggestions),
            "weaknesses": _dedupe_preserve_order(weaknesses),
            "metric_snapshot": metric_snapshot,
            "summary_text": summary_text,
            "factor_update_mode": factor_update_mode,
            "additional_factor_count": max(additional_factor_count, 1),
            "selection_instructions": selection_hints["selection_instructions"],
            "preferred_keywords": selection_hints["preferred_keywords"],
            "avoid_keywords": selection_hints["avoid_keywords"],
            "selection_confidence": selection_hints["selection_confidence"],
            "should_adjust_base_factors": selection_hints["should_adjust_base_factors"],
            "hold_reason": selection_hints["hold_reason"],
            "replace_base_factors": selection_hints["replace_base_factors"],
            "base_factor_components": base_factor_components,
            "recent_expression_components": recent_expression_components,
            "previous_base_factor_codes": self.resolve_base_factor_codes(previous_base_factors),
            "parent_expression": str(parent_expression or "").strip(),
            "parent_raw_expression": str(parent_raw_expression or "").strip(),
            "factor_usage": factor_usage,
        }

    def select_continue_factors(
        self,
        *,
        parent_result: dict[str, Any] | None,
        parent_request: dict[str, Any] | None,
        prompt: str,
        direction: str | None,
        factor_update_mode: str,
        max_factor_count: int,
        candidate_limit: int,
        current_base_factors: list[str] | None = None,
    ) -> dict[str, Any]:
        context = self.build_continuation_context(
            result=parent_result,
            request_payload=parent_request,
            prompt=prompt,
            factor_update_mode=factor_update_mode,
            additional_factor_count=max_factor_count,
        )
        base_factors = context["base_factors"]
        exclude_factors = []
        if factor_update_mode == "append":
            exclude_factors = _dedupe_preserve_order(
                list(base_factors) + list(current_base_factors or [])
            )
        selection = self.select_factors(
            prompt=f"{prompt} {direction or ''}".strip(),
            max_factor_count=max_factor_count,
            candidate_limit=candidate_limit,
            selection_mode="auto",
            direction=direction or parent_request.get("direction"),
            start_date=parent_request.get("start_date"),
            end_date=parent_request.get("end_date"),
            universe=parent_request.get("universe"),
            benchmark=parent_request.get("benchmark"),
            extra_context=context["summary_text"],
            exclude_factors=exclude_factors,
            continuation_context=context,
        )
        selection["continuation_context"] = context
        return selection

    def _build_continuation_selection_hints(
        self,
        *,
        primary_problem: str,
        secondary_problem: str,
        recommended_goal: str,
        suggested_actions: list[str],
        metric_snapshot: dict[str, Any],
        used_new_factors: list[str] | None = None,
        unused_new_factors: list[str] | None = None,
    ) -> dict[str, Any]:
        context_text = " ".join(
            [
                str(primary_problem or ""),
                str(secondary_problem or ""),
                *[str(item or "") for item in suggested_actions],
            ]
        ).lower()
        preferred_keywords: list[str] = []
        avoid_keywords: list[str] = []
        confidence = 0
        hold_reason = ""
        force_replace_new_factors = False
        instructions: list[str] = [
            f"本轮只围绕主要短板“{primary_problem}”做一次受控迭代，不要同时追求多个互相冲突的目标。",
            "优先补充与当前基础因子语义互补、且能直接响应主要短板的基础因子，不要重复已有风格。",
        ]
        if secondary_problem:
            instructions.append(f"次要问题“{secondary_problem}”只作为约束条件，避免新候选明显恶化这一项。")
        hold_signal_detected = _contains_any_keyword(
            context_text,
            [
                "当前结果已具备一定可用性",
                "当前结果已经较稳定",
                "已经较稳定",
                "暂无",
            ],
        )

        turnover = _safe_float(metric_snapshot.get("turnover"))
        rank_ic = abs(_safe_float(metric_snapshot.get("rank_ic")))
        sharpe = _safe_float(metric_snapshot.get("ls_sharpe") or metric_snapshot.get("report_sharpe"))
        max_drawdown = abs(_safe_float(metric_snapshot.get("report_max_drawdown")))
        score = _safe_float(metric_snapshot.get("score"))
        report_cagr = _safe_float(metric_snapshot.get("report_cagr"))
        ls_return = _safe_float(metric_snapshot.get("ls_return"))
        used_new_factors = _dedupe_preserve_order([str(item).strip() for item in (used_new_factors or []) if str(item or "").strip()])
        unused_new_factors = _dedupe_preserve_order([str(item).strip() for item in (unused_new_factors or []) if str(item or "").strip()])
        replace_base_factors: list[str] = []

        if unused_new_factors:
            confidence += 2
            replace_base_factors = list(unused_new_factors)
            force_replace_new_factors = True
            hold_signal_detected = False
            instructions.append(
                f"上一轮新增基础因子 {', '.join(unused_new_factors)} 尚未真正进入最佳表达式，"
                "本轮不要直接 hold，优先替换这些未生效的新因子，避免在 append 模式下无效堆积。"
            )

        if ("换手" in context_text or turnover >= 0.5) and not hold_signal_detected:
            confidence += 2
            preferred_keywords.extend(["ema", "sma", "ma", "mean", "std", "volatility", "atr", "volume"])
            avoid_keywords.extend(["distance_to_high", "distance_to_low", "breakout", "roc", "return", "momentum"])
            instructions.append("如果上一轮问题是换手率偏高，优先选择更平滑、中周期、抗噪声的量价或波动类基础因子，避免短周期突破型因子。")

        if ("区分度" in context_text or "rankic" in context_text or rank_ic < 0.02) and not hold_signal_detected:
            confidence += 2
            preferred_keywords.extend(["rank", "spread", "corr", "volume", "amount", "interaction", "residual", "dispersion"])
            avoid_keywords.extend(["distance_to_high", "distance_to_low", "vwma", "ratio"])
            instructions.append("如果上一轮问题是横截面区分度不足，优先补充更能提升排序能力、量价交互或横截面离散度的信息源。")
            instructions.append("避免继续追加与现有价格比例类因子高度相似的候选，除非它能引入新的成交量、波动或相关性信息。")
        if "rankic 为负" in context_text and not hold_signal_detected:
            confidence += 1
            preferred_keywords.extend(["residual", "interaction", "corr", "dispersion", "reversal"])
            avoid_keywords.extend(["ratio", "breakout"])
            instructions.append("如果上一轮 rankIC 已转负，优先纠正排序方向，补充更稳定的量价交互、离散度或残差信息，避免继续放大原有错误排序结构。")

        if ("sharpe" in context_text or "风险调整后收益" in context_text or sharpe < 1.0) and not hold_signal_detected:
            confidence += 2
            preferred_keywords.extend(["quality", "stability", "volatility", "std", "ema", "atr"])
            avoid_keywords.extend(["breakout", "roc"])
            instructions.append("如果上一轮问题是 Sharpe 偏低，优先考虑能提升稳定性和收益质量的平滑类或风险约束类因子。")
        if ("sharpe 已转负" in context_text or "收益质量明显恶化" in context_text or sharpe < 0) and not hold_signal_detected:
            confidence += 1
            preferred_keywords.extend(["quality", "defensive", "downside", "stability", "atr", "volatility"])
            avoid_keywords.extend(["momentum", "breakout", "roc"])
            instructions.append("如果上一轮 Sharpe 已转负，下一轮先以恢复正向收益和稳定性为目标，优先选择防御性、下行约束或波动控制更强的基础因子，避免继续追逐进攻型信号。")

        if ("回撤" in context_text or max_drawdown >= 0.2) and not hold_signal_detected:
            confidence += 2
            preferred_keywords.extend(["volatility", "atr", "std", "downside", "quality"])
            avoid_keywords.extend(["breakout", "momentum"])
            instructions.append("如果上一轮问题是回撤偏大，避免继续放大高波动、高追涨属性的候选。")

        if ("收益偏弱" in context_text or "return" in context_text) and not hold_signal_detected:
            confidence += 1
            preferred_keywords.extend(["trend", "momentum", "volume", "accumulation"])
        if ("年化收益为负" in context_text or ls_return < 0) and not hold_signal_detected:
            confidence += 1
            preferred_keywords.extend(["trend", "accumulation", "quality", "cashflow"])
            avoid_keywords.extend(["high_beta", "breakout"])
            instructions.append("如果上一轮收益端已经转负，优先恢复正向收益来源，避免再追加高 beta、追涨型或噪声放大型候选。")

        strong_hold_metrics = (
            (score >= 75 and sharpe >= 1.0 and rank_ic >= 0.025 and turnover <= 0.45)
            or (score >= 65 and report_cagr >= 0.2 and ls_return >= 0.2 and turnover <= 0.45)
            or (sharpe >= 1.5 and rank_ic >= 0.03 and turnover <= 0.4)
        )

        if not hold_signal_detected and strong_hold_metrics:
            hold_signal_detected = True

        if hold_signal_detected and not force_replace_new_factors:
            hold_reason = "上一轮没有暴露出足够明确的结构性短板，优先保持当前基础因子组合，先在表达式结构上做微调。"
            instructions.append(hold_reason)

        should_adjust_base_factors = confidence >= 2 and not hold_reason

        preferred_keywords = _dedupe_preserve_order(preferred_keywords)
        avoid_keywords = _dedupe_preserve_order(avoid_keywords)
        if preferred_keywords:
            instructions.append(f"优先考虑这些语义关键词附近的候选：{', '.join(preferred_keywords[:8])}。")
        if avoid_keywords:
            instructions.append(f"尽量回避这些容易偏题的关键词：{', '.join(avoid_keywords[:8])}。")

        return {
            "preferred_keywords": preferred_keywords,
            "avoid_keywords": avoid_keywords,
            "selection_instructions": "\n".join(instructions),
            "selection_confidence": confidence,
            "should_adjust_base_factors": should_adjust_base_factors,
            "hold_reason": hold_reason,
            "replace_base_factors": replace_base_factors,
        }

    def _collect_base_factor_component_tokens(self, base_factors: list[str]) -> dict[str, list[str]]:
        component_fields: list[str] = []
        component_tokens: list[str] = []

        db = get_db_session()
        try:
            repo = FactorRepository(db)
            for factor_name in base_factors:
                normalized_name = str(factor_name or "").strip().lower()
                if normalized_name:
                    component_tokens.extend(_tokenize_text(normalized_name))
                factor = repo.get_by_name(factor_name)
                if not factor:
                    continue
                code = str(factor.code or "").strip().lower()
                if code:
                    component_tokens.extend(_tokenize_text(code))
                task_metadata = factor.task_metadata if isinstance(factor.task_metadata, dict) else {}
                canonical_ast = task_metadata.get("canonical_ast") if isinstance(task_metadata.get("canonical_ast"), dict) else {}
                component_fields.extend(_component_tokens(canonical_ast.get("fields")))
                component_tokens.extend(_component_tokens(canonical_ast.get("operators")))
        finally:
            db.close()

        return {
            "fields": _dedupe_preserve_order(component_fields),
            "tokens": _dedupe_preserve_order(component_tokens),
        }

    def _candidate_component_overlap_penalty(
        self,
        *,
        candidate: dict[str, Any],
        continuation_context: dict[str, Any] | None,
    ) -> int:
        if not continuation_context:
            return 0

        base_components = continuation_context.get("base_factor_components") if isinstance(continuation_context, dict) else {}
        recent_components = continuation_context.get("recent_expression_components") if isinstance(continuation_context, dict) else {}
        existing_field_set = {
            str(item).strip().lower()
            for item in [
                *((base_components or {}).get("fields") or []),
                *((recent_components or {}).get("fields") or []),
            ]
            if str(item or "").strip()
        }
        existing_token_set = {
            str(item).strip().lower()
            for item in [
                *((base_components or {}).get("tokens") or []),
                *((recent_components or {}).get("operators") or []),
            ]
            if str(item or "").strip()
        }
        candidate_tokens = {
            token
            for token in _tokenize_text(
                " ".join(
                    [
                        str(candidate.get("name") or ""),
                        str(candidate.get("category") or ""),
                        str(candidate.get("description") or ""),
                        str(candidate.get("code") or ""),
                    ]
                )
            )
            if token
        }
        if not candidate_tokens:
            return 0

        overlap_fields = candidate_tokens & existing_field_set
        overlap_tokens = candidate_tokens & existing_token_set
        penalty = 0
        if len(overlap_fields) >= 2:
            penalty += 4
        elif overlap_fields:
            penalty += 2
        if len(overlap_tokens) >= max(len(candidate_tokens) // 2, 2):
            penalty += 3
        elif len(overlap_tokens) >= 2:
            penalty += 2
        return penalty

    def _score_candidate_for_continuation(
        self,
        *,
        candidate: dict[str, Any],
        preferred_keywords: list[str],
        avoid_keywords: list[str],
        continuation_context: dict[str, Any] | None,
    ) -> int:
        searchable_text = " ".join(
            [
                str(candidate.get("name") or ""),
                str(candidate.get("category") or ""),
                str(candidate.get("description") or ""),
                str(candidate.get("code") or ""),
            ]
        ).lower()
        score = 0
        for keyword in preferred_keywords:
            if keyword and keyword in searchable_text:
                score += 3
        for keyword in avoid_keywords:
            if keyword and keyword in searchable_text:
                score -= 4
        snapshot_summary = candidate.get("snapshot_summary") or {}
        report_metrics = snapshot_summary.get("report_metrics") or {}
        backtest_summary = snapshot_summary.get("backtest_summary") or {}
        if _safe_float(report_metrics.get("sharpe")) >= 1.0:
            score += 1
        if abs(_safe_float(backtest_summary.get("rank_ic_mean"))) >= 0.02:
            score += 1
        score -= self._candidate_component_overlap_penalty(
            candidate=candidate,
            continuation_context=continuation_context,
        )
        return score

    def _prioritize_continuation_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        continuation_context: dict[str, Any] | None,
        max_factor_count: int,
    ) -> list[dict[str, Any]]:
        if not continuation_context:
            return candidates

        preferred_keywords = list(continuation_context.get("preferred_keywords") or [])
        avoid_keywords = list(continuation_context.get("avoid_keywords") or [])
        if not preferred_keywords and not avoid_keywords:
            return candidates

        scored_candidates: list[tuple[int, int, dict[str, Any]]] = []
        for index, candidate in enumerate(candidates):
            score = self._score_candidate_for_continuation(
                candidate=candidate,
                preferred_keywords=preferred_keywords,
                avoid_keywords=avoid_keywords,
                continuation_context=continuation_context,
            )
            scored_candidates.append((score, -index, candidate))

        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        positive_count = sum(1 for score, _, _ in scored_candidates if score > 0)
        if positive_count <= 0:
            return [candidate for _, _, candidate in scored_candidates]

        prioritized_limit = max(max(int(max_factor_count or 0), 1) * 10, 24)
        prioritized: list[dict[str, Any]] = [candidate for _, _, candidate in scored_candidates[:prioritized_limit]]
        remaining: list[dict[str, Any]] = [candidate for _, _, candidate in scored_candidates[prioritized_limit:]]
        return prioritized + remaining

    def generate_candidate_expressions(
        self,
        *,
        prompt: str,
        base_factor_codes: list[str],
        n_candidates: int,
        previous_expressions: list[str] | None = None,
        continuation_context: dict[str, Any] | None = None,
    ) -> list[str]:
        expressions: list[str] = []

        llm_config = llm_config_service.get_runtime_config()
        if llm_config.get("api_key"):
            expressions = self._generate_candidates_with_llm(
                prompt=prompt,
                base_factor_codes=base_factor_codes,
                n_candidates=n_candidates,
                llm_config=llm_config,
                previous_expressions=previous_expressions or [],
                continuation_context=continuation_context,
            )

        if expressions:
            return expressions[:n_candidates]

        fallback_expressions = self._build_fallback_candidate_expressions(
            base_factor_codes=base_factor_codes,
            n_candidates=n_candidates,
            previous_expressions=previous_expressions or [],
            continuation_context=continuation_context,
        )
        return fallback_expressions[:n_candidates]

    def _build_fallback_candidate_expressions(
        self,
        *,
        base_factor_codes: list[str],
        n_candidates: int,
        previous_expressions: list[str],
        continuation_context: dict[str, Any] | None,
    ) -> list[str]:
        normalized_existing = {_normalize_expression(item) for item in previous_expressions if _normalize_expression(item)}
        deduped_base = _reorder_base_factor_codes_for_continuation(base_factor_codes, continuation_context)
        if not deduped_base:
            deduped_base = ["close", "volume"]

        primary = deduped_base[0]
        secondary = deduped_base[min(1, len(deduped_base) - 1)]
        tertiary = deduped_base[min(2, len(deduped_base) - 1)]
        metric_snapshot = continuation_context.get("metric_snapshot") if continuation_context else {}
        turnover = _safe_float((metric_snapshot or {}).get("turnover"))
        rank_ic = abs(_safe_float((metric_snapshot or {}).get("rank_ic")))
        primary_problem = str((continuation_context or {}).get("primary_problem") or "")
        secondary_problem = str((continuation_context or {}).get("secondary_problem") or "")
        action_text = " ".join(str(item or "") for item in ((continuation_context or {}).get("suggested_actions") or []))
        context_text = f"{primary_problem} {secondary_problem} {action_text}".lower()
        parent_expression = str((continuation_context or {}).get("parent_expression") or "").strip()
        new_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
        prioritize_new_factor_exploration = bool(
            continuation_context
            and continuation_context.get("should_adjust_base_factors")
            and new_codes
        )

        templates: list[str] = []
        hold_mode = bool((continuation_context or {}).get("hold_reason"))
        if parent_expression and prioritize_new_factor_exploration:
            for anchor_code in new_codes[:2]:
                templates.extend(
                    [
                        f"rank(ts_mean({parent_expression}, 5) * ts_mean({anchor_code}, 5))",
                        f"rank(ts_mean({parent_expression}, 5) / (1 + ts_std({anchor_code}, 20)))",
                        f"rank(({parent_expression} - ts_mean({parent_expression}, 10)) * rank({anchor_code}))",
                        f"rank(ts_mean({parent_expression} * {anchor_code}, 5))",
                        f"rank(ts_corr({parent_expression}, {anchor_code}, 10))",
                        f"rank(ts_mean({parent_expression}, 5) * ts_rank({anchor_code}, 10))",
                    ]
                )
        elif parent_expression:
            templates.extend(
                [
                    f"rank(ts_mean({parent_expression}, 3))",
                    f"rank(ts_mean({parent_expression}, 5))",
                    f"rank(ts_zscore({parent_expression}, 20))",
                ]
            )
            if deduped_base:
                templates.append(
                    f"rank(ts_mean({parent_expression}, 5) / (1 + ts_std({secondary}, 20)))"
                )
                templates.append(
                    f"rank(ts_mean({parent_expression}, 5) * ts_mean({primary}, 5))"
                )
            if new_codes:
                anchor_code = new_codes[0]
                templates.extend(
                    [
                        f"rank(ts_mean({parent_expression}, 5) * ts_mean({anchor_code}, 5))",
                        f"rank(ts_mean({parent_expression}, 5) / (1 + ts_std({anchor_code}, 20)))",
                    ]
                )
        if hold_mode:
            templates.extend(
                [
                    f"rank(ts_mean({primary}, 5))",
                    f"rank(ts_zscore({primary}, 20))",
                    f"rank(ts_mean({primary}, 10) / (ts_std({primary}, 20) + 1e-6))",
                ]
            )

        if "换手" in context_text or turnover >= 0.5:
            templates.extend(
                [
                    f"rank(ts_mean({primary}, 10) / (ts_std({primary}, 20) + 1e-6))",
                    f"rank(ts_mean({secondary}, 10) / (ts_std({secondary}, 20) + 1e-6))",
                    f"rank(ts_mean({primary}, 20) - ts_mean({primary}, 5))",
                ]
            )

        if ("sharpe" in context_text or "风险调整后收益" in context_text) and ("区分度" in context_text or "rankic" in context_text):
            templates.extend(
                [
                    f"rank(ts_mean(ts_corr({primary}, {secondary}, 10), 5))",
                    f"rank(ts_mean(({primary} - ts_mean({primary}, 10)) / (ts_std({primary}, 20) + 1e-6), 5))",
                    f"rank(ts_mean({secondary}, 5) / (1 + ts_std({secondary}, 20)))",
                ]
            )

        if len(deduped_base) >= 3:
            templates.extend(
                [
                    f"rank(ts_mean({primary}, 5) / (1 + ts_std({secondary}, 20)))",
                    f"rank(ts_mean(ts_corr({primary}, {secondary}, 10), 5) * ts_mean({tertiary}, 5))",
                    f"rank(ts_mean({primary}, 5) * ({secondary} / (1 + ts_std({secondary}, 20))))",
                ]
            )
            if prioritize_new_factor_exploration:
                templates.extend(
                    [
                        f"rank(ts_mean({primary} * {tertiary}, 5))",
                        f"rank(ts_mean(({primary} - ts_mean({primary}, 10)) * {tertiary}, 5))",
                        f"rank(ts_corr({primary}, {tertiary}, 10) * ts_mean({secondary}, 5))",
                    ]
                )

        if "区分度" in context_text or "rankic" in context_text or rank_ic < 0.02:
            templates.extend(
                [
                    f"rank(ts_mean(ts_delta({primary}, 5), 5))",
                    f"rank(ts_corr({primary}, {secondary}, 10))",
                    f"rank(ts_zscore({secondary}, 20))",
                    f"rank(({primary} - ts_mean({primary}, 10)) / (ts_std({primary}, 20) + 1e-6))",
                    f"rank(ts_mean({secondary}, 5) / (ts_std({secondary}, 20) + 1e-6))",
                ]
            )

        templates.extend(
            [
                f"rank(ts_mean(ts_delta({primary}, 5), 3))",
                f"rank(ts_mean({secondary}, 10) / (ts_std({secondary}, 10) + 1e-6))",
                f"rank(ts_zscore({primary}, 20))",
                f"rank(ts_mean({primary}, 5) / (ts_mean({tertiary}, 10) + 1e-6))",
            ]
        )

        results: list[str] = []
        seen = set(normalized_existing)
        for expression in templates:
            key = _normalize_expression(expression)
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(expression)
            if len(results) >= max(int(n_candidates or 0), 1) * 4:
                break
        logger.info("LLM 未返回可执行候选表达式，使用 fallback 模板生成 %s 条本地候选。", len(results))
        return results

    def _score_evaluation_for_continuation(
        self,
        *,
        evaluation: FactorEvaluationResult,
        continuation_context: dict[str, Any] | None,
        base_factor_codes: list[str],
    ) -> tuple[float, float, int]:
        base_score = float(getattr(evaluation, "score", 0.0) or 0.0)
        if not continuation_context:
            return (base_score, 0.0, 0)

        expression = str(getattr(evaluation, "expression", "") or "")
        backtest_summary = getattr(evaluation, "backtest_summary", {}) or {}
        turnover = _safe_float(backtest_summary.get("turnover"))
        rank_ic = _safe_float(backtest_summary.get("rank_ic_mean") or backtest_summary.get("ic_mean"))
        continuation_text = " ".join(
            [
                str(continuation_context.get("primary_problem") or ""),
                str(continuation_context.get("secondary_problem") or ""),
                " ".join(str(item or "") for item in (continuation_context.get("suggested_actions") or [])),
            ]
        ).lower()
        parent_expression = str(continuation_context.get("parent_expression") or "").strip()

        new_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
        new_factor_hits = sum(1 for code in new_codes if _expression_uses_code(expression, code))
        parent_anchor_hit = 1 if parent_expression and _expression_contains_parent_anchor(expression, parent_expression) else 0
        pure_parent_smoothing = bool(
            parent_expression
            and parent_anchor_hit
            and new_factor_hits == 0
            and _is_parent_smoothing_expression(expression, parent_expression)
        )

        adjusted_score = base_score
        if new_factor_hits:
            adjusted_score += min(new_factor_hits, 2) * 2.5
        elif new_codes:
            adjusted_score -= 2.2
        if parent_anchor_hit:
            adjusted_score += 1.5
            if new_factor_hits:
                adjusted_score += 1.4
        elif parent_expression and new_codes and new_factor_hits:
            adjusted_score -= 0.8
        if pure_parent_smoothing and continuation_context.get("should_adjust_base_factors"):
            adjusted_score -= 2.5

        if ("换手" in continuation_text or "sharpe" in continuation_text or "风险调整后收益" in continuation_text) and turnover >= 0.65:
            adjusted_score -= min((turnover - 0.65) * 10, 4.0)
        if "区分度" in continuation_text or "rankic" in continuation_text:
            adjusted_score += max(rank_ic, -0.02) * 40

        return (adjusted_score, base_score, new_factor_hits + parent_anchor_hit)

    def _prioritize_supported_expressions_for_continuation(
        self,
        *,
        expressions: list[str],
        continuation_context: dict[str, Any] | None,
        base_factor_codes: list[str],
    ) -> list[str]:
        if not continuation_context or not expressions:
            return expressions

        new_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
        if not new_codes:
            return expressions

        parent_expression = str(continuation_context.get("parent_expression") or "").strip()
        scored: list[tuple[int, int, str]] = []
        for index, expression in enumerate(expressions):
            new_hits = sum(1 for code in new_codes if _expression_uses_code(expression, code))
            score = new_hits * 10
            if new_hits >= 1 and any(token in expression for token in ("*", "/", "ts_corr(", "ts_rank(", "rank(")):
                score += 4
            if parent_expression and _expression_contains_parent_anchor(expression, parent_expression):
                score += 6
                if new_hits:
                    score += 3
                if new_hits == 0 and _is_parent_smoothing_expression(expression, parent_expression):
                    score -= 5
            if new_hits == 0:
                score -= 2
            scored.append((score, -index, expression))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [expression for _, _, expression in scored]

    def _select_best_evaluation_for_continuation(
        self,
        *,
        evaluations: list[FactorEvaluationResult],
        continuation_context: dict[str, Any] | None,
        base_factor_codes: list[str],
    ) -> list[FactorEvaluationResult]:
        if not continuation_context or len(evaluations) <= 1:
            return evaluations

        should_adjust_base_factors = bool(continuation_context.get("should_adjust_base_factors"))
        new_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
        if not should_adjust_base_factors or not new_codes:
            return evaluations

        top_evaluation = evaluations[0]
        top_uses_new_codes = self._evaluation_uses_any_code(top_evaluation, new_codes)
        if top_uses_new_codes:
            return evaluations

        target_metric_key = _campaign_metric_key(continuation_context.get("recommended_goal"))
        target_metric_value_getters: dict[str, Callable[[Any], float]] = {
            "score": lambda item: _safe_float(getattr(item, "score", 0.0)),
            "ls_sharpe": lambda item: _safe_float((getattr(item, "backtest_summary", {}) or {}).get("long_short_sharpe")),
            "ls_return": lambda item: _safe_float((getattr(item, "backtest_summary", {}) or {}).get("long_short_annual")),
            "report_sharpe": lambda item: _safe_float((getattr(item, "report_metrics", {}) or {}).get("sharpe")),
            "wq_fitness": lambda item: _safe_float((getattr(item, "wq_brain", {}) or {}).get("wq_fitness")),
            "wq_return": lambda item: _safe_float((getattr(item, "wq_brain", {}) or {}).get("wq_returns")),
        }
        target_metric_getter = target_metric_value_getters.get(target_metric_key, target_metric_value_getters["score"])
        top_target_metric = target_metric_getter(top_evaluation)

        top_adjusted_score = self._score_evaluation_for_continuation(
            evaluation=top_evaluation,
            continuation_context=continuation_context,
            base_factor_codes=base_factor_codes,
        )[0]
        replacement_index: int | None = None
        for index, evaluation in enumerate(evaluations[1:], start=1):
            if not self._evaluation_uses_any_code(evaluation, new_codes):
                continue
            candidate_adjusted_score = self._score_evaluation_for_continuation(
                evaluation=evaluation,
                continuation_context=continuation_context,
                base_factor_codes=base_factor_codes,
            )[0]
            score_gap = top_adjusted_score - candidate_adjusted_score
            target_metric_gap = top_target_metric - target_metric_getter(evaluation)
            if target_metric_key != "score" and target_metric_gap <= 0.0:
                replacement_index = index
                break
            if score_gap <= 10.0:
                replacement_index = index
                break

        if replacement_index is None:
            return evaluations

        reordered = list(evaluations)
        reordered[0], reordered[replacement_index] = reordered[replacement_index], reordered[0]
        return reordered

    def _evaluation_uses_any_code(
        self,
        evaluation: FactorEvaluationResult | Any,
        codes: list[str],
    ) -> bool:
        expression = str(getattr(evaluation, "expression", "") or "")
        return any(_expression_uses_code(expression, code) for code in codes)

    def _build_required_continuation_expressions(
        self,
        *,
        base_factor_codes: list[str],
        continuation_context: dict[str, Any] | None,
        limit: int,
    ) -> list[str]:
        if not continuation_context or limit <= 0:
            return []
        new_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
        if not new_codes:
            return []

        parent_expression = str(continuation_context.get("parent_expression") or "").strip()
        deduped_base = _reorder_base_factor_codes_for_continuation(base_factor_codes, continuation_context)
        primary = deduped_base[0] if deduped_base else "close"
        templates: list[str] = []
        for code in new_codes[: max(limit, 1)]:
            if parent_expression:
                templates.extend(
                    [
                        f"rank(ts_mean({parent_expression}, 5) * ts_mean({code}, 5))",
                        f"rank(ts_mean({parent_expression}, 5) / (1 + ts_std({code}, 20)))",
                        f"rank(({parent_expression} - ts_mean({parent_expression}, 10)) * rank({code}))",
                        f"rank(ts_mean({parent_expression} * {code}, 5))",
                        f"rank(ts_corr({parent_expression}, {code}, 10))",
                        f"rank(ts_mean({parent_expression}, 5) * ts_rank({code}, 10))",
                    ]
                )
            templates.extend(
                [
                    f"rank(ts_mean({primary} * {code}, 5))",
                    f"rank(ts_corr({primary}, {code}, 10))",
                    f"rank(({primary} - ts_mean({primary}, 10)) * rank({code}))",
                ]
            )

        results: list[str] = []
        seen: set[str] = set()
        for expression in templates:
            key = _normalize_expression(expression)
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(expression)
            if len(results) >= limit:
                break
        return results

    def _pick_continuation_seed_candidate(
        self,
        *,
        round_result: dict[str, Any] | None,
        continuation_context: dict[str, Any] | None,
        base_factor_codes: list[str],
    ) -> dict[str, Any] | None:
        if not round_result or not continuation_context:
            return None
        new_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
        if not new_codes:
            return None

        best_candidate: dict[str, Any] | None = None
        best_rank: tuple[float, float, int] | None = None
        for factor in round_result.get("factors", []) or []:
            expression = str(factor.get("expression") or "")
            if not expression:
                continue
            if not any(_expression_uses_code(expression, code) for code in new_codes):
                continue
            candidate = SimpleNamespace(
                expression=expression,
                score=factor.get("score", 0.0),
                backtest_summary=factor.get("backtest_summary", {}) or {},
            )
            candidate_rank = self._score_evaluation_for_continuation(
                evaluation=candidate,
                continuation_context=continuation_context,
                base_factor_codes=base_factor_codes,
            )
            if best_rank is None or candidate_rank > best_rank:
                best_rank = candidate_rank
                best_candidate = factor
        return best_candidate


    def _collect_sample_frames(
        self,
        *,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        max_samples: int = 3,
    ) -> list[pd.DataFrame]:
        samples: list[pd.DataFrame] = []
        for stock_code in stock_codes:
            try:
                stock_df = self.data_service.get_stock_data(stock_code, start_date, end_date)
            except Exception as exc:
                logger.warning("加载样本股票 %s 失败，跳过表达式预检：%s", stock_code, exc)
                continue
            if stock_df is None or stock_df.empty:
                continue
            samples.append(stock_df.copy())
            if len(samples) >= max_samples:
                break
        return samples

    def _filter_supported_expressions(
        self,
        expressions: list[str],
        *,
        sample_frames: list[pd.DataFrame],
        limit: int,
    ) -> list[str]:
        if not expressions or not sample_frames:
            return expressions[:limit]

        supported: list[str] = []
        seen: set[str] = set()
        for expression in expressions:
            key = _normalize_expression(expression)
            if not key or key in seen:
                continue
            seen.add(key)

            executable = False
            try:
                executable = self._quantgpt_engine.can_execute_on_frames(expression, sample_frames)
            except Exception as exc:
                logger.info("QuantGPT 预检失败，跳过候选表达式 %s：%s", expression, exc)

            if executable:
                supported.append(expression)
                if len(supported) >= limit:
                    break
        return supported

    def _generate_candidates_with_llm(
        self,
        *,
        prompt: str,
        base_factor_codes: list[str],
        n_candidates: int,
        llm_config: dict[str, Any],
        previous_expressions: list[str],
        continuation_context: dict[str, Any] | None,
    ) -> list[str]:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK 未安装，跳过 LLM 候选生成")
            return []

        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config.get("base_url") or "https://api.deepseek.com/v1",
        )

        usable_base = _reorder_base_factor_codes_for_continuation(base_factor_codes, continuation_context)[:20]
        continuation_lines: list[str] = []
        if continuation_context:
            new_base_codes = _get_new_base_factor_codes(base_factor_codes, continuation_context)
            continuation_lines.extend(
                [
                    f"上一轮主要问题：{continuation_context.get('primary_problem') or '暂无'}",
                    f"上一轮次要约束：{continuation_context.get('secondary_problem') or '暂无'}",
                    f"上一轮最佳表达式：{continuation_context.get('parent_expression') or '暂无'}",
                    f"上一轮基础因子：{json.dumps(continuation_context.get('previous_base_factors') or [], ensure_ascii=False)}",
                    f"本轮新增基础因子表达式：{json.dumps(new_base_codes, ensure_ascii=False)}",
                    f"建议优化方向：{continuation_context.get('recommended_goal') or '提升综合分数'}",
                    f"建议动作：{json.dumps(continuation_context.get('suggested_actions') or [], ensure_ascii=False)}",
                    f"关键指标：{json.dumps(continuation_context.get('metric_snapshot') or {}, ensure_ascii=False)}",
                ]
            )

        user_prompt = (
            "你是量化因子表达式生成器。\n"
            f"研究目标：{prompt}\n"
            f"基础因子：{json.dumps(usable_base, ensure_ascii=False)}\n"
            f"请生成 {n_candidates} 个不同的候选表达式，优先输出可解释、稳定的量价复合因子。\n"
        )
        if previous_expressions:
            user_prompt += f"禁止重复以下表达式：{json.dumps(previous_expressions[-12:], ensure_ascii=False)}\n"
        if continuation_lines:
            user_prompt += "\n".join(continuation_lines) + "\n"
        if continuation_context and continuation_context.get("secondary_problem"):
            user_prompt += "生成原则：主目标由上一轮主要问题决定；上一轮次要问题只作为约束，避免新表达式明显恶化换手、Sharpe、回撤或稳定性。\n"
        if continuation_context and continuation_context.get("parent_expression"):
            user_prompt += "生成原则：优先基于上一轮最佳表达式做局部改写或受控扩展，保留其有效结构，再引入新增基础因子或风险约束；不要把表达式整体推倒重来。\n"
        if continuation_context and continuation_context.get("previous_base_factors"):
            user_prompt += "生成原则：如果本轮新增了基础因子，至少输出一半候选表达式显式使用这些新增因子，而不是继续只围绕上一轮旧因子做改写。\n"
        user_prompt += "只返回 JSON 数组，不要附加解释。"

        response = client.chat.completions.create(
            model=llm_config.get("model") or "deepseek-chat",
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=1200,
        )
        content = (response.choices[0].message.content or "").strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(content)
        except Exception:
            logger.warning("LLM 候选表达式解析失败：%s", content[:200])
            return []

        expressions: list[str] = []
        seen = {_normalize_expression(item) for item in previous_expressions}
        for item in parsed:
            expression = str(item).strip()
            if not expression:
                continue
            key = _normalize_expression(expression)
            if key in seen:
                continue
            seen.add(key)
            expressions.append(expression)
        return expressions

    def _run_async_tool(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    def _build_research_tool_request(
        self,
        *,
        expression: str,
        start_date: str,
        end_date: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        neutralize_industry: bool,
        neutralize_cap: bool,
        universe: str = "hs300",
    ) -> ResearchToolBaseRequest:
        return ResearchToolBaseRequest(
            expression=expression,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            n_groups=n_groups,
            holding_period=holding_period,
            benchmark=benchmark,
            neutralize_industry=neutralize_industry,
            neutralize_cap=neutralize_cap,
        )

    def _validate_candidate_expression(
        self,
        *,
        expression: str,
        start_date: str,
        end_date: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        neutralize_industry: bool,
        neutralize_cap: bool,
        universe: str = "hs300",
    ) -> dict[str, Any] | None:
        try:
            response = self._run_async_tool(
                validation_service.validate_expression(expression, "local")
            )
        except Exception as exc:
            logger.warning("QuantGPT validation 调用失败，将继续本地评估 %s：%s", expression, exc)
            return {
                "success": False,
                "valid": True,
                "message": f"validation 调用失败，降级继续本地评估：{exc}",
                "raw": {
                    "input_expression": expression,
                    "degraded_to_local_execution": True,
                },
            }

        return {
            "success": bool(getattr(response, "success", False)),
            "valid": bool(getattr(response, "valid", False)),
            "message": getattr(response, "message", ""),
            "raw": getattr(response, "raw", None) or {},
        }

    def _diagnose_candidate_failure(
        self,
        *,
        expression: str,
        start_date: str,
        end_date: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        neutralize_industry: bool,
        neutralize_cap: bool,
        universe: str = "hs300",
    ) -> dict[str, Any] | None:
        request = self._build_research_tool_request(
            expression=expression,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            n_groups=n_groups,
            holding_period=holding_period,
            neutralize_industry=neutralize_industry,
            neutralize_cap=neutralize_cap,
        )
        try:
            response = self._run_async_tool(diagnosis_service.diagnose_factor(request))
        except Exception as exc:
            logger.warning("QuantGPT diagnosis 调用失败 %s：%s", expression, exc)
            return {
                "success": False,
                "error": str(exc),
                "report": None,
                "key_findings": [],
                "improvement_suggestions": [],
                "raw": {"input_expression": expression},
            }

        return {
            "success": bool(getattr(response, "success", False)),
            "error": getattr(response, "error", None),
            "report": getattr(response, "report", None),
            "key_findings": getattr(response, "key_findings", None) or [],
            "improvement_suggestions": getattr(response, "improvement_suggestions", None) or [],
            "raw": getattr(response, "raw", None) or {},
        }

    def run_auto_mining(
        self,
        *,
        prompt: str,
        base_factors: list[str],
        start_date: str,
        end_date: str,
        universe: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        n_candidates: int,
        direction: str | None = None,
        neutralize_industry: bool = True,
        neutralize_cap: bool = True,
        previous_expressions: list[str] | None = None,
        continuation_context: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        base_factor_codes = self.resolve_base_factor_codes(base_factors)
        stock_codes = self.data_service.get_stock_universe(universe, date=start_date)[:30]
        if not stock_codes:
            raise ValueError(f"股票池 {universe} 未返回可用股票")

        sample_frames = self._collect_sample_frames(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
        )
        requested_candidates = max(n_candidates, 1)
        required_new_factor_exploration = bool(
            continuation_context
            and continuation_context.get("should_adjust_base_factors")
            and _get_new_base_factor_codes(base_factor_codes, continuation_context)
        )
        required_parent_expression_continuation = bool(
            continuation_context
            and str(continuation_context.get("parent_expression") or "").strip()
        )
        minimum_exploration_candidates = min(
            max(int(requested_candidates // 2), 1),
            requested_candidates,
        ) if (required_new_factor_exploration or required_parent_expression_continuation) else 0
        attempted_expressions: set[str] = {
            _normalize_expression(item)
            for item in (previous_expressions or [])
            if _normalize_expression(item)
        }
        pending_expressions: list[str] = []

        def _extend_candidate_pool(target_count: int) -> None:
            if target_count <= 0:
                return

            if required_new_factor_exploration or required_parent_expression_continuation:
                deterministic_required = self._build_required_continuation_expressions(
                    base_factor_codes=base_factor_codes,
                    continuation_context=continuation_context,
                    limit=max(minimum_exploration_candidates * 3, requested_candidates * 2),
                )
                deterministic_supported = self._filter_supported_expressions(
                    deterministic_required,
                    sample_frames=sample_frames,
                    limit=len(deterministic_required) or target_count,
                )
                deterministic_supported = self._prioritize_supported_expressions_for_continuation(
                    expressions=deterministic_supported,
                    continuation_context=continuation_context,
                    base_factor_codes=base_factor_codes,
                )
                prioritized_pool = list(deterministic_supported) + list(pending_expressions)
                pending_expressions.clear()
                for expression in prioritized_pool:
                    key = _normalize_expression(expression)
                    if not key or key in attempted_expressions:
                        continue
                    if any(_normalize_expression(item) == key for item in pending_expressions):
                        continue
                    pending_expressions.append(expression)

            seed_expressions = [
                expression
                for expression in [
                    *(previous_expressions or []),
                    *pending_expressions,
                    *list(attempted_expressions),
                ]
                if expression
            ]
            generated = self.generate_candidate_expressions(
                prompt=prompt,
                base_factor_codes=base_factor_codes,
                n_candidates=target_count,
                previous_expressions=seed_expressions,
                continuation_context=continuation_context,
            )
            supported = self._filter_supported_expressions(
                generated,
                sample_frames=sample_frames,
                limit=target_count,
            )
            supported = self._prioritize_supported_expressions_for_continuation(
                expressions=supported,
                continuation_context=continuation_context,
                base_factor_codes=base_factor_codes,
            )
            if not supported:
                fallback_generated = self._build_fallback_candidate_expressions(
                    base_factor_codes=base_factor_codes,
                    n_candidates=target_count,
                    previous_expressions=seed_expressions,
                    continuation_context=continuation_context,
                )
                supported = self._filter_supported_expressions(
                    fallback_generated,
                    sample_frames=sample_frames,
                    limit=target_count,
                )
                supported = self._prioritize_supported_expressions_for_continuation(
                    expressions=supported,
                    continuation_context=continuation_context,
                    base_factor_codes=base_factor_codes,
                )

            for expression in supported:
                key = _normalize_expression(expression)
                if not key or key in attempted_expressions:
                    continue
                if any(_normalize_expression(item) == key for item in pending_expressions):
                    continue
                pending_expressions.append(expression)

        initial_target_count = requested_candidates * 2 if required_new_factor_exploration else requested_candidates
        _extend_candidate_pool(initial_target_count)

        if not pending_expressions:
            raise ValueError("未生成可执行候选表达式")

        evaluations: list[FactorEvaluationResult] = []
        exploration_evaluation_count = 0
        attempt_count = 0
        max_attempts = max(len(pending_expressions), requested_candidates * 6 if required_new_factor_exploration else requested_candidates * 4)
        while pending_expressions and attempt_count < max_attempts:
            if len(evaluations) >= requested_candidates and exploration_evaluation_count >= minimum_exploration_candidates:
                break
            expression = pending_expressions.pop(0)
            key = _normalize_expression(expression)
            if not key or key in attempted_expressions:
                continue
            attempted_expressions.add(key)
            attempt_count += 1
            evaluation = self.evaluate_expression(
                expression=expression,
                prompt=prompt,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                benchmark=benchmark,
                n_groups=n_groups,
                holding_period=holding_period,
                direction=direction or "score",
                neutralize_industry=neutralize_industry,
                neutralize_cap=neutralize_cap,
            )
            if evaluation is None:
                remaining_needed = max(requested_candidates - len(evaluations), minimum_exploration_candidates - exploration_evaluation_count)
                if not pending_expressions and attempt_count < max_attempts:
                    _extend_candidate_pool(min(remaining_needed, max_attempts - attempt_count))
                continue
            evaluations.append(evaluation)
            if required_new_factor_exploration:
                if self._evaluation_uses_any_code(
                    evaluation,
                    _get_new_base_factor_codes(base_factor_codes, continuation_context),
                ):
                    exploration_evaluation_count += 1
            elif required_parent_expression_continuation:
                parent_expression = str((continuation_context or {}).get("parent_expression") or "").strip()
                if parent_expression and _expression_contains_parent_anchor(getattr(evaluation, "expression", ""), parent_expression):
                    exploration_evaluation_count += 1
            if progress_callback is not None:
                progress_callback(
                    len(evaluations),
                    requested_candidates,
                    self._format_candidate_payload(
                        evaluation=evaluation,
                        prompt=prompt,
                        index=len(evaluations) - 1,
                        base_factors=base_factors,
                        round_evaluation=None,
                    ),
                )
            if not pending_expressions and attempt_count < max_attempts:
                remaining_needed = max(requested_candidates - len(evaluations), minimum_exploration_candidates - exploration_evaluation_count)
                if remaining_needed > 0:
                    _extend_candidate_pool(min(remaining_needed * 2, max_attempts - attempt_count))

        if not evaluations:
            raise ValueError("候选表达式评估失败，未产出有效结果")

        evaluations.sort(
            key=lambda item: self._score_evaluation_for_continuation(
                evaluation=item,
                continuation_context=continuation_context,
                base_factor_codes=base_factor_codes,
            ),
            reverse=True,
        )
        evaluations = self._select_best_evaluation_for_continuation(
            evaluations=evaluations,
            continuation_context=continuation_context,
            base_factor_codes=base_factor_codes,
        )
        best_scores = [item.score for item in evaluations]
        fitness_history = {
            "best": [round(max(best_scores[: index + 1]), 4) for index in range(len(best_scores))],
            "average": [round(float(np.mean(best_scores[: index + 1])), 4) for index in range(len(best_scores))],
        }

        round_evaluation = self._build_round_evaluation(
            prompt=prompt,
            base_factors=base_factors,
            best_evaluation=evaluations[0],
            direction=direction or "score",
        )

        factors = [
            self._format_candidate_payload(
                evaluation=item,
                prompt=prompt,
                index=index,
                base_factors=base_factors,
                round_evaluation=round_evaluation if index == 0 else None,
            )
            for index, item in enumerate(evaluations)
        ]

        best_factor = factors[0]
        return {
            "factors": factors,
            "candidates": factors,
            "best_score": best_factor["score"],
            "avg_score": round(float(np.mean(best_scores)), 4),
            "generations": len(factors),
            "fitness_history": fitness_history,
            "round_evaluation": round_evaluation,
            "upstream": {
                "mode": "quantgpt_like_single_round",
            },
        }

    def run_auto_campaign(
        self,
        *,
        prompt: str,
        base_factors: list[str],
        start_date: str,
        end_date: str,
        universe: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        exploration_rounds: int,
        n_candidates_per_round: int,
        additional_factor_count_per_round: int,
        factor_update_mode: str,
        parent_selection_strategy: str,
        direction: str | None = None,
        neutralize_industry: bool = True,
        neutralize_cap: bool = True,
        retention_filter: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        retention_filter = retention_filter or {}
        campaign_run_id = uuid.uuid4().hex[:8]
        campaign_metric_key = _campaign_metric_key(direction)
        rounds: list[dict[str, Any]] = []
        retained_factors: list[dict[str, Any]] = []
        aggregate_best: list[float] = []
        aggregate_avg: list[float] = []
        previous_expressions: list[str] = []
        current_base_factors = list(base_factors)
        current_prompt = prompt
        global_best_result: dict[str, Any] | None = None
        global_best_task_id: str | None = None
        global_best_summary: dict[str, Any] | None = None
        global_best_retained_factors: list[dict[str, Any]] = []
        best_round_result: dict[str, Any] | None = None
        best_round_task_id: str | None = None
        best_round_request: dict[str, Any] | None = None
        best_round_summary: dict[str, Any] | None = None
        best_round_retained_factors: list[dict[str, Any]] = []
        last_round_result: dict[str, Any] | None = None
        last_round_request: dict[str, Any] | None = None
        last_round_summary: dict[str, Any] | None = None
        continuation_parent_result: dict[str, Any] | None = None
        continuation_parent_request: dict[str, Any] | None = None
        continuation_parent_summary: dict[str, Any] | None = None
        last_selection_context: dict[str, Any] | None = None
        campaign_failure_reason: str | None = None

        for round_index in range(1, max(exploration_rounds, 1) + 1):
            continuation_context = None
            continuation_hypothesis = None
            previous_round = rounds[-1] if rounds else None
            analysis_round = previous_round
            analysis_request = last_round_request
            analysis_result = last_round_result
            parent_round = continuation_parent_summary if continuation_parent_summary is not None else previous_round
            parent_request = continuation_parent_request or last_round_request
            parent_result = continuation_parent_result or last_round_result
            previous_base_factors = list(current_base_factors)
            if analysis_round and analysis_result is not None:
                parent_input_base_factors = list(analysis_round.get("input_base_factors", []) or [])
                applied_candidate_factors = [
                    item
                    for item in previous_base_factors
                    if item not in parent_input_base_factors
                ]
                continuation_context = self.build_continuation_context(
                    result=analysis_result,
                    request_payload={
                        "base_factors": parent_input_base_factors,
                        "direction": (analysis_request or {}).get("direction", direction),
                    },
                    prompt=prompt,
                    factor_update_mode=factor_update_mode,
                    additional_factor_count=additional_factor_count_per_round,
                )
                continuation_seed_expression = str((last_selection_context or {}).get("continuation_seed_expression") or "").strip()
                if continuation_seed_expression:
                    continuation_context["parent_expression"] = continuation_seed_expression
                    continuation_context["parent_raw_expression"] = continuation_seed_expression
                continuation_hypothesis = {
                    "hypothesis": f"第 {round_index} 轮基于上一轮单轮研究结果，继续围绕 {continuation_context.get('recommended_goal') or '综合优化'} 调整基础因子组合。",
                    "reason": continuation_context.get("primary_problem") or "上一轮仍存在可提升空间。",
                    "target_goal": continuation_context.get("recommended_goal") or "提升综合分数",
                    "primary_problem": continuation_context.get("primary_problem"),
                    "secondary_problem": continuation_context.get("secondary_problem"),
                    "selection_instructions": continuation_context.get("selection_instructions"),
                    "preferred_keywords": list(continuation_context.get("preferred_keywords") or []),
                    "avoid_keywords": list(continuation_context.get("avoid_keywords") or []),
                    "selection_confidence": continuation_context.get("selection_confidence"),
                    "should_adjust_base_factors": continuation_context.get("should_adjust_base_factors"),
                    "hold_reason": continuation_context.get("hold_reason"),
                    "current_base_factors": list(previous_base_factors),
                    "candidate_factors": list((last_selection_context or {}).get("raw_candidate_factors") or applied_candidate_factors),
                    "selected_for_next_round": list((last_selection_context or {}).get("selected_for_next_round") or applied_candidate_factors),
                    "should_adjust_base_factors": bool((last_selection_context or {}).get("should_adjust_base_factors", bool(applied_candidate_factors))),
                    "hold_reason": (last_selection_context or {}).get("hold_reason"),
                    "selection_confidence": (last_selection_context or {}).get("selection_confidence"),
                    "replace_base_factors": list((last_selection_context or {}).get("replace_base_factors") or []),
                    "factor_update_mode": factor_update_mode,
                }

            current_round_candidates: list[dict[str, Any]] = []

            def _build_in_progress_round_snapshot() -> dict[str, Any]:
                current_scores = [
                    self._extract_result_metrics({"factors": [item]}).get(campaign_metric_key, float(item.get("score", 0.0) or 0.0))
                    for item in current_round_candidates
                ]
                factor_usage = self._build_factor_usage_summary(
                    current_base_factors=current_base_factors,
                    previous_base_factors=analysis_round.get("input_base_factors", []) if analysis_round else [],
                    best_factor=current_round_candidates[0] if current_round_candidates else None,
                )
                return {
                    "round_index": round_index,
                    "task_id": f"campaign-{campaign_run_id}-round-{round_index}-in-progress",
                    "best_score": max(current_scores) if current_scores else 0.0,
                    "avg_score": round(float(np.mean(current_scores)), 4) if current_scores else 0.0,
                    "input_base_factors": list(current_base_factors),
                    "previous_base_factors": analysis_round.get("input_base_factors", []) if analysis_round else [],
                    "factor_changes": self._build_factor_changes(
                        previous_base_factors=analysis_round.get("input_base_factors", []) if analysis_round else [],
                        current_base_factors=current_base_factors,
                    ),
                    "factor_update_mode": "initial" if round_index == 1 else factor_update_mode,
                    "selected_factors": list(current_base_factors),
                    "continuation_hypothesis": continuation_hypothesis,
                    "continuation_plan": continuation_hypothesis,
                    "continuation_feedback": None,
                    "retained_count": 0,
                    "retained_factors": [],
                    "factor_usage": factor_usage,
                    "all_factors": list(current_round_candidates),
                    "round_evaluation": continuation_context if round_index > 1 else None,
                    "final_round_evaluation": continuation_context,
                }

            def _round_progress(done_count: int, total_count: int, candidate: dict[str, Any]) -> None:
                current_round_candidates.append(candidate)
                displayed_total_count = max(int(total_count or 0), 1)
                displayed_done_count = min(len(current_round_candidates), displayed_total_count)
                current_scores = [
                    self._extract_result_metrics({"factors": [item]}).get(campaign_metric_key, float(item.get("score", 0.0) or 0.0))
                    for item in current_round_candidates
                ]
                if progress_callback is not None:
                    completed_best = aggregate_best + ([max(current_scores)] if current_scores else [])
                    completed_avg = aggregate_avg + ([sum(current_scores) / len(current_scores)] if current_scores else [])
                    progress_callback(
                        {
                            "current_round": round_index,
                            "total_rounds": exploration_rounds,
                            "latest_round": _build_in_progress_round_snapshot(),
                            "rounds": list(rounds),
                            "retained_count": len(retained_factors),
                            "fitness_history": {
                                "best": [round(max(completed_best[: idx + 1]), 4) for idx in range(len(completed_best))] if completed_best else [],
                                "average": [round(float(np.mean(completed_avg[: idx + 1])), 4) for idx in range(len(completed_avg))] if completed_avg else [],
                            },
                            "best_fitness": max(completed_best) if completed_best else 0.0,
                            "avg_fitness": round(float(np.mean(completed_avg)), 4) if completed_avg else 0.0,
                            "candidates": list(current_round_candidates),
                            "current_generation": displayed_done_count,
                            "total_generations": displayed_total_count,
                        }
                    )

            try:
                round_result = self.run_auto_mining(
                    prompt=current_prompt,
                    base_factors=current_base_factors,
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    benchmark=benchmark,
                    n_groups=n_groups,
                    holding_period=holding_period,
                    n_candidates=n_candidates_per_round,
                    direction=direction,
                    neutralize_industry=neutralize_industry,
                    neutralize_cap=neutralize_cap,
                    previous_expressions=previous_expressions,
                    continuation_context=continuation_context,
                    progress_callback=_round_progress,
                )
            except Exception as exc:
                campaign_failure_reason = str(exc)
                failed_round_task_id = f"campaign-{campaign_run_id}-round-{round_index}-failed-{datetime.now().strftime('%H%M%S')}"
                failed_feedback = {
                    "decision": False,
                    "accepted_as_best": False,
                    "observations": (
                        "本轮未产出可执行候选表达式，连续探索在当前 parent 上无法继续推进；"
                        "本次 campaign 保留此前已验证的最佳结果并结束。"
                    ),
                    "reason": campaign_failure_reason,
                    "fallback_parent_strategy": parent_selection_strategy if best_round_result is not None else None,
                }
                failed_round_summary = {
                    "round_index": round_index,
                    "task_id": failed_round_task_id,
                    "best_score": 0.0,
                    "avg_score": 0.0,
                    "input_base_factors": list(current_base_factors),
                    "previous_base_factors": analysis_round.get("input_base_factors", []) if analysis_round else [],
                    "factor_changes": self._build_factor_changes(
                        previous_base_factors=analysis_round.get("input_base_factors", []) if analysis_round else [],
                        current_base_factors=current_base_factors,
                    ),
                    "factor_update_mode": "initial" if round_index == 1 else factor_update_mode,
                    "selected_factors": list(current_base_factors),
                    "selection_rationale": "",
                    "per_factor_reason": {},
                    "continuation_hypothesis": continuation_hypothesis,
                    "continuation_plan": continuation_hypothesis,
                    "continuation_feedback": failed_feedback,
                    "retained_count": 0,
                    "retained_factors": [],
                    "factor_usage": self._build_factor_usage_summary(
                        current_base_factors=current_base_factors,
                        previous_base_factors=analysis_round.get("input_base_factors", []) if analysis_round else [],
                        best_factor=None,
                    ),
                    "all_factors": [],
                    "round_evaluation": continuation_context if round_index > 1 else None,
                    "final_round_evaluation": continuation_context,
                    "status": "failed",
                    "error": campaign_failure_reason,
                }
                rounds.append(failed_round_summary)
                last_round_summary = failed_round_summary
                if progress_callback is not None:
                    progress_callback(
                        {
                            "current_round": round_index,
                            "total_rounds": exploration_rounds,
                            "latest_round": failed_round_summary,
                            "rounds": list(rounds),
                            "retained_count": len(retained_factors),
                            "fitness_history": {
                                "best": [round(max(aggregate_best[: idx + 1]), 4) for idx in range(len(aggregate_best))],
                                "average": [round(float(np.mean(aggregate_avg[: idx + 1])), 4) for idx in range(len(aggregate_avg))],
                            },
                            "best_fitness": max(aggregate_best) if aggregate_best else 0.0,
                            "avg_fitness": round(float(np.mean(aggregate_avg)), 4) if aggregate_avg else 0.0,
                            "candidates": [],
                            "current_generation": 0,
                            "total_generations": n_candidates_per_round,
                        }
                    )
                break
            previous_expressions.extend([factor.get("expression", "") for factor in round_result.get("factors", [])])

            round_task_id = f"campaign-{campaign_run_id}-round-{round_index}-{datetime.now().strftime('%H%M%S')}"
            round_result["task_id"] = round_task_id
            selected = self.filter_retained_factors(round_result.get("factors", []), retention_filter)
            if not selected:
                selected = round_result.get("factors", [])[:1]

            factor_changes = self._build_factor_changes(
                previous_base_factors=analysis_round.get("input_base_factors", []) if analysis_round else [],
                current_base_factors=current_base_factors,
            )
            factor_usage = self._build_factor_usage_summary(
                current_base_factors=current_base_factors,
                previous_base_factors=analysis_round.get("input_base_factors", []) if analysis_round else [],
                best_factor=round_result.get("factors", [None])[0] if round_result.get("factors") else None,
            )
            continuation_feedback = self._build_continuation_feedback(
                previous_best_score=best_round_result.get("best_score") if best_round_result else None,
                current_best_score=round_result.get("best_score", 0.0),
                retention_count=len(selected),
                direction=direction or "score",
                previous_best_result=best_round_result,
                current_result=round_result,
                factor_usage=factor_usage,
                continuation_hypothesis=continuation_hypothesis,
            )
            continuation_seed_factor = self._pick_continuation_seed_candidate(
                round_result=round_result,
                continuation_context=continuation_context,
                base_factor_codes=self.resolve_base_factor_codes(current_base_factors),
            )

            round_summary = {
                "round_index": round_index,
                "task_id": round_task_id,
                "best_score": round_result.get("best_score", 0.0),
                "avg_score": round_result.get("avg_score", 0.0),
                "input_base_factors": list(current_base_factors),
                "previous_base_factors": analysis_round.get("input_base_factors", []) if analysis_round else [],
                "factor_changes": factor_changes,
                "factor_update_mode": "initial" if round_index == 1 else factor_update_mode,
                "selected_factors": list(current_base_factors),
                "selection_rationale": self._build_selection_rationale(
                    current_base_factors=current_base_factors,
                    selected=selected,
                    planning_context=continuation_context,
                    result_evaluation=round_result.get("round_evaluation"),
                    factor_update_mode=factor_update_mode,
                    round_index=round_index,
                ),
                "per_factor_reason": self._build_per_factor_reason(
                    current_base_factors=current_base_factors,
                    planning_context=continuation_context,
                    result_evaluation=round_result.get("round_evaluation"),
                ),
                "continuation_hypothesis": continuation_hypothesis,
                "continuation_plan": continuation_hypothesis,
                "continuation_feedback": continuation_feedback,
                "retained_count": len(selected),
                "retained_factors": selected,
                "factor_usage": factor_usage,
                "all_factors": round_result.get("factors", []),
                "round_evaluation": round_result.get("round_evaluation"),
                "final_round_evaluation": round_result.get("round_evaluation"),
                "continuation_seed_factor": continuation_seed_factor,
            }
            rounds.append(round_summary)

            round_metric_snapshot = self._extract_result_metrics(round_result)
            round_campaign_best = round_metric_snapshot.get(campaign_metric_key, round_result.get("best_score", 0.0))
            round_display_best = round_result.get("best_score", 0.0)
            round_display_avg = round_result.get("avg_score", 0.0)
            # Campaign 对外展示的研究曲线统一使用综合分数量纲，避免 best/average 混入不同指标。
            aggregate_best.append(round_display_best)
            aggregate_avg.append(round_display_avg if round_display_avg is not None else round_display_best)
            retained_factors = selected
            last_round_result = round_result
            current_round_request = {
                "base_factors": list(current_base_factors),
                "direction": direction,
                "start_date": start_date,
                "end_date": end_date,
                "universe": universe,
                "benchmark": benchmark,
            }
            last_round_request = {
                "base_factors": list(current_base_factors),
                "direction": direction,
                "start_date": start_date,
                "end_date": end_date,
                "universe": universe,
                "benchmark": benchmark,
            }
            last_round_summary = round_summary

            if global_best_result is None or round_result.get("best_score", float("-inf")) >= float(global_best_result.get("best_score", float("-inf"))):
                global_best_result = round_result
                global_best_task_id = round_task_id
                global_best_summary = round_summary
                global_best_retained_factors = list(selected)

            if best_round_result is None:
                best_round_result = round_result
                best_round_task_id = round_task_id
                best_round_request = dict(current_round_request)
                best_round_summary = round_summary
                best_round_retained_factors = list(selected)
            else:
                if parent_selection_strategy == "latest_round":
                    best_round_result = round_result
                    best_round_task_id = round_task_id
                    best_round_request = dict(current_round_request)
                    best_round_summary = round_summary
                    best_round_retained_factors = list(selected)
                elif continuation_feedback.get("accepted_as_best") and round_campaign_best >= self._extract_result_metrics(best_round_result).get(
                    campaign_metric_key,
                    best_round_result.get("best_score", 0.0),
                ):
                    best_round_result = round_result
                    best_round_task_id = round_task_id
                    best_round_request = dict(current_round_request)
                    best_round_summary = round_summary
                    best_round_retained_factors = list(selected)

            if parent_selection_strategy == "latest_round":
                continuation_parent_result = last_round_result
                continuation_parent_request = dict(last_round_request) if last_round_request else None
                continuation_parent_summary = last_round_summary
            else:
                continuation_parent_result = best_round_result
                continuation_parent_request = dict(best_round_request) if best_round_request else None
                continuation_parent_summary = best_round_summary

            if progress_callback is not None:
                progress_callback(
                    {
                        "current_round": round_index,
                        "total_rounds": exploration_rounds,
                        "latest_round": round_summary,
                        "rounds": list(rounds),
                        "retained_count": len(retained_factors),
                        "fitness_history": {
                            "best": [round(max(aggregate_best[: idx + 1]), 4) for idx in range(len(aggregate_best))],
                            "average": [round(float(np.mean(aggregate_avg[: idx + 1])), 4) for idx in range(len(aggregate_avg))],
                        },
                        "best_fitness": max(aggregate_best) if aggregate_best else 0.0,
                        "avg_fitness": round(float(np.mean(aggregate_avg)), 4) if aggregate_avg else 0.0,
                        "candidates": round_result.get("factors", []),
                        "current_generation": len(round_result.get("factors", [])),
                        "total_generations": n_candidates_per_round,
                    }
                )

            if round_index >= exploration_rounds:
                break

            selection = self.select_continue_factors(
                parent_result=continuation_parent_result,
                parent_request=continuation_parent_request
                or {
                    "base_factors": list(current_base_factors),
                    "direction": direction,
                    "start_date": start_date,
                    "end_date": end_date,
                    "universe": universe,
                    "benchmark": benchmark,
                },
                prompt=prompt,
                direction=direction,
                factor_update_mode=factor_update_mode,
                max_factor_count=additional_factor_count_per_round if factor_update_mode == "append" else max(
                    len(current_base_factors), additional_factor_count_per_round
                ),
                candidate_limit=80,
                current_base_factors=list(current_base_factors),
            )
            selection_context = selection.get("continuation_context") or {}
            should_adjust_base_factors = bool(selection_context.get("should_adjust_base_factors"))
            raw_candidate_factors = list(selection.get("selected_factors") or [])
            candidate_factors = raw_candidate_factors if should_adjust_base_factors else []
            fallback_parent_strategy = str(continuation_feedback.get("fallback_parent_strategy") or "").strip().lower()
            should_force_small_step_exploration = (
                parent_selection_strategy != "latest_round"
                and fallback_parent_strategy == "best_score_so_far"
                and not should_adjust_base_factors
                and bool(raw_candidate_factors)
            )
            if should_force_small_step_exploration:
                candidate_factors = list(raw_candidate_factors[: max(additional_factor_count_per_round, 1)])
                should_adjust_base_factors = True
                selection_context["should_adjust_base_factors"] = True
                selection_context["hold_reason"] = ""
                selection_context["selection_confidence"] = max(
                    int(selection_context.get("selection_confidence") or 0),
                    1,
                )
                selection_context["forced_exploration"] = True
                selection_context["forced_exploration_reason"] = (
                    "虽然最佳轮整体仍偏稳健，但最近一轮已经明显退化；为避免连续原地踏步，"
                    "基于最佳轮候选执行一次受控的小步探索。"
                )
            parent_base_factors_for_next_round = list(
                (continuation_parent_request or {}).get("base_factors")
                or current_base_factors
            )
            replace_base_factors = _dedupe_preserve_order(
                [str(item).strip() for item in list(selection_context.get("replace_base_factors") or []) if str(item or "").strip()]
            )
            applicable_replace_base_factors = [
                item
                for item in replace_base_factors
                if item in parent_base_factors_for_next_round
            ]
            continuation_seed_factor = round_summary.get("continuation_seed_factor") if isinstance(round_summary.get("continuation_seed_factor"), dict) else None
            last_selection_context = {
                "raw_candidate_factors": list(raw_candidate_factors),
                "selected_for_next_round": list(candidate_factors),
                "should_adjust_base_factors": should_adjust_base_factors,
                "hold_reason": selection_context.get("hold_reason"),
                "selection_confidence": selection_context.get("selection_confidence"),
                "parent_base_factors": parent_base_factors_for_next_round,
                "replace_base_factors": list(replace_base_factors),
                "forced_exploration": bool(selection_context.get("forced_exploration")),
                "forced_exploration_reason": selection_context.get("forced_exploration_reason"),
                "continuation_seed_expression": continuation_seed_factor.get("expression") if continuation_seed_factor else None,
            }
            if round_summary.get("continuation_hypothesis") is not None:
                round_summary["continuation_hypothesis"]["replace_base_factors"] = list(replace_base_factors)
                round_summary["continuation_hypothesis"]["next_round_candidate_factors"] = list(raw_candidate_factors)
                round_summary["continuation_hypothesis"]["next_round_selected_factors"] = list(candidate_factors)
                round_summary["continuation_hypothesis"]["next_round_should_adjust_base_factors"] = should_adjust_base_factors
                round_summary["continuation_hypothesis"]["next_round_hold_reason"] = selection_context.get("hold_reason")
                round_summary["continuation_hypothesis"]["next_round_selection_confidence"] = selection_context.get("selection_confidence")
                round_summary["continuation_hypothesis"]["next_round_parent_base_factors"] = list(parent_base_factors_for_next_round)
                round_summary["continuation_hypothesis"]["next_round_replace_base_factors"] = list(replace_base_factors)
                round_summary["continuation_hypothesis"]["next_round_forced_exploration"] = bool(selection_context.get("forced_exploration"))
                round_summary["continuation_hypothesis"]["next_round_forced_exploration_reason"] = selection_context.get("forced_exploration_reason")
                round_summary["continuation_hypothesis"]["next_round_seed_expression"] = continuation_seed_factor.get("expression") if continuation_seed_factor else None

            if should_adjust_base_factors:
                if factor_update_mode == "reselect":
                    current_base_factors = candidate_factors or current_base_factors
                else:
                    next_parent_base_factors = [
                        item for item in parent_base_factors_for_next_round
                        if item not in applicable_replace_base_factors
                    ]
                    current_base_factors = _dedupe_preserve_order(next_parent_base_factors + candidate_factors)
            elif parent_selection_strategy != "latest_round":
                current_base_factors = list(parent_base_factors_for_next_round)
            current_prompt = prompt

        fitness_history = {
            "best": [round(max(aggregate_best[: idx + 1]), 4) for idx in range(len(aggregate_best))],
            "average": [round(float(np.mean(aggregate_avg[: idx + 1])), 4) for idx in range(len(aggregate_avg))],
        }

        return {
            "rounds": rounds,
            "retained_factors": retained_factors,
            "latest_round_retained_factors": retained_factors,
            "best_result_retained_factors": global_best_retained_factors or best_round_retained_factors or retained_factors,
            "final_round_task_id": last_round_result.get("task_id") if last_round_result else None,
            "final_round_result": last_round_result,
            "latest_round_task_id": last_round_summary.get("task_id") if last_round_summary else None,
            "latest_round_result": last_round_result if campaign_failure_reason is None else None,
            "best_parent_task_id": best_round_task_id,
            "best_parent_result": best_round_result,
            "best_parent_retained_factors": best_round_retained_factors,
            "campaign_metric": campaign_metric_key,
            "best_score": float((global_best_result or best_round_result or {}).get("best_score", 0.0) or 0.0),
            "avg_score": round(float(np.mean(aggregate_avg)), 4) if aggregate_avg else 0.0,
            "fitness_history": fitness_history,
            "selection_mode": retention_filter.get("match_mode", "all"),
            "retention_filter": retention_filter,
            "completed_with_failures": campaign_failure_reason is not None,
            "failure_reason": campaign_failure_reason,
        }

    def filter_retained_factors(
        self,
        factors: list[dict[str, Any]],
        retention_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not factors:
            return []

        checks: list[Callable[[dict[str, Any]], bool]] = []
        score_min = retention_filter.get("score_min")
        if score_min is not None:
            checks.append(lambda factor: _safe_float(factor.get("score")) >= _safe_float(score_min))

        ratings = [str(item) for item in retention_filter.get("wq_ratings", []) if item]
        if ratings:
            checks.append(lambda factor: str(factor.get("wq_brain", {}).get("wq_rating", "")) in ratings)

        ls_sharpe_min = retention_filter.get("ls_sharpe_min")
        if ls_sharpe_min is not None:
            checks.append(
                lambda factor: _safe_float(factor.get("backtest_summary", {}).get("long_short_sharpe"))
                >= _safe_float(ls_sharpe_min)
            )

        ls_return_min = retention_filter.get("ls_return_min")
        if ls_return_min is not None:
            checks.append(
                lambda factor: _safe_float(factor.get("backtest_summary", {}).get("long_short_annual"))
                >= _safe_float(ls_return_min)
            )

        wq_return_min = retention_filter.get("wq_return_min")
        if wq_return_min is not None:
            checks.append(
                lambda factor: _safe_float(factor.get("wq_brain", {}).get("wq_returns"))
                >= _safe_float(wq_return_min)
            )

        if not checks:
            return list(factors[: max(min(len(factors), 3), 1)])

        match_mode = str(retention_filter.get("match_mode") or "all").lower()
        retained: list[dict[str, Any]] = []
        for factor in factors:
            results = [check(factor) for check in checks]
            if (match_mode == "any" and any(results)) or (match_mode != "any" and all(results)):
                retained.append(factor)

        return retained

    def evaluate_expression(
        self,
        *,
        expression: str,
        prompt: str,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        direction: str,
        neutralize_industry: bool,
        neutralize_cap: bool,
    ) -> FactorEvaluationResult | None:
        validation_result = self._validate_candidate_expression(
            expression=expression,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            n_groups=n_groups,
            holding_period=holding_period,
            neutralize_industry=neutralize_industry,
            neutralize_cap=neutralize_cap,
        )
        if validation_result and not validation_result.get("valid", False):
            logger.info(
                "QuantGPT validation 未通过，候选失效 %s：%s",
                expression,
                validation_result.get("message") or "unknown validation failure",
            )
            return None

        try:
            panel_df = self._quantgpt_engine.build_panel_data(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                expression=expression,
                stock_data_loader=self.data_service.get_stock_data,
            )
            if panel_df.empty:
                logger.info("QuantGPT 执行器未构建出有效 panel，候选失效：%s", expression)
                return None

            execution = self._quantgpt_engine.execute_on_panel(panel_df, expression)
            panel_result = self._factor_evaluation_service.evaluate_factor_panel(
                expression=expression,
                prompt=prompt,
                panel_df=panel_df,
                factor_series=execution.factor_series if execution.factor_series is not None else pd.Series(dtype=float),
                benchmark=benchmark,
                start_date=start_date,
                end_date=end_date,
                n_groups=n_groups,
                holding_period=holding_period,
                direction=direction,
                benchmark_loader=self._load_benchmark_returns,
                report_writer=self._write_candidate_report,
                engine_type=execution.engine_type,
                dialect=execution.dialect,
                canonical_expression=execution.canonical_expression,
                canonical_ast=execution.canonical_ast,
                diagnostics=execution.diagnostics,
                execution_meta=execution.execution_meta,
                metrics_source=execution.metrics_source,
            )
            if panel_result is not None:
                panel_result.execution_meta.setdefault("research_tools", {})
                if validation_result is not None:
                    panel_result.execution_meta["research_tools"]["validation"] = validation_result
                return panel_result

            diagnosis_result = self._diagnose_candidate_failure(
                expression=expression,
                start_date=start_date,
                end_date=end_date,
                benchmark=benchmark,
                n_groups=n_groups,
                holding_period=holding_period,
                neutralize_industry=neutralize_industry,
                neutralize_cap=neutralize_cap,
            )
            logger.info("QuantGPT 面板评估未产出有效结果，候选失效：%s", expression)
            if diagnosis_result and diagnosis_result.get("success"):
                logger.info("QuantGPT diagnosis %s：%s", expression, diagnosis_result.get("report"))
            return None
        except Exception as exc:
            diagnosis_result = self._diagnose_candidate_failure(
                expression=expression,
                start_date=start_date,
                end_date=end_date,
                benchmark=benchmark,
                n_groups=n_groups,
                holding_period=holding_period,
                neutralize_industry=neutralize_industry,
                neutralize_cap=neutralize_cap,
            )
            diagnosis_hint = ""
            if diagnosis_result and diagnosis_result.get("success"):
                diagnosis_hint = f"，diagnosis={diagnosis_result.get('report')}"
            logger.warning("QuantGPT 执行器评估失败，候选失效 %s：%s%s", expression, exc, diagnosis_hint)
            return None

    def _build_round_evaluation(
        self,
        *,
        prompt: str,
        base_factors: list[str],
        best_evaluation: FactorEvaluationResult,
        direction: str,
    ) -> dict[str, Any]:
        normalized_direction = str(direction or "score").strip().lower()
        metric_snapshot = {
            "score": best_evaluation.score,
            "report_sharpe": best_evaluation.report_metrics.get("sharpe"),
            "report_max_drawdown": best_evaluation.report_metrics.get("max_drawdown"),
            "report_volatility": best_evaluation.report_metrics.get("volatility"),
            "ls_sharpe": best_evaluation.backtest_summary.get("long_short_sharpe"),
            "ls_return": best_evaluation.backtest_summary.get("long_short_annual"),
            "rank_ic": best_evaluation.backtest_summary.get("rank_ic_mean"),
            "turnover": best_evaluation.backtest_summary.get("turnover"),
            "wq_fitness": best_evaluation.backtest_summary.get("wq_fitness"),
        }

        weaknesses = [
            str(item).strip()
            for item in list(best_evaluation.interpretation.get("weaknesses") or [])
            if str(item or "").strip()
        ]
        current_score = float(best_evaluation.score or 0.0)
        inferred_primary_problem = infer_primary_problem_from_metrics(
            best_evaluation.report_metrics,
            best_evaluation.backtest_summary,
            current_score,
        )
        inferred_primary_label = _classify_round_problem(inferred_primary_problem)
        prioritized_problem_labels = _rank_round_problem_candidates(
            direction=normalized_direction,
            report_metrics=best_evaluation.report_metrics,
            backtest_summary=best_evaluation.backtest_summary,
            score=current_score,
        )
        primary_problem = _build_metric_specific_problem_text(
            label=prioritized_problem_labels[0] if prioritized_problem_labels else inferred_primary_label,
            report_metrics=best_evaluation.report_metrics,
            backtest_summary=best_evaluation.backtest_summary,
            score=current_score,
        )
        ordered_problem_labels = (
            [inferred_primary_label] + [label for label in prioritized_problem_labels if label != inferred_primary_label]
            if inferred_primary_label
            else prioritized_problem_labels
        )
        for label in ordered_problem_labels:
            matched = next(
                (weakness for weakness in weaknesses if _classify_round_problem(weakness) == label),
                "",
            )
            if matched and label == inferred_primary_label:
                primary_problem = matched
                break
        secondary_problem = ""
        primary_label = _classify_round_problem(primary_problem)
        for weakness in weaknesses:
            normalized = str(weakness or "").strip()
            weakness_label = _classify_round_problem(normalized)
            if not normalized or normalized == primary_problem or weakness_label == primary_label:
                continue
            secondary_problem = normalized
            break
        if not secondary_problem:
            for label in ordered_problem_labels:
                if label == primary_label:
                    continue
                matched = next(
                    (weakness for weakness in weaknesses if _classify_round_problem(weakness) == label),
                    "",
                )
                if matched:
                    secondary_problem = matched
                    break
                specific_text = _build_metric_specific_problem_text(
                    label=label,
                    report_metrics=best_evaluation.report_metrics,
                    backtest_summary=best_evaluation.backtest_summary,
                    score=current_score,
                )
                if specific_text and specific_text != primary_problem:
                    secondary_problem = specific_text
                    break
        if normalized_direction:
            recommended_goal = normalized_direction
        elif "Sharpe" in primary_problem or "风险调整后收益" in primary_problem:
            recommended_goal = "ls_sharpe"
        elif "收益" in primary_problem:
            recommended_goal = "ls_return"
        elif "WQ" in primary_problem or "Fitness" in primary_problem:
            recommended_goal = "wq_fitness"
        else:
            recommended_goal = "score"

        suggested_actions = [
            f"下一轮只围绕“{recommended_goal}”做小步优化，不要同时追求多个目标。",
            "优先保留当前有效结构，只调整与主要短板直接相关的基础因子或表达式局部。",
        ]
        if base_factors:
            suggested_actions.append("先评估当前基础因子是否存在语义重复或风格过于单一，再决定补充或替换。")
        if secondary_problem:
            suggested_actions.append(f"次要问题可暂缓，先不要同时处理：{secondary_problem}")
        research_tools = (getattr(best_evaluation, "execution_meta", {}) or {}).get("research_tools", {})
        return {
            "prompt": prompt,
            "base_factors": list(base_factors),
            "primary_problem": primary_problem,
            "secondary_problem": secondary_problem,
            "recommended_goal": recommended_goal,
            "suggested_actions": suggested_actions[:3],
            "metric_snapshot": metric_snapshot,
            "research_tools": research_tools,
        }

    def _load_benchmark_returns(
        self,
        benchmark: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series | None:
        try:
            benchmark_df = self.data_service.get_benchmark_returns(
                benchmark=benchmark,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            logger.warning("加载 benchmark 数据失败，将生成无基准报告：%s", exc)
            return None

        if benchmark_df is None or benchmark_df.empty or "daily_return" not in benchmark_df.columns:
            return None

        series = pd.Series(
            pd.to_numeric(benchmark_df["daily_return"], errors="coerce").values,
            index=pd.to_datetime(benchmark_df["trade_date"]),
            dtype=float,
        ).dropna()
        return series.sort_index() if not series.empty else None

    def _write_candidate_report(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series | None,
        periods_per_year: int,
    ) -> tuple[dict[str, Any], str]:
        report_result = generate_report(
            strategy_returns,
            benchmark_returns=benchmark_returns,
            title="Factor Top-Group Backtest",
            output_dir=str(AUTO_MINING_REPORT_DIR),
            periods_per_year=periods_per_year,
        )
        report_path = Path(report_result["report_path"])
        return report_result["metrics"], f"/api/mining/reports/{report_path.name}"

    def _format_candidate_payload(
        self,
        *,
        evaluation: FactorEvaluationResult | Any,
        prompt: str,
        index: int,
        base_factors: list[str],
        round_evaluation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw_expression = getattr(evaluation, "raw_expression", evaluation.expression)
        engine_type = getattr(evaluation, "engine_type", "factorhub")
        dialect = getattr(evaluation, "dialect", "factorhub_native")
        canonical_expression = getattr(evaluation, "canonical_expression", None)
        canonical_ast = getattr(evaluation, "canonical_ast", None)
        execution_meta = getattr(evaluation, "execution_meta", {})
        details = {
            "params": {
                "expression": raw_expression,
                "prompt": prompt,
            },
            "expression": evaluation.expression,
            "raw_expression": raw_expression,
            "engine_type": engine_type,
            "dialect": dialect,
            "canonical_expression": canonical_expression,
            "canonical_ast": canonical_ast,
            "llm": {
                "prompt": prompt,
                "generated_expression": raw_expression,
            },
            "report_url": evaluation.report_url,
            "report_metrics": evaluation.report_metrics,
            "backtest_summary": evaluation.backtest_summary,
            "wq_brain": evaluation.wq_brain,
            "component_scores": evaluation.component_scores,
            "anti_overfit": evaluation.anti_overfit,
            "interpretation": evaluation.interpretation,
            "scoring": {
                "score": evaluation.score,
                "grade": evaluation.grade,
            },
            "diagnostics": evaluation.diagnostics,
            "execution_meta": execution_meta,
            "round_evaluation": round_evaluation,
        }
        return {
            "name": f"Auto_Factor_{index + 1}",
            "expression": evaluation.expression,
            "raw_expression": raw_expression,
            "score": evaluation.score,
            "grade": evaluation.grade,
            "fitness": evaluation.wq_brain["wq_fitness"],
            "ic": evaluation.backtest_summary["ic_mean"],
            "ir": evaluation.backtest_summary["ic_ir"],
            "rank_ic": evaluation.backtest_summary["rank_ic_mean"],
            "sharpe": evaluation.backtest_summary["long_short_sharpe"],
            "status": "computed",
            "source": "factorhub_auto_mining",
            "engine_type": engine_type,
            "dialect": dialect,
            "canonical_expression": canonical_expression,
            "canonical_ast": canonical_ast,
            "report_url": evaluation.report_url,
            "report_metrics": evaluation.report_metrics,
            "backtest_summary": evaluation.backtest_summary,
            "component_scores": evaluation.component_scores,
            "anti_overfit": evaluation.anti_overfit,
            "wq_brain": evaluation.wq_brain,
            "interpretation": evaluation.interpretation,
            "task_details": details,
            "quantgpt_task_details": details,
            "task_id": None,
            "execution_meta": execution_meta,
            "base_factors": list(base_factors),
        }

    def _build_factor_changes(
        self,
        *,
        previous_base_factors: list[str],
        current_base_factors: list[str],
    ) -> dict[str, list[str]]:
        previous_set = set(previous_base_factors)
        current_set = set(current_base_factors)
        return {
            "added": [item for item in current_base_factors if item not in previous_set],
            "removed": [item for item in previous_base_factors if item not in current_set],
            "retained": [item for item in current_base_factors if item in previous_set],
        }

    def _build_factor_usage_summary(
        self,
        *,
        current_base_factors: list[str],
        previous_base_factors: list[str],
        best_factor: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        if not best_factor:
            return {
                "used_base_factors": [],
                "unused_base_factors": list(current_base_factors),
                "used_new_factors": [],
                "unused_new_factors": [],
            }

        expression_candidates = [
            str(best_factor.get("expression") or "").strip(),
            str(best_factor.get("canonical_expression") or "").strip(),
            str(((best_factor.get("task_details") or {}).get("expression")) or "").strip(),
            str(((best_factor.get("task_details") or {}).get("canonical_expression")) or "").strip(),
        ]
        expression_text = next((item for item in expression_candidates if item), "")
        base_factor_codes = {
            name: code
            for name, code in zip(
                current_base_factors,
                self.resolve_base_factor_codes(current_base_factors),
            )
            if str(name or "").strip() and str(code or "").strip()
        }
        used_base_factors = [
            name for name in current_base_factors
            if _expression_uses_code(expression_text, base_factor_codes.get(name, ""))
        ]
        round_evaluation = (best_factor.get("task_details") or {}).get("round_evaluation") or {}
        parent_expression = str(round_evaluation.get("parent_expression") or "").strip()
        inherited_new_factors: list[str] = []
        if parent_expression and _expression_contains_parent_anchor(expression_text, parent_expression):
            previous_set = set(previous_base_factors)
            inherited_new_factors = [
                name for name in current_base_factors
                if name not in previous_set and str(base_factor_codes.get(name, "")).strip()
            ]
            used_base_factors = _dedupe_preserve_order(used_base_factors + inherited_new_factors)
        unused_base_factors = [name for name in current_base_factors if name not in used_base_factors]
        previous_set = set(previous_base_factors)
        used_new_factors = [name for name in used_base_factors if name not in previous_set]
        unused_new_factors = [name for name in unused_base_factors if name not in previous_set]
        return {
            "used_base_factors": used_base_factors,
            "unused_base_factors": unused_base_factors,
            "used_new_factors": used_new_factors,
            "unused_new_factors": unused_new_factors,
        }

    def _build_selection_rationale(
        self,
        *,
        current_base_factors: list[str],
        selected: list[dict[str, Any]],
        planning_context: dict[str, Any] | None,
        result_evaluation: dict[str, Any] | None,
        factor_update_mode: str,
        round_index: int,
    ) -> str:
        active_context = planning_context or result_evaluation or {}
        reason = active_context.get("primary_problem") if active_context else "暂无"
        return (
            f"第 {round_index} 轮使用 {len(current_base_factors)} 个基础因子，"
            f"本轮保留 {len(selected)} 个候选；更新方式为 {factor_update_mode}。"
            f" 主要针对的问题：{reason}"
        )

    def _build_per_factor_reason(
        self,
        *,
        current_base_factors: list[str],
        planning_context: dict[str, Any] | None,
        result_evaluation: dict[str, Any] | None,
    ) -> dict[str, str]:
        active_context = planning_context or result_evaluation or {}
        reason = active_context.get("recommended_goal") if active_context else "提升综合分数"
        return {factor_name: f"围绕“{reason}”保留或补充该基础因子。" for factor_name in current_base_factors}

    def _build_continuation_feedback(
        self,
        *,
        previous_best_score: float | None,
        current_best_score: float,
        retention_count: int,
        direction: str,
        previous_best_result: dict[str, Any] | None = None,
        current_result: dict[str, Any] | None = None,
        factor_usage: dict[str, list[str]] | None = None,
        continuation_hypothesis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if previous_best_score is None:
            return {
                "decision": True,
                "accepted_as_best": True,
                "observations": "首轮结果已作为后续探索基准。",
                "hypothesis_evaluation": "初始轮建立基线。",
                "reason": "需要建立第一轮真实评估结果。",
                "score_delta": 0.0,
                "parent_best_score": None,
                "current_best_score": current_best_score,
                "next_hypothesis": f"围绕 {direction} 继续扩大有效信号。",
            }

        score_delta = round(current_best_score - previous_best_score, 4)
        acceptance = self._should_accept_round_as_continuation_parent(
            previous_best_result=previous_best_result,
            current_result=current_result,
            direction=direction,
            retention_count=retention_count,
            factor_usage=factor_usage,
            continuation_hypothesis=continuation_hypothesis,
        )
        return {
            "decision": acceptance["decision"],
            "accepted_as_best": acceptance["accepted_as_best"],
            "observations": "本轮结果已纳入连续探索轨迹。",
            "hypothesis_evaluation": acceptance["hypothesis_evaluation"],
            "reason": acceptance["reason"],
            "score_delta": score_delta,
            "parent_best_score": previous_best_score,
            "current_best_score": current_best_score,
            "next_hypothesis": acceptance["next_hypothesis"],
            "metric_deltas": acceptance["metric_deltas"],
            "fallback_parent_strategy": acceptance["fallback_parent_strategy"],
        }

    def _extract_result_metrics(self, result: dict[str, Any] | None) -> dict[str, float]:
        result = result or {}
        best_factor = (result.get("factors") or [{}])[0] if result.get("factors") else {}
        round_evaluation = result.get("round_evaluation") or best_factor.get("task_details", {}).get("round_evaluation") or {}
        metric_snapshot = round_evaluation.get("metric_snapshot") or {}
        report_metrics = best_factor.get("report_metrics") or {}
        backtest_summary = best_factor.get("backtest_summary") or {}
        wq_brain = best_factor.get("wq_brain") or {}
        return {
            "score": _safe_float(best_factor.get("score", result.get("best_score", metric_snapshot.get("score")))),
            "rank_ic": _safe_float(backtest_summary.get("rank_ic_mean", metric_snapshot.get("rank_ic"))),
            "ls_sharpe": _safe_float(backtest_summary.get("long_short_sharpe", metric_snapshot.get("ls_sharpe"))),
            "ls_return": _safe_float(backtest_summary.get("long_short_annual", metric_snapshot.get("ls_return"))),
            "turnover": _safe_float(backtest_summary.get("turnover", metric_snapshot.get("turnover"))),
            "report_sharpe": _safe_float(report_metrics.get("sharpe", metric_snapshot.get("report_sharpe"))),
            "wq_fitness": _safe_float(wq_brain.get("wq_fitness", metric_snapshot.get("wq_fitness"))),
            "wq_return": _safe_float(wq_brain.get("wq_returns", metric_snapshot.get("wq_return"))),
        }

    def _should_accept_round_as_continuation_parent(
        self,
        *,
        previous_best_result: dict[str, Any] | None,
        current_result: dict[str, Any] | None,
        direction: str,
        retention_count: int,
        factor_usage: dict[str, list[str]] | None = None,
        continuation_hypothesis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous_metrics = self._extract_result_metrics(previous_best_result)
        current_metrics = self._extract_result_metrics(current_result)
        score_delta = round(current_metrics["score"] - previous_metrics["score"], 4)
        metric_deltas = {
            "score": score_delta,
            "rank_ic": round(current_metrics["rank_ic"] - previous_metrics["rank_ic"], 6),
            "ls_sharpe": round(current_metrics["ls_sharpe"] - previous_metrics["ls_sharpe"], 6),
            "ls_return": round(current_metrics["ls_return"] - previous_metrics["ls_return"], 6),
            "turnover": round(current_metrics["turnover"] - previous_metrics["turnover"], 6),
            "report_sharpe": round(current_metrics["report_sharpe"] - previous_metrics["report_sharpe"], 6),
        }

        normalized_direction = str(direction or "score").strip().lower()
        target_metric = {
            "ls_sharpe": "ls_sharpe",
            "ls_return": "ls_return",
            "report_sharpe": "report_sharpe",
        }.get(normalized_direction, "score")
        target_delta = metric_deltas.get(target_metric, score_delta)
        rank_ic_delta = metric_deltas["rank_ic"]
        turnover_delta = metric_deltas["turnover"]

        severe_regression = False
        regression_reasons: list[str] = []
        if target_metric != "score" and target_delta < -0.03:
            severe_regression = True
            regression_reasons.append(f"{target_metric} 明显下降")
        if rank_ic_delta < -0.003:
            severe_regression = True
            regression_reasons.append("rank_ic 明显下降")
        if turnover_delta > 0.08:
            severe_regression = True
            regression_reasons.append("turnover 明显上升")

        expected_new_factors = list((continuation_hypothesis or {}).get("selected_for_next_round") or [])
        should_adjust_base_factors = bool((continuation_hypothesis or {}).get("should_adjust_base_factors"))
        used_new_factors = list((factor_usage or {}).get("used_new_factors") or [])
        missing_expected_new_factor_usage = (
            should_adjust_base_factors
            and bool(expected_new_factors)
            and not bool(used_new_factors)
        )
        if missing_expected_new_factor_usage:
            severe_regression = True
            regression_reasons.append("本轮最佳表达式未吸收新增基础因子")

        target_metric_improved = (
            target_metric != "score"
            and target_delta >= 0.05
            and rank_ic_delta >= -0.002
            and turnover_delta <= 0.05
        )
        improved = not severe_regression and (score_delta >= 0 or target_metric_improved)
        decision = improved or retention_count > 0
        if improved:
            if score_delta >= 0:
                hypothesis_evaluation = "本轮较上一轮有所提升，且目标指标未出现明显退化。"
                reason = "综合分数和关键目标指标表现可接受，当前轮可作为新的续轮 parent。"
            else:
                hypothesis_evaluation = "虽然综合分数略有回落，但本轮主目标指标已有明确改善，仍可升级为新的续轮 parent。"
                reason = f"{target_metric} 明显改善，且未伴随严重退化，当前轮可作为新的续轮 parent。"
        elif severe_regression:
            hypothesis_evaluation = "本轮分数或许有变化，但关键目标指标出现退化，不能作为新的续轮 parent。"
            reason = "；".join(regression_reasons) or "关键目标指标退化，回退到历史最佳轮继续探索。"
        else:
            hypothesis_evaluation = "本轮未超过上一轮最佳，但仍保留可用候选。"
            reason = "分数未超越上一轮，下一轮需调整基础因子结构。"

        return {
            "decision": decision,
            "accepted_as_best": improved,
            "hypothesis_evaluation": hypothesis_evaluation,
            "reason": reason,
            "metric_deltas": metric_deltas,
            "fallback_parent_strategy": "best_score_so_far" if not improved else None,
            "next_hypothesis": (
                f"继续围绕 {target_metric} 和 rank_ic 的平衡做更稳健的因子重组。"
                if severe_regression
                else f"继续针对 {direction} 做更强的因子重组。"
            ),
        }


auto_factor_mining_service = AutoFactorMiningService()
