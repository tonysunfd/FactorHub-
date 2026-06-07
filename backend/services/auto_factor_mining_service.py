"""
自动因子挖掘服务。
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.repositories.factor_repository import FactorRepository
from backend.services.expression_schema import FactorEvaluationResult
from backend.services.factor_evaluation_service import FactorEvaluationService
from backend.services.factor_generator_service import factor_generator_service
from backend.services.factor_service import factor_service
from backend.services.quantgpt_expression_engine import QuantGPTExpressionEngine
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
        extra_context: str | None = None,
        exclude_factors: list[str] | None = None,
    ) -> dict[str, Any]:
        db = get_db_session()
        try:
            repo = FactorRepository(db)
            factors = repo.get_all(active_only=True)
        finally:
            db.close()

        if not factors:
            return {
                "selected_factors": [],
                "selection_rationale": "当前因子库为空，无法筛选基础因子。",
                "per_factor_reason": {},
            }

        exclude = {name for name in (exclude_factors or []) if name}
        combined_prompt = " ".join(item for item in [prompt, extra_context or ""] if item).strip()
        prompt_tokens = set(_tokenize_text(combined_prompt))
        scored_items: list[tuple[float, Any, str]] = []
        for factor in factors[: max(candidate_limit, 1)]:
            if factor.name in exclude:
                continue

            searchable = " ".join(
                [
                    factor.name or "",
                    factor.description or "",
                    factor.category or "",
                    factor.code or "",
                ]
            ).lower()
            overlap = [token for token in prompt_tokens if token and token in searchable]
            score = float(len(overlap))

            if selection_mode == "manual_genetic" and factor.source == "preset":
                score += 0.25
            if any(keyword in searchable for keyword in ("close", "volume", "amount", "rsi", "macd", "sma")):
                score += 0.1
            if extra_context and any(keyword in searchable for keyword in ("volatility", "std", "atr", "drawdown", "风险", "波动")):
                score += 0.05
            if factor.category and factor.category.lower() in combined_prompt.lower():
                score += 0.2

            reason = (
                f"命中提示词：{', '.join(overlap[:4])}" if overlap else "作为默认量价/技术因子候选补充进入候选池"
            )
            scored_items.append((score, factor, reason))

        scored_items.sort(key=lambda item: (-item[0], item[1].name))
        selected = scored_items[: max(max_factor_count, 1)]
        selected_names = [factor.name for _, factor, _ in selected]
        per_factor_reason = {factor.name: reason for _, factor, reason in selected}
        rationale = (
            f"基于提示词与因子名称、描述、分类的匹配度，从前 {min(candidate_limit, len(factors))} 个候选中筛选出 "
            f"{len(selected_names)} 个基础因子。"
        )
        if extra_context:
            rationale += " 已结合上一轮诊断补充筛选偏好。"
        return {
            "selected_factors": selected_names,
            "selection_rationale": rationale,
            "per_factor_reason": per_factor_reason,
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

        summary_text = "；".join(_dedupe_preserve_order([primary_problem, *weaknesses[:2], *suggestions[:2]]))
        return {
            "base_factors": list(base_factors),
            "primary_problem": primary_problem,
            "recommended_goal": recommended_goal,
            "suggested_actions": _dedupe_preserve_order(suggestions),
            "weaknesses": _dedupe_preserve_order(weaknesses),
            "metric_snapshot": metric_snapshot,
            "summary_text": summary_text,
            "factor_update_mode": factor_update_mode,
            "additional_factor_count": max(additional_factor_count, 1),
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
    ) -> dict[str, Any]:
        context = self.build_continuation_context(
            result=parent_result,
            request_payload=parent_request,
            prompt=prompt,
            factor_update_mode=factor_update_mode,
            additional_factor_count=max_factor_count,
        )
        base_factors = context["base_factors"]
        exclude_factors = base_factors if factor_update_mode == "append" else []
        selection = self.select_factors(
            prompt=f"{prompt} {direction or ''}".strip(),
            max_factor_count=max_factor_count,
            candidate_limit=candidate_limit,
            selection_mode="auto",
            extra_context=context["summary_text"],
            exclude_factors=exclude_factors,
        )
        selection["continuation_context"] = context
        return selection

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

        pool = factor_generator_service.generate_hybrid_factors(
            base_factor_codes or ["close", "volume", "amount"],
            n_factors=max(n_candidates * 4, 24),
        )
        deduped: list[str] = []
        seen = {_normalize_expression(item) for item in (previous_expressions or [])}
        for item in pool:
            expression = str(item.get("expression") or "").strip()
            if not expression:
                continue
            key = _normalize_expression(expression)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(expression)
            if len(deduped) >= n_candidates:
                break
        return deduped

    def _generate_fallback_candidate_expressions(
        self,
        *,
        base_factor_codes: list[str],
        n_candidates: int,
        previous_expressions: list[str] | None = None,
        extra_excludes: list[str] | None = None,
    ) -> list[str]:
        pool = factor_generator_service.generate_hybrid_factors(
            base_factor_codes or ["close", "volume", "amount"],
            n_factors=max(n_candidates * 4, 24),
        )
        deduped: list[str] = []
        seen = {
            _normalize_expression(item)
            for item in [*(previous_expressions or []), *(extra_excludes or [])]
        }
        for item in pool:
            expression = str(item.get("expression") or "").strip()
            if not expression:
                continue
            key = _normalize_expression(expression)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(expression)
            if len(deduped) >= n_candidates:
                break
        return deduped

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

        usable_base = base_factor_codes[:20]
        continuation_lines: list[str] = []
        if continuation_context:
            continuation_lines.extend(
                [
                    f"上一轮主要问题：{continuation_context.get('primary_problem') or '暂无'}",
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
        attempted_expressions: set[str] = {
            _normalize_expression(item)
            for item in (previous_expressions or [])
            if _normalize_expression(item)
        }
        pending_expressions: list[str] = []

        def _extend_candidate_pool(target_count: int) -> None:
            if target_count <= 0:
                return

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

            if len(supported) < target_count:
                fallback_candidates = self._generate_fallback_candidate_expressions(
                    base_factor_codes=base_factor_codes,
                    n_candidates=max(target_count * 2, target_count),
                    previous_expressions=seed_expressions,
                    extra_excludes=supported,
                )
                supported.extend(
                    self._filter_supported_expressions(
                        fallback_candidates,
                        sample_frames=sample_frames,
                        limit=target_count - len(supported),
                    )
                )

            for expression in supported:
                key = _normalize_expression(expression)
                if not key or key in attempted_expressions:
                    continue
                if any(_normalize_expression(item) == key for item in pending_expressions):
                    continue
                pending_expressions.append(expression)

        _extend_candidate_pool(requested_candidates)

        if not pending_expressions:
            raise ValueError("未生成可执行候选表达式")

        evaluations: list[FactorEvaluationResult] = []
        attempt_count = 0
        max_attempts = max(len(pending_expressions), requested_candidates * 4)
        while pending_expressions and len(evaluations) < requested_candidates and attempt_count < max_attempts:
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
                remaining_needed = requested_candidates - len(evaluations)
                if not pending_expressions and attempt_count < max_attempts:
                    _extend_candidate_pool(min(remaining_needed, max_attempts - attempt_count))
                continue
            evaluations.append(evaluation)
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
            if not pending_expressions and len(evaluations) < requested_candidates and attempt_count < max_attempts:
                _extend_candidate_pool(min(requested_candidates - len(evaluations), max_attempts - attempt_count))

        if not evaluations:
            raise ValueError("候选表达式评估失败，未产出有效结果")

        evaluations.sort(key=lambda item: item.score, reverse=True)
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
        rounds: list[dict[str, Any]] = []
        retained_factors: list[dict[str, Any]] = []
        aggregate_best: list[float] = []
        aggregate_avg: list[float] = []
        previous_expressions: list[str] = []
        current_base_factors = list(base_factors)
        current_prompt = prompt
        best_round_result: dict[str, Any] | None = None
        best_round_task_id: str | None = None

        for round_index in range(1, max(exploration_rounds, 1) + 1):
            continuation_context = None
            continuation_hypothesis = None
            previous_round = rounds[-1] if rounds else None
            previous_base_factors = list(current_base_factors)
            if previous_round:
                continuation_context = {
                    "primary_problem": previous_round.get("continuation_feedback", {}).get("reason")
                    or previous_round.get("final_round_evaluation", {}).get("primary_problem"),
                    "recommended_goal": direction or previous_round.get("final_round_evaluation", {}).get("recommended_goal"),
                    "suggested_actions": previous_round.get("final_round_evaluation", {}).get("suggested_actions", []),
                    "metric_snapshot": previous_round.get("final_round_evaluation", {}).get("metric_snapshot", {}),
                }
                continuation_hypothesis = {
                    "hypothesis": f"第 {round_index} 轮继续围绕 {continuation_context.get('recommended_goal') or '综合优化'} 调整基础因子组合。",
                    "reason": continuation_context.get("primary_problem") or "上一轮仍存在可提升空间。",
                    "target_goal": continuation_context.get("recommended_goal") or "提升综合分数",
                    "primary_problem": continuation_context.get("primary_problem"),
                    "current_base_factors": list(previous_base_factors),
                    "candidate_factors": [],
                    "factor_update_mode": factor_update_mode,
                }

            current_round_candidates: list[dict[str, Any]] = []

            def _build_in_progress_round_snapshot() -> dict[str, Any]:
                current_scores = [float(item.get("score", 0.0) or 0.0) for item in current_round_candidates]
                return {
                    "round_index": round_index,
                    "task_id": f"campaign-round-{round_index}-in-progress",
                    "best_score": max(current_scores) if current_scores else 0.0,
                    "avg_score": round(float(np.mean(current_scores)), 4) if current_scores else 0.0,
                    "input_base_factors": list(current_base_factors),
                    "previous_base_factors": previous_round.get("input_base_factors", []) if previous_round else [],
                    "factor_changes": self._build_factor_changes(
                        previous_base_factors=previous_round.get("input_base_factors", []) if previous_round else [],
                        current_base_factors=current_base_factors,
                    ),
                    "factor_update_mode": "initial" if round_index == 1 else factor_update_mode,
                    "selected_factors": list(current_base_factors),
                    "continuation_hypothesis": continuation_hypothesis,
                    "continuation_feedback": None,
                    "retained_count": 0,
                    "retained_factors": [],
                    "all_factors": list(current_round_candidates),
                    "final_round_evaluation": continuation_context,
                }

            def _round_progress(done_count: int, total_count: int, candidate: dict[str, Any]) -> None:
                current_round_candidates.append(candidate)
                current_scores = [float(item.get("score", 0.0) or 0.0) for item in current_round_candidates]
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
                            "current_generation": done_count,
                            "total_generations": total_count,
                        }
                    )

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
            previous_expressions.extend([factor.get("expression", "") for factor in round_result.get("factors", [])])

            round_task_id = f"campaign-round-{round_index}-{datetime.now().strftime('%H%M%S')}"
            round_result["task_id"] = round_task_id
            selected = self.filter_retained_factors(round_result.get("factors", []), retention_filter)
            if not selected:
                selected = round_result.get("factors", [])[:1]

            factor_changes = self._build_factor_changes(
                previous_base_factors=previous_round.get("input_base_factors", []) if previous_round else [],
                current_base_factors=current_base_factors,
            )
            continuation_feedback = self._build_continuation_feedback(
                previous_best_score=previous_round.get("best_score") if previous_round else None,
                current_best_score=round_result.get("best_score", 0.0),
                retention_count=len(selected),
                direction=direction or "score",
            )

            round_summary = {
                "round_index": round_index,
                "task_id": round_task_id,
                "best_score": round_result.get("best_score", 0.0),
                "avg_score": round_result.get("avg_score", 0.0),
                "input_base_factors": list(current_base_factors),
                "previous_base_factors": previous_round.get("input_base_factors", []) if previous_round else [],
                "factor_changes": factor_changes,
                "factor_update_mode": "initial" if round_index == 1 else factor_update_mode,
                "selected_factors": list(current_base_factors),
                "selection_rationale": self._build_selection_rationale(
                    current_base_factors=current_base_factors,
                    selected=selected,
                    round_evaluation=round_result.get("round_evaluation"),
                    factor_update_mode=factor_update_mode,
                    round_index=round_index,
                ),
                "per_factor_reason": self._build_per_factor_reason(
                    current_base_factors=current_base_factors,
                    round_evaluation=round_result.get("round_evaluation"),
                ),
                "continuation_hypothesis": continuation_hypothesis,
                "continuation_feedback": continuation_feedback,
                "retained_count": len(selected),
                "retained_factors": selected,
                "all_factors": round_result.get("factors", []),
                "final_round_evaluation": round_result.get("round_evaluation"),
            }
            rounds.append(round_summary)

            aggregate_best.append(round_result.get("best_score", 0.0))
            aggregate_avg.append(round_result.get("avg_score", 0.0))
            retained_factors = selected

            if best_round_result is None:
                best_round_result = round_result
                best_round_task_id = round_task_id
            else:
                if parent_selection_strategy == "latest_round":
                    best_round_result = round_result
                    best_round_task_id = round_task_id
                elif round_result.get("best_score", 0.0) >= best_round_result.get("best_score", 0.0):
                    best_round_result = round_result
                    best_round_task_id = round_task_id

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

            next_factor_context = self.build_continuation_context(
                result=round_result,
                request_payload={
                    "base_factors": current_base_factors,
                    "direction": direction,
                },
                prompt=prompt,
                factor_update_mode=factor_update_mode,
                additional_factor_count=additional_factor_count_per_round,
            )
            selection = self.select_factors(
                prompt=f"{prompt} {direction or ''}".strip(),
                max_factor_count=additional_factor_count_per_round if factor_update_mode == "append" else max(
                    len(current_base_factors), additional_factor_count_per_round
                ),
                candidate_limit=80,
                selection_mode="auto",
                extra_context=next_factor_context["summary_text"],
                exclude_factors=current_base_factors if factor_update_mode == "append" else [],
            )
            candidate_factors = selection.get("selected_factors", [])
            if round_summary.get("continuation_hypothesis") is not None:
                round_summary["continuation_hypothesis"]["candidate_factors"] = list(candidate_factors)

            if factor_update_mode == "reselect":
                current_base_factors = candidate_factors or current_base_factors
            else:
                current_base_factors = _dedupe_preserve_order(current_base_factors + candidate_factors)
            current_prompt = prompt

        fitness_history = {
            "best": [round(max(aggregate_best[: idx + 1]), 4) for idx in range(len(aggregate_best))],
            "average": [round(float(np.mean(aggregate_avg[: idx + 1])), 4) for idx in range(len(aggregate_avg))],
        }

        return {
            "rounds": rounds,
            "retained_factors": retained_factors,
            "final_round_task_id": best_round_task_id,
            "final_round_result": best_round_result,
            "best_score": max(aggregate_best) if aggregate_best else 0.0,
            "avg_score": round(float(np.mean(aggregate_avg)), 4) if aggregate_avg else 0.0,
            "fitness_history": fitness_history,
            "selection_mode": retention_filter.get("match_mode", "all"),
            "retention_filter": retention_filter,
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
        del neutralize_industry, neutralize_cap
        fallback_reason: str | None = None
        try:
            panel_df = self._quantgpt_engine.build_panel_data(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                expression=expression,
                stock_data_loader=self.data_service.get_stock_data,
            )
            if panel_df.empty:
                fallback_reason = f"QuantGPT 未构建出有效 panel：{expression}"
                logger.info("QuantGPT 执行器未构建出有效 panel，回退到原有评估链：%s", expression)
            else:
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
                    return panel_result

                fallback_reason = f"QuantGPT 面板评估未产出有效结果：{expression}"
                logger.info("QuantGPT 面板评估未产出有效结果，回退到原有评估链：%s", expression)
        except Exception as exc:
            logger.warning("QuantGPT 执行器评估失败，回退到原有评估链 %s：%s", expression, exc)
            fallback_reason = f"QuantGPT 执行失败，已回退到 FactorHub 原生评估链：{expression}"

        fallback_result = self._factor_evaluation_service.evaluate_factor_expression(
            expression=expression,
            prompt=prompt,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            n_groups=n_groups,
            holding_period=holding_period,
            direction=direction,
            stock_data_loader=self.data_service.get_stock_data,
            expression_executor=lambda df, expr: factor_service.calculator.calculate(df.copy(), expr),
            benchmark_loader=self._load_benchmark_returns,
            report_writer=self._write_candidate_report,
            engine_type="factorhub",
            dialect="factorhub_native",
            canonical_expression=None,
            canonical_ast=None,
        )
        if fallback_result is None:
            return None
        fallback_result.diagnostics.insert(
            0,
            {
                "type": "warning",
                "label": "执行器回退",
                "text": fallback_reason or f"QuantGPT 执行失败，已回退到 FactorHub 原生评估链：{expression}",
            },
        )
        return fallback_result

    def _build_round_evaluation(
        self,
        *,
        prompt: str,
        base_factors: list[str],
        best_evaluation: FactorEvaluationResult,
        direction: str,
    ) -> dict[str, Any]:
        metric_snapshot = {
            "score": best_evaluation.score,
            "report_sharpe": best_evaluation.report_metrics.get("sharpe"),
            "report_max_drawdown": best_evaluation.report_metrics.get("max_drawdown"),
            "ls_sharpe": best_evaluation.backtest_summary.get("long_short_sharpe"),
            "ls_return": best_evaluation.backtest_summary.get("long_short_annual"),
            "rank_ic": best_evaluation.backtest_summary.get("rank_ic_mean"),
            "turnover": best_evaluation.backtest_summary.get("turnover"),
            "wq_fitness": best_evaluation.backtest_summary.get("wq_fitness"),
        }

        weaknesses = list(best_evaluation.interpretation.get("weaknesses") or [])
        primary_problem = weaknesses[0] if weaknesses else "暂无"
        suggested_actions = list(best_evaluation.interpretation.get("next_steps") or [])
        return {
            "prompt": prompt,
            "base_factors": list(base_factors),
            "primary_problem": primary_problem,
            "recommended_goal": direction,
            "suggested_actions": suggested_actions,
            "metric_snapshot": metric_snapshot,
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

    def _build_selection_rationale(
        self,
        *,
        current_base_factors: list[str],
        selected: list[dict[str, Any]],
        round_evaluation: dict[str, Any] | None,
        factor_update_mode: str,
        round_index: int,
    ) -> str:
        reason = round_evaluation.get("primary_problem") if round_evaluation else "暂无"
        return (
            f"第 {round_index} 轮使用 {len(current_base_factors)} 个基础因子，"
            f"本轮保留 {len(selected)} 个候选；更新方式为 {factor_update_mode}。"
            f" 主要针对的问题：{reason}"
        )

    def _build_per_factor_reason(
        self,
        *,
        current_base_factors: list[str],
        round_evaluation: dict[str, Any] | None,
    ) -> dict[str, str]:
        reason = round_evaluation.get("recommended_goal") if round_evaluation else "提升综合分数"
        return {factor_name: f"围绕“{reason}”保留或补充该基础因子。" for factor_name in current_base_factors}

    def _build_continuation_feedback(
        self,
        *,
        previous_best_score: float | None,
        current_best_score: float,
        retention_count: int,
        direction: str,
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
        improved = score_delta >= 0
        return {
            "decision": improved or retention_count > 0,
            "accepted_as_best": improved,
            "observations": "本轮结果已纳入连续探索轨迹。",
            "hypothesis_evaluation": "本轮较上一轮有所提升。" if improved else "本轮未超过上一轮最佳，但仍保留可用候选。",
            "reason": "最佳分数提升。" if improved else "分数未超越上一轮，下一轮需调整基础因子结构。",
            "score_delta": score_delta,
            "parent_best_score": previous_best_score,
            "current_best_score": current_best_score,
            "next_hypothesis": f"继续针对 {direction} 做更强的因子重组。",
        }


auto_factor_mining_service = AutoFactorMiningService()
