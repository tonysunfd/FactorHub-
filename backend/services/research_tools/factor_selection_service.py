from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_factor_expression_key(expression: Any) -> str:
    return "".join(str(expression or "").lower().split())


OPTIMIZATION_DIRECTION_LABELS = {
    "score": "优化 Score",
    "ls_sharpe": "优化 L/S Sharpe",
    "ls_return": "优化 L/S Return",
    "wq_rating": "优化 WQ Rating",
    "wq_fitness": "优化 WQ Fitness",
    "wq_return": "优化 WQ Return",
    "report_sharpe": "优化 Report Sharpe",
}


class FactorSelectionService:
    def dedupe_factor_names(self, names: List[str]) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for name in names:
            normalized = str(name or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def normalize_wq_rating(self, value: Any) -> str:
        return str(value or "").strip().lower()

    def is_manual_genetic_candidate(self, factor: Dict[str, Any]) -> bool:
        scope_type = str(factor.get("scope_type") or "").strip().lower()
        origin_type = str(factor.get("origin_type") or "").strip().lower()
        code = str(factor.get("code") or "").strip()
        target_universe = str(factor.get("target_universe") or "").strip()
        target_stock_code = str(factor.get("target_stock_code") or "").strip()

        if not code:
            return False
        if scope_type == "universe":
            return False
        if target_universe:
            return False
        if origin_type == "auto_mining":
            return False
        if scope_type not in {"", "stock", "base"}:
            return False
        if target_stock_code:
            return False
        return True

    def build_factor_snapshot_summary(self, snapshot_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = snapshot_payload or {}
        report_metrics = payload.get("report_metrics") or payload.get("metrics") or {}
        backtest_summary = payload.get("backtest_summary") or {}
        wq_brain = payload.get("wq_brain") or {}
        scoring = payload.get("scoring") or {}
        interpretation = payload.get("interpretation") or {}
        return {
            "report_metrics": {
                "sharpe": report_metrics.get("sharpe"),
                "cagr": report_metrics.get("cagr"),
                "max_drawdown": report_metrics.get("max_drawdown"),
                "volatility": report_metrics.get("volatility"),
            },
            "backtest_summary": {
                "ic_mean": backtest_summary.get("ic_mean"),
                "rank_ic_mean": backtest_summary.get("rank_ic_mean"),
                "monotonicity_score": backtest_summary.get("monotonicity_score"),
                "turnover": backtest_summary.get("turnover"),
            },
            "wq_brain": {
                "fitness": wq_brain.get("fitness") or wq_brain.get("wq_fitness"),
                "rating": wq_brain.get("rating") or wq_brain.get("wq_rating"),
                "passes": wq_brain.get("pass_count") or wq_brain.get("wq_pass_count"),
            },
            "scoring": {
                "score": scoring.get("score"),
                "grade": scoring.get("grade"),
                "component_scores": scoring.get("component_scores") or {},
            },
            "interpretation": {
                "logic": interpretation.get("logic"),
                "source": interpretation.get("source"),
                "risk": interpretation.get("risk"),
            },
        }

    def is_reviewed_rdagent_candidate(self, factor: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        origin_type = str(factor.get("origin_type") or "").strip().lower()
        category = str(factor.get("category") or "").strip()
        if origin_type != "rdagent_mining" and category != "RDAgent 挖掘":
            return True

        metadata = factor.get("task_metadata") if isinstance(factor.get("task_metadata"), dict) else {}
        review_status = str(metadata.get("review_status") or payload.get("review_status") or "").strip().lower()
        if review_status != "confirmed":
            return False

        summary = self.build_factor_snapshot_summary(payload if isinstance(payload, dict) else {})
        scoring = summary.get("scoring") or {}
        report_metrics = summary.get("report_metrics") or {}
        backtest_summary = summary.get("backtest_summary") or {}
        score = float(scoring.get("score") or 0)
        rank_ic = abs(float(backtest_summary.get("rank_ic_mean") or backtest_summary.get("ic_mean") or 0))
        sharpe = float(report_metrics.get("sharpe") or 0)
        return score >= 35 or rank_ic >= 0.01 or sharpe >= 1.0

    def load_factor_candidates_for_llm(self, limit: int = 80, selection_mode: str = "auto") -> List[Dict[str, Any]]:
        from backend.services.factor_service import factor_service

        all_factors = factor_service.get_all_factors()
        candidates: List[Dict[str, Any]] = []
        seen_names: set[str] = set()
        seen_codes: set[str] = set()
        normalized_mode = str(selection_mode or "auto").strip().lower()

        for factor in all_factors:
            if not factor.get("is_active", True):
                continue
            name = (factor.get("name") or "").strip()
            code = (factor.get("code") or "").strip()
            code_key = normalize_factor_expression_key(code)
            if not name or not code:
                continue
            if name in seen_names or code_key in seen_codes:
                continue
            if normalized_mode == "manual_genetic" and not self.is_manual_genetic_candidate(factor):
                continue

            latest_snapshot = factor.get("latest_task_snapshot") or {}
            payload = latest_snapshot.get("payload") or factor.get("task_metadata") or {}
            if not self.is_reviewed_rdagent_candidate(factor, payload if isinstance(payload, dict) else {}):
                continue
            candidates.append({
                "name": name,
                "category": factor.get("category") or "未分类",
                "source": factor.get("source") or "user",
                "description": factor.get("description") or "",
                "code": code,
                "scope_type": factor.get("scope_type") or "",
                "origin_type": factor.get("origin_type") or "manual",
                "target_universe": factor.get("target_universe") or "",
                "target_stock_code": factor.get("target_stock_code") or "",
                "snapshot_summary": self.build_factor_snapshot_summary(payload if isinstance(payload, dict) else {}),
            })
            seen_names.add(name)
            seen_codes.add(code_key)
            if len(candidates) >= max(int(limit or 0), 1):
                break
        return candidates

    def infer_primary_problem_from_metrics(
        self,
        report_metrics: Dict[str, Any],
        backtest_summary: Dict[str, Any],
        score: float,
    ) -> str:
        report_sharpe = float(report_metrics.get("sharpe") or 0)
        report_cagr = float(report_metrics.get("cagr") or 0)
        report_max_drawdown = float(report_metrics.get("max_drawdown") or 0)
        report_volatility = float(report_metrics.get("volatility") or 0)
        ls_sharpe = float(backtest_summary.get("long_short_sharpe") or 0)
        ls_return = float(backtest_summary.get("long_short_annual") or 0)

        if abs(report_max_drawdown) >= 0.2:
            return "最大回撤偏大，收益回撤比不够理想，优先控制回撤。"
        if report_volatility >= 0.3:
            return "波动偏高，净值稳定性不足，需优先收敛波动。"
        if report_sharpe < 1.0 or ls_sharpe < 1.0:
            return "Sharpe 偏低，风险调整后收益不足，说明收益质量还不够稳定。"
        if report_cagr < 0.12 or ls_return < 0.12:
            return "收益偏弱，当前结构的超额收益释放不足。"
        if score < 70:
            return "综合评分偏低，当前表达式质量和可用性仍然不足。"
        return "当前结果已具备一定可用性，但缺少明确短板描述，建议优先检查 Sharpe、回撤与波动之间的平衡。"

    def build_round_evaluation(
        self,
        parent_task: Dict[str, Any],
        best_factor: Dict[str, Any],
        normalize_direction,
    ) -> Dict[str, Any]:
        parent_params = parent_task.get("params") or {}
        current_base_factors = self.dedupe_factor_names(parent_params.get("base_factors") or [])
        report_metrics = best_factor.get("report_metrics") or {}
        backtest_summary = best_factor.get("backtest_summary") or {}
        interpretation = best_factor.get("interpretation") or {}
        direction_key = normalize_direction(parent_params.get("direction"))
        direction_label = OPTIMIZATION_DIRECTION_LABELS.get(direction_key, "")

        weakness_hints: List[str] = []
        for key in ["weaknesses", "risks", "limitations", "next_steps", "improvement_ideas"]:
            value = interpretation.get(key)
            if isinstance(value, list):
                weakness_hints.extend([str(item).strip() for item in value if str(item or "").strip()])
            elif isinstance(value, str) and value.strip():
                weakness_hints.append(value.strip())

        for key in ["risk", "summary", "commentary", "explanation"]:
            value = interpretation.get(key)
            if isinstance(value, str) and value.strip():
                weakness_hints.append(value.strip())

        weakness_hints = [item for item in weakness_hints if item]
        primary_problem = weakness_hints[0] if weakness_hints else self.infer_primary_problem_from_metrics(
            report_metrics,
            backtest_summary,
            float(best_factor.get("score") or 0),
        )
        secondary_problem = weakness_hints[1] if len(weakness_hints) > 1 else ""

        if direction_label:
            recommended_goal = direction_label
        elif "综合评分" in primary_problem:
            recommended_goal = "优化 Score"
        elif "Sharpe" in primary_problem or "风险调整后收益" in primary_problem:
            recommended_goal = "优化 L/S Sharpe"
        elif "收益" in primary_problem:
            recommended_goal = "优化 L/S Return"
        elif "回撤" in primary_problem:
            recommended_goal = "优化 Score"
        elif "波动" in primary_problem:
            recommended_goal = "优化 Score"
        elif "WQ" in primary_problem or "Fitness" in primary_problem:
            recommended_goal = "优化 WQ Fitness"
        else:
            recommended_goal = "优化 Score"

        suggested_actions: List[str] = [
            f"下一轮只围绕“{recommended_goal}”做小步优化，不要同时追求多个目标。",
            "优先保留当前有效结构，只调整与主要短板直接相关的基础因子或表达式局部。",
        ]
        if current_base_factors:
            suggested_actions.append("先评估当前基础因子是否存在语义重复或风格过于单一，再决定补充或替换。")
        if secondary_problem:
            suggested_actions.append(f"次要问题可暂缓，先不要同时处理：{secondary_problem}")

        metric_snapshot = {
            "score": float(best_factor.get("score") or 0),
            "grade": best_factor.get("grade"),
            "report_sharpe": report_metrics.get("sharpe"),
            "report_cagr": report_metrics.get("cagr"),
            "report_max_drawdown": report_metrics.get("max_drawdown"),
            "report_volatility": report_metrics.get("volatility"),
            "ls_sharpe": backtest_summary.get("long_short_sharpe"),
            "ls_return": backtest_summary.get("long_short_annual"),
        }

        return {
            "base_factors": current_base_factors,
            "primary_problem": primary_problem,
            "secondary_problem": secondary_problem,
            "recommended_goal": recommended_goal,
            "suggested_actions": suggested_actions[:3],
            "metric_snapshot": metric_snapshot,
        }

    def build_factor_improvement_context(
        self,
        parent_task: Dict[str, Any],
        best_factor: Dict[str, Any],
        normalize_direction,
    ) -> str:
        round_evaluation = (
            (best_factor.get("task_details") or {}).get("round_evaluation")
            or parent_task.get("round_evaluation")
            or self.build_round_evaluation(parent_task, best_factor, normalize_direction)
        )
        base_factors = round_evaluation.get("base_factors") or []
        primary_problem = round_evaluation.get("primary_problem") or "未显式给出，请基于指标自行判断"
        metric_snapshot = round_evaluation.get("metric_snapshot") or {}
        recommended_goal = round_evaluation.get("recommended_goal") or "未指定"
        suggested_actions = "；".join(round_evaluation.get("suggested_actions") or ["无"])
        return (
            "上一轮简要诊断：\n"
            f"- 基础因子: {base_factors}\n"
            f"- 主要短板: {primary_problem}\n"
            f"- 建议优化方向: {recommended_goal}\n"
            f"- 建议动作: {suggested_actions}\n"
            f"- 关键指标: {metric_snapshot}\n"
        )

    def build_llm_factor_selector_prompt(self, request, candidates: List[Dict[str, Any]]) -> str:
        candidate_blocks: List[str] = []
        for idx, item in enumerate(candidates, start=1):
            compact = {
                "name": item.get("name"),
                "category": item.get("category"),
                "source": item.get("source"),
                "description": item.get("description"),
                "code": item.get("code"),
                "scope_type": item.get("scope_type"),
                "origin_type": item.get("origin_type"),
                "target_universe": item.get("target_universe"),
                "target_stock_code": item.get("target_stock_code"),
                "snapshot_summary": item.get("snapshot_summary") or {},
            }
            candidate_blocks.append(f"候选因子{idx}: {compact}")

        user_goal = (request.prompt or "").strip()
        direction = (request.direction or "").strip()
        direction_text = direction if direction else "未指定，需结合提示词自主判断优化方向"
        selection_mode = str(request.selection_mode or "auto").strip().lower()
        if selection_mode == "manual_genetic":
            task_intro = "你是一位专业量化研究员，负责为手动遗传挖掘挑选一组最适合作为基础因子的 seed/base factors。"
            task_constraints = (
                "你的任务不是机械打分排序，而是根据研究目标、因子语义、互补性和可计算性，"
                "只选择适合在单股票 OHLCV 数据上直接计算的基础因子。"
            )
        else:
            task_intro = "你是一位专业量化研究员，负责为自动因子挖掘挑选一组最适合作为 seed/base factors 的基础因子。"
            task_constraints = "你的任务不是机械打分排序，而是根据研究目标、因子语义、互补性、历史研究摘要和可进化性，自主选择一组合适的基础因子。"

        return (
            f"{task_intro}\n"
            f"{task_constraints}\n\n"
            f"研究目标: {user_goal}\n"
            f"优化方向: {direction_text}\n"
            f"股票池: {request.universe}\n"
            f"基准: {request.benchmark}\n"
            f"时间范围: {request.start_date} ~ {request.end_date}\n"
            f"最多选择数量: {max(int(request.max_factor_count or 12), 1)}\n\n"
            "选择要求：\n"
            "1. 优先选择与研究目标/优化方向匹配的因子；\n"
            "2. 尽量保证因子之间互补，而不是语义重复；\n"
            "3. 可以参考历史指标摘要，但不要被单一数值机械绑定；\n"
            "4. 请在保证质量和互补性的前提下，自主决定最终数量，但不能超过最多选择数量；\n"
            "5. 如果某些因子明显不适合当前目标，可以不选；\n"
            "6. 只允许从候选列表中选择，不要发明新因子名；\n"
            "7. 如果是手动遗传挖掘场景，优先选择适合单股票量价时间序列直接计算的基础因子，避免选择依赖股票池横截面或自动挖掘结果表达式的因子。\n\n"
            "请只输出 JSON，格式如下：\n"
            "{\n"
            "  \"selected_factors\": [\"因子名1\", \"因子名2\"],\n"
            "  \"selection_rationale\": \"整体选择理由\",\n"
            "  \"per_factor_reason\": {\"因子名1\": \"原因\", \"因子名2\": \"原因\"}\n"
            "}\n\n"
            "候选因子列表：\n"
            + "\n".join(candidate_blocks)
        )


factor_selection_service = FactorSelectionService()


def dedupe_factor_names(names: List[str]) -> List[str]:
    return factor_selection_service.dedupe_factor_names(names)


def normalize_wq_rating(value: Any) -> str:
    return factor_selection_service.normalize_wq_rating(value)


def is_manual_genetic_candidate(factor: Dict[str, Any]) -> bool:
    return factor_selection_service.is_manual_genetic_candidate(factor)


def build_factor_snapshot_summary(snapshot_payload: Dict[str, Any]) -> Dict[str, Any]:
    return factor_selection_service.build_factor_snapshot_summary(snapshot_payload)


def load_factor_candidates_for_llm(limit: int = 80, selection_mode: str = "auto") -> List[Dict[str, Any]]:
    return factor_selection_service.load_factor_candidates_for_llm(limit=limit, selection_mode=selection_mode)


def infer_primary_problem_from_metrics(
    report_metrics: Dict[str, Any],
    backtest_summary: Dict[str, Any],
    score: float,
) -> str:
    return factor_selection_service.infer_primary_problem_from_metrics(report_metrics, backtest_summary, score)


def build_round_evaluation(
    parent_task: Dict[str, Any],
    best_factor: Dict[str, Any],
    normalize_direction,
) -> Dict[str, Any]:
    return factor_selection_service.build_round_evaluation(parent_task, best_factor, normalize_direction)


def build_factor_improvement_context(
    parent_task: Dict[str, Any],
    best_factor: Dict[str, Any],
    normalize_direction,
) -> str:
    return factor_selection_service.build_factor_improvement_context(parent_task, best_factor, normalize_direction)


def build_llm_factor_selector_prompt(request, candidates: List[Dict[str, Any]]) -> str:
    return factor_selection_service.build_llm_factor_selector_prompt(request, candidates)
