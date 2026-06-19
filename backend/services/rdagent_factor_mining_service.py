from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from backend.services.auto_factor_mining_service import AutoFactorMiningService
from backend.services.expression_schema import FactorEvaluationResult
from backend.services.factor_evaluation_service import FactorEvaluationService
from backend.services.llm_config_service import llm_config_service
from backend.services.rdagent_local_pipeline import (
    FactorHubRDAgentCoder,
    FactorHubRDAgentFeedback,
    FactorHubRDAgentRunner,
    build_factorhub_rdagent_pipeline_metadata,
)
from backend.services.rdagent_native_code_executor import RDAgentNativeCodeExecutor
from backend.services.rdagent_runtime import get_rdagent_runtime_status, probe_rdagent_module_import
from backend.services.rdagent_upstream_proposal_adapter import RDAgentUpstreamProposalAdapter
from backend.services.research_tools.expression_adapter import ExpressionAdapter
from backend.services.research_tools.rdagent_expression_contract import (
    RDAgentExpressionFormatError,
    normalize_rdagent_expression_for_parser,
    rdagent_expression_contract_text,
    validate_rdagent_expression_contract,
)

logger = logging.getLogger(__name__)

MAX_RDAGENT_ITERATIONS = 8

_RDAGENT_HYPOTHESIS_SYSTEM_PROMPT = """你是一个量化研究员，负责为因子挖掘提出下一轮研究假设。

你必须遵守以下规则：
1. 只返回 JSON 对象，不要输出 Markdown 或额外解释。
2. 输出字段必须包含 statement、reason、research_direction、expected_signal。
3. statement 必须是一条可验证的研究假设，聚焦单一优化方向。
4. 必须显式结合上一轮反馈、已有基础因子和候选字段，避免重复。
"""

_RDAGENT_EXPERIMENT_SYSTEM_PROMPT = """你是一个量化研究规划助手，负责把研究假设转成可执行实验。

你必须遵守以下规则：
1. 只返回 JSON 对象，不要输出 Markdown 或额外解释。
2. 输出字段必须包含 hypothesis_summary、factor_formulations、base_factors、evaluation_focus。
3. factor_formulations 必须是 1 到 N 条单行 FactorHub 表达式。
4. 所有表达式都必须遵守给定的 FactorHub 表达式契约。
"""


class RDAgentTaskCancelled(RuntimeError):
    """Raised when the caller cancels a running RDAgent task."""


@dataclass
class RDAgentMiningConfig:
    task_id: str
    objective: str
    max_iterations: int = 1
    candidates_per_iteration: int = 1
    base_factors: list[str] = field(default_factory=list)
    candidate_universe: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    universe: str = "all"
    benchmark: str = "000300.SH"
    n_groups: int = 5
    holding_period: int = 5
    direction: str | None = None
    neutralize_industry: bool = True
    neutralize_cap: bool = True
    acceptance_policy: Any = None
    continuation_of: str | None = None
    previous_feedback_id: str | None = None
    previous_expressions: list[str] = field(default_factory=list)
    previous_sota_expressions: list[str] = field(default_factory=list)
    execution_mode: str = "native_code"
    cancel_check: Callable[[], None] | None = None


class RDAgentFactorMiningService:
    """独立的 RDAgent 因子挖掘执行器。"""

    def __init__(
        self,
        *,
        auto_mining_service: AutoFactorMiningService | None = None,
        llm_client_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._auto_mining_service = auto_mining_service or AutoFactorMiningService()
        self._llm_client_factory = llm_client_factory
        self._native_code_executor = RDAgentNativeCodeExecutor()
        self._upstream_proposal_adapter = RDAgentUpstreamProposalAdapter()
        self._factor_evaluation_service = FactorEvaluationService()
        self._local_coder = FactorHubRDAgentCoder(code_experiment_fn=self._code_experiment)
        self._local_runner = FactorHubRDAgentRunner(run_experiment_fn=self._run_experiment)
        self._local_feedback = FactorHubRDAgentFeedback(generate_feedback_fn=self._generate_feedback)

    def run(
        self,
        *,
        task_id: str,
        config: RDAgentMiningConfig,
        on_progress: Callable[[int, str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        retained_factors: list[dict[str, Any]] = []
        watchlist_factors: list[dict[str, Any]] = []
        sota_candidates: list[dict[str, Any]] = [
            {
                "name": f"Historical_SOTA_{index + 1}",
                "expression": expression,
                "score": 0.0,
            }
            for index, expression in enumerate(_dedupe_strings(list(config.previous_sota_expressions or [])))
        ]
        fitness_history: dict[str, list[float]] = {"best": [], "average": []}
        known_expressions = list(config.previous_expressions or [])

        total_iterations = max(1, min(int(config.max_iterations or 1), MAX_RDAGENT_ITERATIONS))
        total_stages = total_iterations * 5
        stage_count = 0
        current_base_factors = list(config.base_factors or [])

        def emit(stage: str, iteration: int, payload: dict[str, Any]) -> None:
            nonlocal stage_count
            stage_count += 1
            progress = min(int(stage_count / max(total_stages, 1) * 100), 99)
            if on_progress:
                on_progress(progress, stage, {"iteration": iteration, "payload": payload})

        for iteration in range(1, total_iterations + 1):
            self._raise_if_cancelled(config)

            hypothesis = self._propose_hypothesis(
                config=config,
                rounds=rounds,
                iteration=iteration,
                current_base_factors=current_base_factors,
            )
            emit("rdagent_hypothesis", iteration, hypothesis)

            self._raise_if_cancelled(config)
            experiment = self._hypothesis_to_experiment(
                config=config,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
                current_base_factors=current_base_factors,
                known_expressions=known_expressions,
            )
            emit("rdagent_experiment", iteration, experiment)

            self._raise_if_cancelled(config)
            coded_experiment = self._local_coder.develop(
                config=config,
                experiment=experiment,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
            )
            emit("rdagent_coding", iteration, coded_experiment)

            self._raise_if_cancelled(config)
            run_result = self._local_runner.develop(
                config=config,
                coded_experiment=coded_experiment,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
                sota_candidates=sota_candidates,
            )
            candidates = list(run_result.get("candidates") or [])
            evaluation = {
                "iteration": iteration,
                "metrics": run_result.get("metrics") or {},
                "report_ref": run_result.get("report_ref"),
                "backtest_engine": run_result.get("backtest_engine") or "factorhub",
                "best_score": float((run_result.get("metrics") or {}).get("score") or 0.0),
                "avg_score": _average_score(candidates),
                "report_metrics": (run_result.get("best_candidate") or {}).get("report_metrics") or {},
                "backtest_summary": (run_result.get("best_candidate") or {}).get("backtest_summary") or {},
            }
            emit(
                "rdagent_running",
                iteration,
                {
                    "candidates": candidates,
                    "evaluation": evaluation,
                    "best_score": evaluation["best_score"],
                    "avg_score": evaluation["avg_score"],
                },
            )

            self._raise_if_cancelled(config)
            feedback = self._local_feedback.generate_feedback(
                config=config,
                hypothesis=hypothesis,
                experiment=experiment,
                run_result=run_result,
                rounds=rounds,
                iteration=iteration,
            )
            emit(
                "rdagent_feedback",
                iteration,
                {
                    "hypothesis": hypothesis,
                    "feedback": feedback,
                    "candidates": candidates,
                    "best_score": evaluation["best_score"],
                    "avg_score": evaluation["avg_score"],
                },
            )

            continuation_selection = None
            next_base_factors = list(experiment.get("base_factors") or current_base_factors)
            if iteration < total_iterations:
                continuation_selection = self._select_next_base_factors(
                    config=config,
                    current_base_factors=current_base_factors,
                    hypothesis=hypothesis,
                    run_result=run_result,
                )
                selected_for_next_round = list((continuation_selection or {}).get("selected_factors") or [])
                if selected_for_next_round:
                    next_base_factors = _dedupe_strings(list(next_base_factors) + selected_for_next_round)

            updated_sota_candidates = self._merge_sota_candidates(sota_candidates, candidates)

            round_item = {
                "round_index": iteration,
                "hypothesis": hypothesis,
                "experiment": experiment,
                "coded_experiment": coded_experiment,
                "candidates": candidates,
                "all_factors": candidates,
                "evaluation": evaluation,
                "feedback": feedback,
                "continuation_selection": continuation_selection,
                "next_base_factors": list(next_base_factors),
                "sota_candidates": list(updated_sota_candidates),
                "pipeline": build_factorhub_rdagent_pipeline_metadata(
                    execution_mode=str(config.execution_mode or "native_code").strip().lower(),
                ),
            }
            rounds.append(round_item)

            retained_factors.extend([item for item in candidates if item.get("status") == "accepted"])
            watchlist_factors.extend([item for item in candidates if item.get("status") == "watchlist"])
            sota_candidates = updated_sota_candidates
            fitness_history["best"].append(evaluation["best_score"])
            fitness_history["average"].append(evaluation["avg_score"])

            for candidate in candidates:
                expression = str(candidate.get("expression") or "").strip()
                if expression:
                    known_expressions.append(expression)

            current_base_factors = _dedupe_strings(next_base_factors)

        final_round = rounds[-1] if rounds else {}
        top_factors = self._collect_top_factors(rounds, limit=5)
        final_round_result = {
            **(final_round.get("evaluation") or {}),
            "factors": top_factors or final_round.get("candidates") or [],
        }
        return {
            "task_id": task_id,
            "objective": config.objective,
            "rounds": self._sanitize_payload(rounds),
            "retained_factors": self._sanitize_payload(retained_factors),
            "watchlist_factors": self._sanitize_payload(watchlist_factors),
            "top_factors": self._sanitize_payload(top_factors),
            "sota_candidates": self._sanitize_payload(sota_candidates),
            "fitness_history": fitness_history,
            "final_round_result": self._sanitize_payload(final_round_result),
            "continue_mining_request": self._build_continue_request(
                config=config,
                final_round=final_round,
                known_expressions=known_expressions,
            ),
        }

    def _select_next_base_factors(
        self,
        *,
        config: RDAgentMiningConfig,
        current_base_factors: list[str],
        hypothesis: dict[str, Any],
        run_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        round_evaluation = {
            "base_factors": list(current_base_factors),
            "primary_problem": hypothesis.get("reason"),
            "recommended_goal": hypothesis.get("research_direction") or config.direction or "score",
            "suggested_actions": [hypothesis.get("expected_signal")] if hypothesis.get("expected_signal") else [],
            "metric_snapshot": run_result.get("metrics") or {},
        }
        parent_result = {
            "factors": list(run_result.get("candidates") or []),
            "best_score": (run_result.get("metrics") or {}).get("score"),
            "avg_score": (run_result.get("metrics") or {}).get("avg_score"),
            "round_evaluation": round_evaluation,
        }
        parent_request = {
            "base_factors": list(current_base_factors),
            "direction": hypothesis.get("research_direction") or config.direction or "score",
            "start_date": config.start_date,
            "end_date": config.end_date,
            "universe": config.universe,
            "benchmark": config.benchmark,
        }
        try:
            return self._auto_mining_service.select_continue_factors(
                parent_result=parent_result,
                parent_request=parent_request,
                prompt=config.objective,
                direction=hypothesis.get("research_direction") or config.direction,
                factor_update_mode="append",
                max_factor_count=max(int(config.candidates_per_iteration or 1), 1),
                candidate_limit=40,
                current_base_factors=list(current_base_factors),
            )
        except Exception as exc:
            logger.warning("RDAgent 选择下一轮基础因子失败，保留当前基础因子：%s", exc)
            return None

    def _propose_hypothesis(
        self,
        *,
        config: RDAgentMiningConfig,
        rounds: list[dict[str, Any]],
        iteration: int,
        current_base_factors: list[str],
    ) -> dict[str, Any]:
        execution_mode = str(config.execution_mode or "native_code").strip().lower()
        if execution_mode == "upstream_rdagent":
            proposal = self._generate_upstream_round_plan(
                config=config,
                rounds=rounds,
                iteration=iteration,
                current_base_factors=current_base_factors,
            )
            upstream_hypothesis = proposal.get("hypothesis") or {}
            return {
                "statement": str(upstream_hypothesis.get("statement") or f"第 {iteration} 轮围绕 {config.objective} 挖掘新因子").strip(),
                "reason": str(upstream_hypothesis.get("reason") or "由 upstream RD-Agent proposal 生成。").strip(),
                "research_direction": str(config.direction or "score").strip(),
                "expected_signal": str(upstream_hypothesis.get("concise_observation") or "提升综合分数与稳定性").strip(),
                "base_factors": list(current_base_factors),
                "candidate_universe": list(config.candidate_universe),
                "previous_feedback_id": config.previous_feedback_id,
                "upstream_proposal": proposal,
            }
        llm_response = self._call_llm_json(
            system_prompt=_RDAGENT_HYPOTHESIS_SYSTEM_PROMPT,
            user_prompt=self._build_hypothesis_prompt(
                config=config,
                rounds=rounds,
                iteration=iteration,
                current_base_factors=current_base_factors,
            ),
        )
        fallback_direction = str(config.direction or "score")
        return {
            "statement": str(llm_response.get("statement") or f"第 {iteration} 轮围绕 {fallback_direction} 优化 {config.objective}").strip(),
            "reason": str(llm_response.get("reason") or "结合上一轮反馈和当前基础因子继续优化。").strip(),
            "research_direction": str(llm_response.get("research_direction") or fallback_direction).strip(),
            "expected_signal": str(llm_response.get("expected_signal") or "提升综合分数与稳定性").strip(),
            "base_factors": list(current_base_factors),
            "candidate_universe": list(config.candidate_universe),
            "previous_feedback_id": config.previous_feedback_id,
        }

    def _hypothesis_to_experiment(
        self,
        *,
        config: RDAgentMiningConfig,
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
        current_base_factors: list[str],
        known_expressions: list[str],
    ) -> dict[str, Any]:
        execution_mode = str(config.execution_mode or "native_code").strip().lower()
        if execution_mode == "upstream_rdagent":
            proposal = hypothesis.get("upstream_proposal") or self._generate_upstream_round_plan(
                config=config,
                rounds=rounds,
                iteration=iteration,
                current_base_factors=current_base_factors,
            )
            tasks = list(proposal.get("tasks") or [])
            factor_formulations = [
                str(task.get("formulation") or task.get("factor_name") or "").strip()
                for task in tasks
                if str(task.get("formulation") or task.get("factor_name") or "").strip()
            ]
            base_factors = list(current_base_factors)
            if not base_factors:
                selection = self._auto_mining_service.select_factors(
                    prompt=f"{config.objective} {config.direction or ''}".strip(),
                    max_factor_count=max(int(config.candidates_per_iteration or 1), 1),
                    candidate_limit=40,
                    selection_mode="auto",
                )
                base_factors = selection.get("selected_factors", [])
            return {
                "round_index": iteration,
                "candidate_limit": max(int(config.candidates_per_iteration or 1), 1),
                "base_factors": base_factors,
                "candidate_universe": list(config.candidate_universe),
                "execution_mode": config.execution_mode,
                "sota_expressions": [
                    str(item.get("expression") or "").strip()
                    for item in (rounds[-1].get("sota_candidates") or [])
                ] if rounds else [],
                "hypothesis_summary": hypothesis.get("statement"),
                "evaluation_focus": hypothesis.get("expected_signal"),
                "factor_formulations": factor_formulations,
                "upstream_tasks": tasks,
            }
        factor_formulations = self._generate_factor_formulations(
            config=config,
            hypothesis=hypothesis,
            rounds=rounds,
            iteration=iteration,
            current_base_factors=current_base_factors,
            known_expressions=known_expressions,
        )
        base_factors = list(current_base_factors)
        if not base_factors:
            try:
                selection = self._auto_mining_service.select_factors(
                    prompt=f"{config.objective} {hypothesis.get('research_direction') or ''}".strip(),
                    max_factor_count=max(int(config.candidates_per_iteration or 1), 1),
                    candidate_limit=40,
                    selection_mode="auto",
                )
                base_factors = selection.get("selected_factors", [])
            except Exception as exc:
                logger.warning("RDAgent 基础因子筛选失败，回退为空基础因子继续执行：%s", exc)
                base_factors = []
        return {
            "round_index": iteration,
            "candidate_limit": max(int(config.candidates_per_iteration or 1), 1),
            "base_factors": base_factors,
            "candidate_universe": list(config.candidate_universe),
            "execution_mode": config.execution_mode,
            "sota_expressions": [
                str(item.get("expression") or "").strip()
                for item in (rounds[-1].get("sota_candidates") or [])
            ] if rounds else [],
            "hypothesis_summary": hypothesis.get("statement"),
            "evaluation_focus": hypothesis.get("expected_signal"),
            "factor_formulations": factor_formulations,
        }

    def _code_experiment(
        self,
        *,
        config: RDAgentMiningConfig,
        experiment: dict[str, Any],
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        execution_mode = str(config.execution_mode or "native_code").strip().lower()
        if execution_mode == "native_code":
            return self._code_native_experiment(
                config=config,
                experiment=experiment,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
            )
        if execution_mode == "upstream_rdagent":
            runtime_status = get_rdagent_runtime_status()
            proposal_importable, proposal_import_error = probe_rdagent_module_import(
                "rdagent.scenarios.qlib.proposal.factor_proposal"
            )
            if not runtime_status.get("active_path") or not proposal_importable:
                raise ValueError(
                    "upstream_rdagent 执行器当前不可用：reference RD-Agent proposal 运行时未就绪。"
                    f" import_error={proposal_import_error or runtime_status.get('import_error') or 'unknown'}"
                )
            upstream_tasks = list(experiment.get("upstream_tasks") or [])
            coded_items = []
            for index, task in enumerate(upstream_tasks):
                factor_name = str(task.get("factor_name") or f"UpstreamFactor{index + 1}").strip()
                formulation = str(task.get("formulation") or factor_name).strip()
                converted = self._prepare_upstream_candidate(
                    formulation=formulation,
                    variables=dict(task.get("variables") or {}),
                )
                diagnostics = [
                    {
                        "type": "info",
                        "label": "upstream_rdagent",
                        "text": "候选由 reference RD-Agent proposal 生成，执行与评估仍使用 FactorHub 本地链路。",
                    }
                ]
                conversion_note = converted.get("conversion_note")
                if conversion_note:
                    diagnostics.append(
                        {
                            "type": "info",
                            "label": "公式转换",
                            "text": str(conversion_note),
                        }
                    )
                if converted.get("conversion_failed"):
                    diagnostics.append(
                        {
                            "type": "warning",
                            "label": "公式转换失败",
                            "text": "未能把 upstream formulation 转成可执行的 FactorHub 表达式或代码。",
                        }
                    )
                coded_items.append(
                    {
                        "candidate_id": f"{config.task_id}-round-{iteration}-candidate-{index + 1}",
                        "raw_expression": formulation,
                        "expression": str(converted.get("expression") or formulation),
                        "factor_name": factor_name,
                        "description": str(task.get("description") or factor_name),
                        "factor_formulation": formulation,
                        "variables": dict(task.get("variables") or {}),
                        "implementation_code": converted.get("implementation_code"),
                        "upstream_conversion_failed": bool(converted.get("conversion_failed")),
                        "diagnostics": diagnostics,
                    }
                )
            return {
                "round_index": iteration,
                "base_factors": list(experiment.get("base_factors") or []),
                "candidate_universe": list(experiment.get("candidate_universe") or []),
                "execution_mode": "upstream_rdagent",
                "coded_items": coded_items,
            }

        coded_items: list[dict[str, Any]] = []
        for index, expression in enumerate(experiment.get("factor_formulations") or []):
            normalized_expression = normalize_rdagent_expression_for_parser(expression)
            diagnostics: list[dict[str, Any]] = []
            raw_expression = str(expression or "").strip()
            try:
                validate_rdagent_expression_contract(normalized_expression)
            except RDAgentExpressionFormatError as exc:
                diagnostics.append(
                    {
                        "type": "warning",
                        "label": "契约修复",
                        "text": str(exc),
                    }
                )
                normalized_expression = normalize_rdagent_expression_for_parser(raw_expression)
            coded_items.append(
                {
                    "candidate_id": f"{config.task_id}-round-{iteration}-candidate-{index + 1}",
                    "raw_expression": raw_expression,
                    "expression": normalized_expression,
                    "diagnostics": diagnostics,
                }
            )
        return {
            "round_index": iteration,
            "base_factors": list(experiment.get("base_factors") or []),
            "candidate_universe": list(experiment.get("candidate_universe") or []),
            "coded_items": coded_items,
        }

    def _code_native_experiment(
        self,
        *,
        config: RDAgentMiningConfig,
        experiment: dict[str, Any],
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        coded_items: list[dict[str, Any]] = []
        accepted_code_examples = [
            str((item.get("execution_meta") or {}).get("implementation_code") or "").strip()
            for item in (rounds[-1].get("candidates") or [])
            if str((item.get("execution_meta") or {}).get("implementation_code") or "").strip()
        ] if rounds else []
        for index in range(max(int(config.candidates_per_iteration or 1), 1)):
            response = self._call_llm_json(
                system_prompt=self._native_code_executor.system_prompt,
                user_prompt=self._native_code_executor.build_user_prompt(
                    objective=config.objective,
                    hypothesis=hypothesis,
                    candidate_fields=list(experiment.get("candidate_universe") or []),
                    base_factors=list(experiment.get("base_factors") or []),
                    known_implementations=list(config.previous_expressions or []),
                    accepted_code_examples=accepted_code_examples,
                ),
            )
            diagnostics: list[dict[str, Any]] = []
            try:
                factor_name, implementation_code = self._native_code_executor.extract_code_from_llm_response(response)
            except Exception as exc:
                diagnostics.append({"type": "warning", "label": "代码回退", "text": str(exc)})
                factor_name, implementation_code = self._native_code_executor.fallback_code(
                    candidate_fields=list(experiment.get("candidate_universe") or []),
                )
            coded_items.append(
                {
                    "candidate_id": f"{config.task_id}-round-{iteration}-candidate-{index + 1}",
                    "raw_expression": implementation_code,
                    "expression": factor_name,
                    "implementation_code": implementation_code,
                    "diagnostics": diagnostics,
                }
            )
        return {
            "round_index": iteration,
            "base_factors": list(experiment.get("base_factors") or []),
            "candidate_universe": list(experiment.get("candidate_universe") or []),
            "execution_mode": "native_code",
            "coded_items": coded_items,
        }

    def _run_experiment(
        self,
        *,
        config: RDAgentMiningConfig,
        coded_experiment: dict[str, Any],
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
        sota_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stock_codes = self._auto_mining_service.data_service.get_stock_universe(config.universe, date=config.start_date)[:30]
        if not stock_codes:
            raise ValueError(f"股票池 {config.universe} 未返回可用股票")

        candidates: list[dict[str, Any]] = []
        evaluation_results: list[FactorEvaluationResult] = []
        skipped_candidates: list[dict[str, Any]] = []
        for index, coded_item in enumerate(coded_experiment.get("coded_items") or []):
            evaluation = self._evaluate_candidate(
                config=config,
                coded_item=coded_item,
                stock_codes=stock_codes,
                hypothesis=hypothesis,
            )
            if evaluation is None:
                skipped_candidates.append(
                    self._build_skipped_candidate_diagnostic(
                        config=config,
                        coded_item=coded_item,
                        iteration=iteration,
                        index=index,
                    )
                )
                continue
            evaluation_results.append(evaluation)
            candidate_payload = self._format_candidate_payload(
                evaluation=evaluation,
                coded_item=coded_item,
                hypothesis=hypothesis,
                iteration=iteration,
                index=index,
                acceptance_policy=config.acceptance_policy,
            )
            candidates.append(candidate_payload)

        if not candidates:
            raise ValueError(self._build_empty_candidate_error_message(
                config=config,
                iteration=iteration,
                skipped_candidates=skipped_candidates,
            ))

        candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        policy = dict(config.acceptance_policy or {})
        policy["_sota_candidates"] = list(sota_candidates)
        for rank, candidate in enumerate(candidates):
            self._apply_acceptance_policy(candidate, policy, rank)

        best_candidate = candidates[0]
        metrics = {
            "score": float(best_candidate.get("score") or 0.0),
            "avg_score": _average_score(candidates),
            "accepted_count": len([candidate for candidate in candidates if candidate.get("status") == "accepted"]),
        }
        return {
            "candidates": candidates,
            "metrics": metrics,
            "report_ref": best_candidate.get("report_url"),
            "backtest_engine": "factorhub_rdagent_executor",
            "best_candidate": best_candidate,
        }

    def _build_skipped_candidate_diagnostic(
        self,
        *,
        config: RDAgentMiningConfig,
        coded_item: dict[str, Any],
        iteration: int,
        index: int,
    ) -> dict[str, Any]:
        execution_mode = str(config.execution_mode or "native_code").strip().lower()
        factor_name = str(
            coded_item.get("factor_name")
            or coded_item.get("expression")
            or coded_item.get("raw_expression")
            or f"candidate_{index + 1}"
        ).strip()
        diagnostics = list(coded_item.get("diagnostics") or [])
        if coded_item.get("upstream_conversion_failed"):
            reason = "upstream formulation 未能转换为 FactorHub 可执行表达式或 Python 因子函数"
        elif execution_mode == "native_code" and not str(coded_item.get("implementation_code") or "").strip():
            reason = "native_code 候选未生成 implementation_code"
        elif execution_mode == "upstream_rdagent" and str(coded_item.get("implementation_code") or "").strip():
            reason = "Python 因子函数在本地数据评估时未产出有效 panel"
        else:
            reason = "候选表达式未通过本地评估，可能是字段/函数不支持、panel 为空或回测数据不足"
        return {
            "candidate_id": coded_item.get("candidate_id") or f"{config.task_id}-round-{iteration}-candidate-{index + 1}",
            "factor_name": factor_name,
            "reason": reason,
            "raw_expression": str(coded_item.get("raw_expression") or "")[:240],
            "expression": str(coded_item.get("expression") or "")[:240],
            "diagnostics": diagnostics,
        }

    def _build_empty_candidate_error_message(
        self,
        *,
        config: RDAgentMiningConfig,
        iteration: int,
        skipped_candidates: list[dict[str, Any]],
    ) -> str:
        if not skipped_candidates:
            return "RDAgent 本轮没有产出可评估的候选表达式"

        reason_counter: dict[str, int] = {}
        for item in skipped_candidates:
            reason = str(item.get("reason") or "unknown").strip() or "unknown"
            reason_counter[reason] = reason_counter.get(reason, 0) + 1

        summary = "；".join(f"{reason} x{count}" for reason, count in reason_counter.items())
        samples = []
        for item in skipped_candidates[:3]:
            factor_name = str(item.get("factor_name") or item.get("candidate_id") or "candidate").strip()
            raw_expression = str(item.get("raw_expression") or item.get("expression") or "").strip()
            if raw_expression:
                samples.append(f"{factor_name}: {raw_expression[:120]}")
            else:
                samples.append(f"{factor_name}: <empty>")
        sample_text = " | ".join(samples)
        return (
            f"RDAgent 本轮没有产出可评估的候选表达式。"
            f" execution_mode={config.execution_mode}，round={iteration}。"
            f" 跳过原因汇总：{summary}。"
            f" 候选样例：{sample_text}"
        )

    def _evaluate_candidate(
        self,
        *,
        config: RDAgentMiningConfig,
        coded_item: dict[str, Any],
        stock_codes: list[str],
        hypothesis: dict[str, Any],
    ) -> FactorEvaluationResult | None:
        execution_mode = str(config.execution_mode or "native_code").strip().lower()
        if execution_mode == "upstream_rdagent" and str(coded_item.get("implementation_code") or "").strip():
            implementation_code = str(coded_item.get("implementation_code") or "").strip()
            return self._evaluate_native_code_candidate(
                config=config,
                coded_item={
                    **coded_item,
                    "expression": str(coded_item.get("expression") or coded_item.get("factor_name") or "RDAgentUpstreamFactor"),
                    "raw_expression": implementation_code,
                    "implementation_code": implementation_code,
                },
                stock_codes=stock_codes,
                hypothesis=hypothesis,
                engine_type="rdagent_upstream_native_code",
            )

        if execution_mode != "native_code":
            if coded_item.get("upstream_conversion_failed"):
                return None
            expression = coded_item.get("expression") or ""
            return self._auto_mining_service.evaluate_expression(
                expression=expression,
                prompt=str(hypothesis.get("statement") or config.objective),
                stock_codes=stock_codes,
                start_date=config.start_date,
                end_date=config.end_date,
                benchmark=config.benchmark,
                n_groups=config.n_groups,
                holding_period=config.holding_period,
                direction=str(hypothesis.get("research_direction") or config.direction or "score"),
                neutralize_industry=config.neutralize_industry,
                neutralize_cap=config.neutralize_cap,
            )

        return self._evaluate_native_code_candidate(
            config=config,
            coded_item=coded_item,
            stock_codes=stock_codes,
            hypothesis=hypothesis,
            engine_type="rdagent_native_code",
        )

    def _evaluate_native_code_candidate(
        self,
        *,
        config: RDAgentMiningConfig,
        coded_item: dict[str, Any],
        stock_codes: list[str],
        hypothesis: dict[str, Any],
        engine_type: str,
    ) -> FactorEvaluationResult | None:
        implementation_code = str(coded_item.get("implementation_code") or coded_item.get("raw_expression") or "").strip()
        if not implementation_code:
            return None

        panel, diagnostics = self._factor_evaluation_service._build_panel_from_stock_rows(
            expression=implementation_code,
            stock_codes=stock_codes,
            start_date=config.start_date,
            end_date=config.end_date,
            holding_period=config.holding_period,
            stock_data_loader=self._auto_mining_service.data_service.get_stock_data,
            expression_executor=lambda stock_df, code: self._native_code_executor.execute_on_frame(stock_df, code),
        )
        if panel.empty:
            return None

        result = self._factor_evaluation_service._evaluate_panel(
            panel=panel,
            expression=str(coded_item.get("expression") or "RDAgentNativeFactor"),
            prompt=str(hypothesis.get("statement") or config.objective),
            benchmark=config.benchmark,
            n_groups=config.n_groups,
            holding_period=config.holding_period,
            direction=str(hypothesis.get("research_direction") or config.direction or "score"),
            benchmark_loader=self._auto_mining_service._load_benchmark_returns,
            report_writer=self._auto_mining_service._write_candidate_report,
            engine_type=engine_type,
            dialect="python_factor_function",
            canonical_expression=str(coded_item.get("expression") or "RDAgentNativeFactor"),
            canonical_ast=None,
            diagnostics=list(diagnostics or []) + list(coded_item.get("diagnostics") or []),
            start_date=config.start_date,
            end_date=config.end_date,
            metrics_source="rdagent_native_code_executor",
            execution_meta={
                "implementation_code": implementation_code,
                "execution_mode": "native_code",
            },
        )
        return result

    def _prepare_upstream_candidate(self, *, formulation: str, variables: dict[str, Any]) -> dict[str, Any]:
        raw_formulation = str(formulation or "").strip()
        if not raw_formulation:
            return {"expression": "", "conversion_failed": True}

        if raw_formulation.lstrip().startswith("def "):
            return {
                "expression": "RDAgentUpstreamFunctionFactor",
                "implementation_code": raw_formulation,
                "conversion_note": "upstream 直接返回了 Python 因子函数，已走本地代码执行评估。",
            }

        heuristic_expression = self._heuristic_convert_upstream_formulation(
            raw_formulation,
            variables=variables,
        )
        if heuristic_expression:
            return {
                "expression": heuristic_expression,
                "conversion_note": "已将 upstream formulation 自动归一化为 FactorHub 可执行表达式。",
            }

        llm_converted = self._convert_formulation_to_expression_with_llm(raw_formulation, variables)
        if llm_converted:
            if llm_converted.lstrip().startswith("def "):
                return {
                    "expression": "RDAgentUpstreamFunctionFactor",
                    "implementation_code": llm_converted,
                    "conversion_note": "已通过 LLM 将 upstream formulation 转为 Python 因子函数。",
                }
            normalized_expression = self._normalize_candidate_expression(llm_converted)
            if normalized_expression:
                return {
                    "expression": normalized_expression,
                    "conversion_note": "已通过 LLM 将 upstream formulation 转为 FactorHub 表达式。",
                }

        return {"expression": raw_formulation, "conversion_failed": True}

    def _heuristic_convert_upstream_formulation(
        self,
        formulation: str,
        *,
        variables: dict[str, Any] | None = None,
    ) -> str:
        expression = str(formulation or "").strip()
        if not expression:
            return ""

        normalized_expression = self._normalize_candidate_expression(expression)
        if normalized_expression and self._looks_like_factorhub_expression(normalized_expression):
            return normalized_expression

        expression = re.sub(r"^\s*[A-Za-z][A-Za-z0-9_]*(?:_\{[^}]+\})?\s*\(t\)\s*=\s*", "", expression)
        expression = re.sub(r"^\s*[A-Za-z][A-Za-z0-9_]*(?:_\{[^}]+\})?\s*=\s*", "", expression)
        expression = re.sub(r"^\s*F_t\s*=\s*", "", expression, flags=re.IGNORECASE)
        expression = self._replace_symbolic_window_tokens(expression, variables or {})
        expression = expression.replace("\\\\", "\\")
        expression = expression.replace("\\left", "").replace("\\right", "")
        expression = expression.replace("\\cdot", "*").replace("\\times", "*")
        expression = expression.replace("\\,", "")
        expression = expression.replace("\\ ", "")
        expression = expression.replace("\\_", "_")
        expression = re.sub(r",?\s*\\quad\s*[A-Za-z]\s*=\s*\d+\s*$", "", expression)
        expression = re.sub(r"\\text\{[^}]*\}", "", expression)
        expression = re.sub(r"^\s*[A-Za-z][A-Za-z0-9_]*_\{\}\s*=\s*", "", expression)
        top_level_equal = self._find_top_level_equal(expression)
        if top_level_equal != -1:
            lhs = expression[:top_level_equal]
            rhs = expression[top_level_equal + 1:]
            if not any(op in lhs for op in ["/", "*", "+", "-"]):
                expression = rhs.strip()
        expression = expression.replace("\\varepsilon", "1e-6").replace("\\epsilon", "1e-6")
        expression = re.sub(r"\\operatorname\{(ts_[A-Za-z_][A-Za-z0-9_]*)\}", r"\1", expression)
        expression = self._replace_reference_window_operators(expression)
        expression = self._replace_reference_volatility_term(expression)
        expression = self._replace_reference_sum_averages(expression)
        expression = re.sub(r"\\ln\s*\(", "log(", expression, flags=re.IGNORECASE)
        expression = re.sub(r"\\log\s*\(", "log(", expression, flags=re.IGNORECASE)
        expression = re.sub(r"\\sqrt\s*\(", "sqrt(", expression, flags=re.IGNORECASE)
        expression = self._replace_latex_sqrt(expression)
        expression = self._replace_latex_sums(expression)
        expression = self._replace_latex_frac(expression)
        expression = self._replace_latex_tokens(expression)
        expression = self._replace_absolute_value_bars(expression)
        expression = expression.replace("10**(-6)", "1e-6").replace("10**(-06)", "1e-6")
        expression = self._strip_ts_function_time_suffix(expression)
        expression = re.sub(r"^\s*[A-Za-z][A-Za-z0-9_]*_\(\)\s*=\s*", "", expression)
        expression = re.sub(r"(\))(?=ts_[A-Za-z_]+\()", r"\1 * ", expression)
        expression = re.sub(r"\s+", " ", expression).strip()

        normalized_expression = self._normalize_candidate_expression(expression)
        if normalized_expression and self._looks_like_factorhub_expression(normalized_expression):
            return normalized_expression
        return ""

    def _replace_symbolic_window_tokens(self, expression: str, variables: dict[str, Any]) -> str:
        result = str(expression or "")
        resolved_windows = self._infer_symbolic_windows(result, variables)
        for symbol, value in resolved_windows.items():
            result = re.sub(rf"\\sum_\{{i=0\}}\^\{{{re.escape(symbol)}-1\}}", rf"\\sum_{{i=0}}^{{{value - 1}}}", result)
            result = re.sub(rf"\\sum_\{{i=1\}}\^\{{{re.escape(symbol)}\}}", rf"\\sum_{{i=1}}^{{{value}}}", result)
            result = re.sub(rf"\{{1\}}\{{{re.escape(symbol)}\}}", rf"{{1}}{{{value}}}", result)
            result = re.sub(rf"_\{{{re.escape(symbol)}\}}", f"_{{{value}}}", result)
            result = re.sub(rf"_\{{t-{re.escape(symbol)}\}}", f"_{{t-{value}}}", result)
            result = re.sub(rf"_\{{t-{re.escape(symbol)}-1\}}", f"_{{t-{value}-1}}", result)
            result = re.sub(rf"\^\{{\({re.escape(symbol)}\)\}}", f"^{{({value})}}", result)
            result = re.sub(rf"\^\{{{re.escape(symbol)}\}}", f"^{{{value}}}", result)
            result = re.sub(rf"\^\{{{re.escape(symbol)}-1\}}", f"^{{{value}-1}}", result)
        return result

    def _infer_symbolic_windows(self, expression: str, variables: dict[str, Any]) -> dict[str, int]:
        result: dict[str, int] = {}
        for key, value in (variables or {}).items():
            if str(key).strip().lower() not in {"n", "window", "lookback"}:
                continue
            text = str(value or "")
            match = re.search(r"(\d+)", text)
            if match:
                result[str(key).strip()] = int(match.group(1))

        for match in re.finditer(r"(?:^|,|\\quad)\s*([A-Za-z])\s*=\s*(\d+)\b", expression):
            symbol = match.group(1).strip()
            if symbol not in result:
                result[symbol] = int(match.group(2))

        if "n" not in result and re.search(r"\b(?:MA|Mean|Std)_\{n\}|_\{t-n\}|\^\{n-1\}", expression, flags=re.IGNORECASE):
            result["n"] = 5
        return result

    def _replace_reference_window_operators(self, expression: str) -> str:
        result = str(expression or "")
        field_aliases = {
            "C": "close",
            "V": "volume",
            "A": "amount",
            "O": "open",
            "H": "high",
            "L": "low",
            "C_t": "close",
            "V_t": "volume",
            "A_t": "amount",
        }

        std_window_pattern = re.compile(
            r"\\operatorname\{std\}\(\s*(.+?)\s*,\s*\\tau\s*=\s*t-(\d+)\s*,\s*\\ldots\s*,\s*t\s*\)",
            flags=re.IGNORECASE,
        )
        while True:
            match = std_window_pattern.search(result)
            if not match:
                break
            inner_expression = match.group(1).strip()
            lookback = int(match.group(2)) + 1
            normalized_inner = inner_expression
            normalized_inner = re.sub(r"\\frac\s*\{\s*C_\{\\tau\}\s*\}\s*\{\s*C_\{\\tau-1\}\s*\}\s*-\s*1", "returns", normalized_inner, flags=re.IGNORECASE)
            normalized_inner = re.sub(r"\\frac\s*\{\s*close_\{\\tau\}\s*\}\s*\{\s*close_\{\\tau-1\}\s*\}\s*-\s*1", "returns", normalized_inner, flags=re.IGNORECASE)
            replacement = f"ts_std({normalized_inner},{lookback})"
            result = result[:match.start()] + replacement + result[match.end():]

        ma_function_pattern = re.compile(
            r"\\operatorname\{MA\}\(([^,]+),\s*(\d+)\)_t",
            flags=re.IGNORECASE,
        )
        while True:
            match = ma_function_pattern.search(result)
            if not match:
                break
            raw_field = match.group(1).strip()
            window = match.group(2)
            mapped_field = field_aliases.get(raw_field, raw_field)
            mapped_field = mapped_field.replace("A/V", "(amount / volume)").replace("a/v", "(amount / volume)")
            replacement = f"ts_mean({mapped_field},{window})"
            result = result[:match.start()] + replacement + result[match.end():]

        operator_pattern = re.compile(
            r"\\operatorname\{(MA|Mean|Std)\}_(?:\{(\d+)\}|(\d+))\(",
            flags=re.IGNORECASE,
        )
        while True:
            match = operator_pattern.search(result)
            if not match:
                break
            operator_name = match.group(1).lower()
            window = match.group(2) or match.group(3)
            arg_start = match.end() - 1
            arg_end = self._find_matching_parenthesis(result, arg_start)
            if arg_end == -1:
                break
            raw_field = result[arg_start + 1:arg_end].strip()
            mapped_field = field_aliases.get(raw_field, raw_field)
            mapped_field = mapped_field.replace("A/V", "(amount / volume)").replace("a/v", "(amount / volume)")
            if operator_name == "std":
                func = "ts_std"
            else:
                func = "ts_mean"
            replacement = f"{func}({mapped_field},{window})"
            suffix_end = arg_end + 1
            if result[suffix_end:suffix_end + 2] == "_t":
                suffix_end += 2
            result = result[:match.start()] + replacement + result[suffix_end:]
        return result

    def _replace_latex_frac(self, expression: str) -> str:
        result = expression
        while "\\frac" in result:
            match = re.search(r"\\frac\s*\{", result)
            if not match:
                break
            numerator_start = match.end() - 1
            numerator_end = self._find_matching_brace(result, numerator_start)
            if numerator_end == -1:
                break
            denominator_start = numerator_end + 1
            while denominator_start < len(result) and result[denominator_start].isspace():
                denominator_start += 1
            if denominator_start >= len(result) or result[denominator_start] != "{":
                break
            denominator_end = self._find_matching_brace(result, denominator_start)
            if denominator_end == -1:
                break
            numerator = result[numerator_start + 1:numerator_end]
            denominator = result[denominator_start + 1:denominator_end]
            replacement = f"(({numerator}) / ({denominator}))"
            result = result[:match.start()] + replacement + result[denominator_end + 1:]
        return result

    def _replace_latex_sums(self, expression: str) -> str:
        result = str(expression or "")
        field_aliases = {
            "c": "close",
            "o": "open",
            "h": "high",
            "l": "low",
            "v": "volume",
            "amt": "amount",
            "a": "amount",
            "r": "returns",
            "close": "close",
            "open": "open",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "return": "returns",
            "returns": "returns",
        }
        sum_pattern = re.compile(
            r"\\sum_\{i=1\}\^\{(\d+)\}\s*([A-Za-z_][A-Za-z0-9_]*)_\{t-i\}",
            flags=re.IGNORECASE,
        )
        while True:
            match = sum_pattern.search(result)
            if not match:
                break
            window = match.group(1)
            field = field_aliases.get(match.group(2).lower(), match.group(2).lower())
            replacement = f"ts_sum({field},{window})"
            result = result[:match.start()] + replacement + result[match.end():]
        sum_pattern_zero = re.compile(
            r"\\sum_\{i=0\}\^\{(\d+)\}\s*([A-Za-z_][A-Za-z0-9_]*)_\{t-i\}",
            flags=re.IGNORECASE,
        )
        while True:
            match = sum_pattern_zero.search(result)
            if not match:
                break
            window = int(match.group(1)) + 1
            field = field_aliases.get(match.group(2).lower(), match.group(2).lower())
            replacement = f"ts_sum({field},{window})"
            result = result[:match.start()] + replacement + result[match.end():]
        return result

    def _replace_reference_sum_averages(self, expression: str) -> str:
        result = str(expression or "")

        # 1/N * sum_{i=0}^{N-1} field_{t-i} -> ts_mean(field, N)
        avg_field_pattern = re.compile(
            r"\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*\\sum_\{i=0\}\^\{(\d+)\}\s*([A-Za-z_][A-Za-z0-9_]*)_\{t-i\}",
            flags=re.IGNORECASE,
        )
        field_aliases = {
            "c": "close",
            "v": "volume",
            "a": "amount",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
        while True:
            match = avg_field_pattern.search(result)
            if not match:
                break
            denom = int(match.group(1))
            upper = int(match.group(2))
            field = field_aliases.get(match.group(3).lower(), match.group(3).lower())
            window = upper + 1
            if denom == window:
                replacement = f"ts_mean({field},{window})"
            else:
                replacement = f"(((1) / ({denom})) * ts_sum({field},{window}))"
            result = result[:match.start()] + replacement + result[match.end():]

        avg_field_symbolic_pattern = re.compile(
            r"\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*\\sum_\{i=0\}\^\{(\d+)-1\}\s*([A-Za-z_][A-Za-z0-9_]*)_\{t-i\}",
            flags=re.IGNORECASE,
        )
        while True:
            match = avg_field_symbolic_pattern.search(result)
            if not match:
                break
            denom = int(match.group(1))
            upper_base = int(match.group(2))
            field = field_aliases.get(match.group(3).lower(), match.group(3).lower())
            window = upper_base
            if denom == window:
                replacement = f"ts_mean({field},{window})"
            else:
                replacement = f"(((1) / ({denom})) * ts_sum({field},{window}))"
            result = result[:match.start()] + replacement + result[match.end():]

        avg_field_expanded_pattern = re.compile(
            r"\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*\\sum_\{i=0\}\^\{(\d+)\}\s*([A-Za-z_][A-Za-z0-9_]*)_\{t-0\}",
            flags=re.IGNORECASE,
        )
        while True:
            match = avg_field_expanded_pattern.search(result)
            if not match:
                break
            denom = int(match.group(1))
            upper = int(match.group(2))
            field = field_aliases.get(match.group(3).lower(), match.group(3).lower())
            window = upper + 1
            if denom == window:
                replacement = f"ts_mean({field},{window})"
            else:
                replacement = f"(((1) / ({denom})) * ts_sum({field},{window}))"
            result = result[:match.start()] + replacement + result[match.end():]

        # 1/N * sum_{i=0}^{N-1} (expr_{t-i}) -> ts_mean(expr, N) for amount/volume style ratio
        avg_ratio_pattern = re.compile(
            r"\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*\\sum_\{i=0\}\^\{(\d+)\}\s*"
            r"(?:\\left\(|\()\s*\\frac\{([A-Za-z_][A-Za-z0-9_]*)_\{t-(?:i|0)\}\}\{([A-Za-z_][A-Za-z0-9_]*)_\{t-(?:i|0)\}\}\s*(?:\\right\)|\))",
            flags=re.IGNORECASE,
        )
        while True:
            match = avg_ratio_pattern.search(result)
            if not match:
                break
            denom = int(match.group(1))
            upper = int(match.group(2))
            numerator_field = field_aliases.get(match.group(3).lower(), match.group(3).lower())
            denominator_field = field_aliases.get(match.group(4).lower(), match.group(4).lower())
            window = upper + 1
            base_expr = f"(({numerator_field}) / ({denominator_field}))"
            if denom == window:
                replacement = f"ts_mean({base_expr},{window})"
            else:
                replacement = f"(((1) / ({denom})) * ts_sum({base_expr},{window}))"
            result = result[:match.start()] + replacement + result[match.end():]

        avg_ratio_symbolic_pattern = re.compile(
            r"\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*\\sum_\{i=0\}\^\{(\d+)-1\}\s*"
            r"(?:\\left\(|\()\s*\\frac\{([A-Za-z_][A-Za-z0-9_]*)_\{t-(?:i|0)\}\}\{([A-Za-z_][A-Za-z0-9_]*)_\{t-(?:i|0)\}\}\s*(?:\\right\)|\))",
            flags=re.IGNORECASE,
        )
        while True:
            match = avg_ratio_symbolic_pattern.search(result)
            if not match:
                break
            denom = int(match.group(1))
            upper_base = int(match.group(2))
            numerator_field = field_aliases.get(match.group(3).lower(), match.group(3).lower())
            denominator_field = field_aliases.get(match.group(4).lower(), match.group(4).lower())
            window = upper_base
            base_expr = f"(({numerator_field}) / ({denominator_field}))"
            if denom == window:
                replacement = f"ts_mean({base_expr},{window})"
            else:
                replacement = f"(((1) / ({denom})) * ts_sum({base_expr},{window}))"
            result = result[:match.start()] + replacement + result[match.end():]

        return result

    def _replace_reference_volatility_term(self, expression: str) -> str:
        result = str(expression or "")
        volatility_pattern = re.compile(
            r"\\sqrt\s*\{\s*\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*"
            r"\\sum_\{i=1\}\^\{\1\}\s*"
            r"\(?\s*r_\{t-i\+1\}\s*-\s*\\bar\s*r_t\s*\)?\s*\^2\s*\}",
            flags=re.IGNORECASE,
        )
        while True:
            match = volatility_pattern.search(result)
            if not match:
                break
            window = match.group(1)
            replacement = f"ts_std(returns,{window})"
            result = result[:match.start()] + replacement + result[match.end():]
        return result

    def _replace_latex_sqrt(self, expression: str) -> str:
        result = str(expression or "")
        while "\\sqrt" in result:
            match = re.search(r"\\sqrt\s*\{", result)
            if not match:
                break
            inner_start = match.end() - 1
            inner_end = self._find_matching_brace(result, inner_start)
            if inner_end == -1:
                break
            inner = result[inner_start + 1:inner_end]
            replacement = f"sqrt({inner})"
            result = result[:match.start()] + replacement + result[inner_end + 1:]
        return result

    @staticmethod
    def _find_matching_brace(text: str, start: int) -> int:
        if start < 0 or start >= len(text) or text[start] != "{":
            return -1
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @staticmethod
    def _find_matching_parenthesis(text: str, start: int) -> int:
        if start < 0 or start >= len(text) or text[start] != "(":
            return -1
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @staticmethod
    def _strip_ts_function_time_suffix(expression: str) -> str:
        result = str(expression or "")
        search_start = 0
        while True:
            match = re.search(r"\bts_[A-Za-z_][A-Za-z0-9_]*\(", result[search_start:])
            if not match:
                break
            paren_start = search_start + match.end() - 1
            paren_end = RDAgentFactorMiningService._find_matching_parenthesis(result, paren_start)
            if paren_end == -1:
                search_start = paren_start + 1
                continue
            if result[paren_end + 1:paren_end + 3] == "_t":
                result = result[:paren_end + 1] + result[paren_end + 3:]
            search_start = paren_end + 1
        return result

    @staticmethod
    def _replace_absolute_value_bars(expression: str) -> str:
        result = str(expression or "")
        while "|" in result:
            start = result.find("|")
            if start == -1:
                break
            end = result.find("|", start + 1)
            if end == -1:
                break
            inner = result[start + 1:end].strip()
            result = result[:start] + f"abs({inner})" + result[end + 1:]
        return result

    @staticmethod
    def _find_top_level_equal(text: str) -> int:
        paren_depth = 0
        brace_depth = 0
        for index, char in enumerate(str(text or "")):
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(paren_depth - 1, 0)
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(brace_depth - 1, 0)
            elif char == "=" and paren_depth == 0 and brace_depth == 0:
                return index
        return -1

    def _replace_latex_tokens(self, expression: str) -> str:
        result = str(expression or "")
        field_aliases = {
            "c": "close",
            "o": "open",
            "h": "high",
            "l": "low",
            "v": "volume",
            "amt": "amount",
            "a": "amount",
            "r": "returns",
            "close": "close",
            "open": "open",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "return": "returns",
            "returns": "returns",
        }
        for token, field in sorted(field_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            result = re.sub(
                rf"\b{token}_\{{t-(\d+)\}}",
                lambda match, field_name=field: f"ts_shift({field_name},{match.group(1)})",
                result,
                flags=re.IGNORECASE,
            )
            result = re.sub(
                rf"\b{token}_\{{t-i\+1\}}",
                lambda match, field_name=field: f"ts_shift({field_name},i-1)",
                result,
                flags=re.IGNORECASE,
            )
            result = re.sub(
                rf"\b{token}_\{{t\}}",
                field,
                result,
                flags=re.IGNORECASE,
            )
            result = re.sub(rf"\b{token}_t\b", field, result, flags=re.IGNORECASE)
            result = re.sub(
                rf"\b{token}_\{{t-i\}}",
                f"ts_shift({field},1)",
                result,
                flags=re.IGNORECASE,
            )

        result = re.sub(r"\\bar\s*r_t", "ts_mean(returns,20)", result, flags=re.IGNORECASE)
        result = result.replace("{", "(").replace("}", ")")
        result = result.replace("^", "**")
        return result

    def _normalize_candidate_expression(self, expression: str) -> str:
        adapted_expression = ExpressionAdapter.adapt(str(expression or "").strip())
        normalized_expression = normalize_rdagent_expression_for_parser(adapted_expression)
        return str(normalized_expression or "").strip()

    @staticmethod
    def _looks_like_factorhub_expression(expression: str) -> bool:
        text = str(expression or "").strip()
        if not text:
            return False
        if "\\" in text or "text(" in text.lower():
            return False
        if not re.search(r"\b(rank|ts_|close|open|high|low|volume|amount|returns)\b", text, flags=re.IGNORECASE):
            return False

        allowed_functions = {
            "rank",
            "where",
            "abs",
            "log",
            "max",
            "min",
            "clip",
            "zscore",
            "sign",
            "sqrt",
            "power",
            "returns",
            "obv",
            "sma",
            "ma",
            "exp",
            "sigmoid",
            "tanh",
        }
        function_names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        for name in function_names:
            lowered = name.lower()
            if lowered.startswith("ts_"):
                continue
            if lowered in allowed_functions:
                continue
            return False
        return True

    def _convert_formulation_to_expression_with_llm(self, formulation: str, variables: dict[str, Any]) -> str | None:
        runtime_config = llm_config_service.get_runtime_config()
        if not str(runtime_config.get("api_key") or "").strip():
            return None

        try:
            from backend.engines.factor_engine import _get_client, _get_model
        except Exception:
            return None

        prompt = (
            "你是一个量化因子公式转换助手。\n"
            "请把下面的 RD-Agent formulation，转换成 FactorHub 可执行的表达式或 def calculate_factor(df) 函数。\n"
            "要求：\n"
            "1. 如果可以直接写成表达式，就只返回表达式。\n"
            "2. 如果需要多步逻辑，就返回 def calculate_factor(df): ...\n"
            "3. 只能使用 FactorHub 常见字段：open/high/low/close/volume/amount，以及 pandas/NumPy 常见滚动写法。\n"
            "4. 不要输出解释，不要加代码块。\n"
            f"formulation：{formulation}\n"
            f"variables：{variables}\n"
        )

        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=_get_model(),
                messages=[
                    {"role": "system", "content": "你只返回可执行因子代码。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=600,
                timeout=60,
            )
            content = str((response.choices[0].message.content or "")).strip()
        except Exception:
            return None

        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("python"):
                    content = content[6:]
        return content.strip() or None

    def _generate_upstream_round_plan(
        self,
        *,
        config: RDAgentMiningConfig,
        rounds: list[dict[str, Any]],
        iteration: int,
        current_base_factors: list[str],
    ) -> dict[str, Any]:
        return self._upstream_proposal_adapter.generate_round_plan(
            objective=config.objective,
            iteration=iteration,
            candidate_universe=list(config.candidate_universe),
            current_base_factors=list(current_base_factors),
            rounds=rounds,
        )

    def _generate_feedback(
        self,
        *,
        config: RDAgentMiningConfig,
        hypothesis: dict[str, Any],
        experiment: dict[str, Any],
        run_result: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        candidates = list(run_result.get("candidates") or [])
        best_candidate = candidates[0] if candidates else {}
        score = float(best_candidate.get("score") or 0.0)
        accepted_count = len([candidate for candidate in candidates if candidate.get("status") == "accepted"])
        supported = accepted_count > 0 and score >= 60
        next_goal = str(hypothesis.get("research_direction") or config.direction or "score")
        if not supported and next_goal == "score":
            next_goal = "ls_sharpe"
        observations = (
            f"本轮共评估 {len(candidates)} 个候选，最佳 Score 为 {score:.2f}，"
            f"人工确认候选 {accepted_count} 个。"
        )
        return {
            "observations": observations,
            "hypothesis_evaluation": "supported" if supported else "needs_revision",
            "next_hypothesis": f"继续围绕 {next_goal} 优化表达式稳定性与可执行性。",
            "reason": "保留通过统一评价且满足 RDAgent 接受策略的候选，未达标则切换到更稳健的目标。",
            "decision": supported,
            "acceptable": supported,
            "accepted_count": accepted_count,
        }

    def _generate_factor_formulations(
        self,
        *,
        config: RDAgentMiningConfig,
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
        current_base_factors: list[str],
        known_expressions: list[str],
    ) -> list[str]:
        sota_expressions = [
            str(item.get("expression") or "").strip()
            for item in (rounds[-1].get("sota_candidates") or [])
        ] if rounds else []
        llm_response = self._call_llm_json(
            system_prompt=_RDAGENT_EXPERIMENT_SYSTEM_PROMPT,
            user_prompt=self._build_experiment_prompt(
                config=config,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
                current_base_factors=current_base_factors,
                known_expressions=known_expressions,
                sota_expressions=sota_expressions,
            ),
        )
        values = llm_response.get("factor_formulations") or llm_response.get("expressions") or []
        if not isinstance(values, list):
            values = []
        normalized: list[str] = []
        for value in values:
            expression = normalize_rdagent_expression_for_parser(value)
            if not expression:
                continue
            try:
                validate_rdagent_expression_contract(expression)
            except RDAgentExpressionFormatError as exc:
                logger.info("RDAgent 表达式被契约过滤: %s", exc)
                continue
            normalized.append(expression)
            if len(normalized) >= max(int(config.candidates_per_iteration or 1), 1):
                break
        if normalized:
            return _dedupe_strings(normalized)

        logger.info("RDAgent LLM 未返回有效表达式，使用 fallback 表达式模板。")
        fallback_fields = list(config.candidate_universe or ["close", "volume"])
        primary_field = fallback_fields[0]
        secondary_field = fallback_fields[min(1, len(fallback_fields) - 1)]
        return _dedupe_strings(
            [
                f"rank(ts_delta({primary_field}, 5))",
                f"rank(ts_mean({secondary_field}, 10) / (ts_std({secondary_field}, 10) + 1e-6))",
                "rank(ts_zscore(close, 20))",
            ][: max(int(config.candidates_per_iteration or 1), 1)]
        )

    def _build_hypothesis_prompt(
        self,
        *,
        config: RDAgentMiningConfig,
        rounds: list[dict[str, Any]],
        iteration: int,
        current_base_factors: list[str],
    ) -> str:
        latest_feedback = rounds[-1].get("feedback") if rounds else {}
        latest_evaluation = rounds[-1].get("evaluation") if rounds else {}
        acceptance_policy = self._serialize_acceptance_policy(config.acceptance_policy)
        return (
            f"研究目标：{config.objective}\n"
            f"当前轮次：{iteration}\n"
            f"总轮数预算：{max(1, min(int(config.max_iterations or 1), MAX_RDAGENT_ITERATIONS))}\n"
            f"优化方向：{config.direction or 'score'}\n"
            f"回测区间：{config.start_date} 至 {config.end_date}\n"
            f"股票池：{config.universe}\n"
            f"基准：{config.benchmark}\n"
            f"分组数：{config.n_groups}\n"
            f"持有期：{config.holding_period}\n"
            f"行业中性化：{'是' if config.neutralize_industry else '否'}\n"
            f"市值中性化：{'是' if config.neutralize_cap else '否'}\n"
            f"基础因子：{json.dumps(current_base_factors, ensure_ascii=False)}\n"
            f"候选字段：{json.dumps(config.candidate_universe, ensure_ascii=False)}\n"
            f"候选数量预算：本轮最多生成 {max(int(config.candidates_per_iteration or 1), 1)} 条候选表达式\n"
            f"验收阈值：{json.dumps(acceptance_policy, ensure_ascii=False)}\n"
            f"延续任务：{config.continuation_of or '否'}\n"
            f"上一轮反馈 ID：{config.previous_feedback_id or '无'}\n"
            f"上一轮反馈：{json.dumps(latest_feedback or {}, ensure_ascii=False)}\n"
            f"上一轮评估：{json.dumps(latest_evaluation or {}, ensure_ascii=False)}\n"
            "请严格围绕以上预算、验收阈值和研究边界输出下一轮 RDAgent 因子研究假设，避免提出无法通过当前阈值的宽泛方案。"
        )

    def _build_experiment_prompt(
        self,
        *,
        config: RDAgentMiningConfig,
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
        current_base_factors: list[str],
        known_expressions: list[str],
        sota_expressions: list[str],
    ) -> str:
        acceptance_policy = self._serialize_acceptance_policy(config.acceptance_policy)
        return (
            f"研究目标：{config.objective}\n"
            f"当前轮次：{iteration}\n"
            f"总轮数预算：{max(1, min(int(config.max_iterations or 1), MAX_RDAGENT_ITERATIONS))}\n"
            f"研究假设：{json.dumps(hypothesis, ensure_ascii=False)}\n"
            f"回测区间：{config.start_date} 至 {config.end_date}\n"
            f"股票池：{config.universe}\n"
            f"基准：{config.benchmark}\n"
            f"分组数：{config.n_groups}\n"
            f"持有期：{config.holding_period}\n"
            f"行业中性化：{'是' if config.neutralize_industry else '否'}\n"
            f"市值中性化：{'是' if config.neutralize_cap else '否'}\n"
            f"基础因子：{json.dumps(current_base_factors, ensure_ascii=False)}\n"
            f"候选字段：{json.dumps(config.candidate_universe, ensure_ascii=False)}\n"
            f"验收阈值：{json.dumps(acceptance_policy, ensure_ascii=False)}\n"
            f"历史表达式（禁止重复）：{json.dumps(known_expressions[-20:], ensure_ascii=False)}\n"
            f"当前 SOTA 候选表达式：{json.dumps(sota_expressions[-10:], ensure_ascii=False)}\n"
            f"表达式契约：{rdagent_expression_contract_text()}\n"
            f"需要生成 {max(int(config.candidates_per_iteration or 1), 1)} 条候选表达式。"
            "请只生成在当前候选字段范围内、且有机会满足验收阈值的表达式。"
        )

    @staticmethod
    def _serialize_acceptance_policy(policy: Any) -> dict[str, float]:
        raw = dict(policy or {})
        return {
            "max_correlation_with_sota": _safe_float(raw.get("max_correlation_with_sota"), 0.99),
            "min_rank_ic": _safe_float(raw.get("min_rank_ic"), 0.0),
            "min_annualized_return_delta": _safe_float(raw.get("min_annualized_return_delta"), 0.0),
            "max_drawdown_regression": _safe_float(raw.get("max_drawdown_regression"), 0.05),
            "min_valid_coverage": _safe_float(raw.get("min_valid_coverage"), 0.8),
        }

    def _call_llm_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        runtime_config = llm_config_service.get_runtime_config()
        if not runtime_config.get("api_key"):
            return {}

        client = self._create_llm_client(runtime_config)
        try:
            response = client.chat.completions.create(
                model=runtime_config.get("model") or "deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1600,
            )
        except Exception as exc:
            logger.warning("RDAgent LLM 调用失败，转用 fallback：%s", exc)
            return {}

        content = (response.choices[0].message.content or "").strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(content)
        except Exception:
            logger.warning("RDAgent LLM 返回了无法解析的 JSON：%s", content[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _create_llm_client(self, runtime_config: dict[str, Any]) -> Any:
        if self._llm_client_factory is not None:
            return self._llm_client_factory(runtime_config)
        from openai import OpenAI

        return OpenAI(
            api_key=runtime_config["api_key"],
            base_url=runtime_config.get("base_url") or "https://api.deepseek.com/v1",
        )

    def _format_candidate_payload(
        self,
        *,
        evaluation: FactorEvaluationResult,
        coded_item: dict[str, Any],
        hypothesis: dict[str, Any],
        iteration: int,
        index: int,
        acceptance_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        task_details = {
            "expression": evaluation.expression,
            "raw_expression": coded_item.get("raw_expression"),
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
                "component_scores": evaluation.component_scores,
                "wq_fitness": evaluation.wq_brain.get("wq_fitness"),
            },
            "rdagent": {
                "raw_expression": coded_item.get("raw_expression"),
                "adapter_normalized": coded_item.get("raw_expression") != evaluation.expression,
                "candidate_score": {
                    "score": evaluation.score,
                    "report_metrics": evaluation.report_metrics,
                    "backtest_summary": evaluation.backtest_summary,
                    "report_url": evaluation.report_url,
                    "rank_ic": evaluation.backtest_summary.get("rank_ic_mean"),
                    "sharpe": evaluation.report_metrics.get("sharpe", evaluation.backtest_summary.get("long_short_sharpe")),
                    "valid_coverage": 1.0,
                    "max_correlation_with_sota": None,
                },
                "hypothesis": hypothesis,
                "acceptance_policy": acceptance_policy or {},
            },
            "diagnostics": list(evaluation.diagnostics or []) + list(coded_item.get("diagnostics") or []),
            "round_evaluation": {
                "recommended_goal": hypothesis.get("expected_signal"),
                "primary_problem": hypothesis.get("reason"),
                "metric_snapshot": {
                    "score": evaluation.score,
                    "report_sharpe": evaluation.report_metrics.get("sharpe"),
                    "ls_sharpe": evaluation.backtest_summary.get("long_short_sharpe"),
                    "rank_ic": evaluation.backtest_summary.get("rank_ic_mean"),
                    "wq_fitness": evaluation.backtest_summary.get("wq_fitness"),
                },
            },
        }
        return {
            "candidate_id": coded_item.get("candidate_id"),
            "name": f"RDAgent_Factor_{iteration}_{index + 1}",
            "expression": evaluation.expression,
            "raw_expression": coded_item.get("raw_expression"),
            "score": evaluation.score,
            "grade": evaluation.grade,
            "fitness": evaluation.wq_brain.get("wq_fitness"),
            "ic": evaluation.backtest_summary.get("ic_mean"),
            "ir": evaluation.backtest_summary.get("ic_ir"),
            "rank_ic": evaluation.backtest_summary.get("rank_ic_mean"),
            "sharpe": evaluation.backtest_summary.get("long_short_sharpe"),
            "status": "watchlist",
            "source": "factorhub_rdagent_mining",
            "engine_type": evaluation.engine_type,
            "dialect": evaluation.dialect,
            "canonical_expression": evaluation.canonical_expression,
            "canonical_ast": evaluation.canonical_ast,
            "report_url": evaluation.report_url,
            "report_metrics": evaluation.report_metrics,
            "backtest_summary": evaluation.backtest_summary,
            "component_scores": evaluation.component_scores,
            "anti_overfit": evaluation.anti_overfit,
            "wq_brain": evaluation.wq_brain,
            "interpretation": evaluation.interpretation,
            "task_details": task_details,
            "quantgpt_task_details": task_details,
            "automation_meta": {
                "round_index": iteration,
                "source": "rdagent",
            },
            "execution_meta": evaluation.execution_meta,
            "_factor_frame": self._extract_factor_frame({"execution_meta": evaluation.execution_meta}),
        }

    def _apply_acceptance_policy(self, candidate: dict[str, Any], policy: dict[str, Any], rank: int) -> None:
        reasons: list[str] = []
        backtest_summary = candidate.get("backtest_summary") or {}
        report_metrics = candidate.get("report_metrics") or {}

        min_rank_ic = _safe_float(policy.get("min_rank_ic"), 0.0)
        rank_ic = _safe_float(backtest_summary.get("rank_ic_mean"))
        if rank_ic < min_rank_ic:
            reasons.append(f"rank_ic {rank_ic:.4f} 低于阈值 {min_rank_ic:.4f}")

        min_return = _safe_float(policy.get("min_annualized_return_delta"), 0.0)
        annual_return = _safe_float(backtest_summary.get("long_short_annual"))
        if annual_return < min_return:
            reasons.append(f"annualized_return {annual_return:.4f} 低于阈值 {min_return:.4f}")

        max_drawdown = _safe_float(policy.get("max_drawdown_regression"), 0.05)
        drawdown = abs(_safe_float(report_metrics.get("max_drawdown")))
        if drawdown > max_drawdown:
            reasons.append(f"max_drawdown {drawdown:.4f} 高于阈值 {max_drawdown:.4f}")

        coverage_floor = _safe_float(policy.get("min_valid_coverage"), 0.8)
        coverage = 1.0
        if coverage < coverage_floor:
            reasons.append(f"valid_coverage {coverage:.4f} 低于阈值 {coverage_floor:.4f}")

        max_corr = _safe_float(policy.get("max_correlation_with_sota"), 0.99)
        correlation = self._estimate_sota_correlation(candidate, policy.get("_sota_candidates") or [])
        if correlation is not None and correlation > max_corr:
            reasons.append(f"max_correlation_with_sota {correlation:.4f} 高于阈值 {max_corr:.4f}")

        status = "accepted" if not reasons else "watchlist"
        candidate["status"] = status
        rdagent_details = candidate.setdefault("task_details", {}).setdefault("rdagent", {})
        rdagent_details["policy_failure_reasons"] = reasons
        candidate["policy_diagnostics"] = {"failure_reasons": reasons}
        candidate["task_details"]["rdagent"]["candidate_score"]["valid_coverage"] = coverage
        candidate["task_details"]["rdagent"]["candidate_score"]["max_correlation_with_sota"] = correlation

    @staticmethod
    def _merge_sota_candidates(existing: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = list(existing)
        seen = {_normalize_expression_key(str(item.get("expression") or "")) for item in merged if str(item.get("expression") or "").strip()}
        for candidate in sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True):
            if candidate.get("status") != "accepted":
                continue
            expression = str(candidate.get("expression") or "").strip()
            if not expression:
                continue
            key = _normalize_expression_key(expression)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "name": candidate.get("name"),
                    "expression": expression,
                    "score": float(candidate.get("score") or 0.0),
                    "factor_snapshot": ((candidate.get("execution_meta") or {}).get("factor_snapshot") or []),
                }
            )
        merged.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return merged[:20]

    @classmethod
    def _estimate_sota_correlation(cls, candidate: dict[str, Any], sota_candidates: list[dict[str, Any]]) -> float | None:
        if not isinstance(sota_candidates, list):
            return None

        candidate_frame = cls._extract_factor_frame(candidate)
        if candidate_frame.empty:
            return None

        max_correlation: float | None = None
        for sota_candidate in sota_candidates:
            sota_frame = cls._extract_factor_frame(sota_candidate or {})
            if sota_frame.empty:
                continue
            correlation = cls._calculate_factor_frame_correlation(candidate_frame, sota_frame)
            if correlation is None:
                continue
            max_correlation = correlation if max_correlation is None else max(max_correlation, correlation)
        return round(max_correlation, 4) if max_correlation is not None else None

    @staticmethod
    def _extract_factor_frame(payload: dict[str, Any]) -> pd.DataFrame:
        runtime_frame = payload.get("_factor_frame")
        if isinstance(runtime_frame, pd.DataFrame):
            return runtime_frame.copy()

        execution_meta = payload.get("execution_meta") or {}
        snapshot = execution_meta.get("factor_snapshot")
        if snapshot is None:
            snapshot = payload.get("factor_snapshot")
        return RDAgentFactorMiningService._factor_frame_from_snapshot(snapshot)

    @staticmethod
    def _factor_frame_from_snapshot(snapshot: Any) -> pd.DataFrame:
        if not isinstance(snapshot, list) or not snapshot:
            return pd.DataFrame(columns=["date", "stock_code", "factor"])

        frame = pd.DataFrame(snapshot)
        required_columns = {"date", "stock_code", "factor"}
        if not required_columns.issubset(frame.columns):
            return pd.DataFrame(columns=["date", "stock_code", "factor"])

        frame = frame.loc[:, ["date", "stock_code", "factor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["stock_code"] = frame["stock_code"].astype(str)
        frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
        frame = frame.dropna(subset=["date", "stock_code", "factor"])
        return frame

    @staticmethod
    def _calculate_factor_frame_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
        if left.empty or right.empty:
            return None

        merged = left.merge(right, on=["date", "stock_code"], how="inner", suffixes=("_left", "_right"))
        merged["factor_left"] = pd.to_numeric(merged["factor_left"], errors="coerce")
        merged["factor_right"] = pd.to_numeric(merged["factor_right"], errors="coerce")
        merged = merged.dropna(subset=["factor_left", "factor_right"])
        if len(merged) < 2:
            return None

        pearson_values: list[float] = []
        spearman_values: list[float] = []
        for _, group in merged.groupby("date"):
            if len(group) < 2:
                continue
            pearson = group["factor_left"].corr(group["factor_right"], method="pearson")
            spearman = group["factor_left"].corr(group["factor_right"], method="spearman")
            if pd.notna(pearson):
                pearson_values.append(float(pearson))
            if pd.notna(spearman):
                spearman_values.append(float(spearman))

        daily_metrics = [abs(sum(values) / len(values)) for values in (pearson_values, spearman_values) if values]
        if daily_metrics:
            return max(daily_metrics)

        overall_pearson = merged["factor_left"].corr(merged["factor_right"], method="pearson")
        overall_spearman = merged["factor_left"].corr(merged["factor_right"], method="spearman")
        fallback_metrics = [abs(float(value)) for value in (overall_pearson, overall_spearman) if pd.notna(value)]
        return max(fallback_metrics) if fallback_metrics else None

    @classmethod
    def _sanitize_payload(cls, payload: Any) -> Any:
        if isinstance(payload, dict):
            sanitized: dict[str, Any] = {}
            for key, value in payload.items():
                if key.startswith("_"):
                    continue
                if key == "factor_snapshot":
                    continue
                sanitized[key] = cls._sanitize_payload(value)
            return sanitized
        if isinstance(payload, list):
            return [cls._sanitize_payload(item) for item in payload]
        return payload

    def _build_continue_request(
        self,
        *,
        config: RDAgentMiningConfig,
        final_round: dict[str, Any],
        known_expressions: list[str],
    ) -> dict[str, Any]:
        feedback = final_round.get("feedback") or {}
        experiment = final_round.get("experiment") or {}
        payload = {
            "objective": feedback.get("next_hypothesis") or config.objective,
            "candidate_universe": list(config.candidate_universe),
            "base_factors": list(experiment.get("base_factors") or config.base_factors),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "universe": config.universe,
            "benchmark": config.benchmark,
            "max_iterations": max(2, int(config.max_iterations or 1)),
            "candidates_per_iteration": max(2, int(config.candidates_per_iteration or 1)),
            "n_groups": config.n_groups,
            "holding_period": config.holding_period,
            "direction": feedback.get("next_hypothesis") and config.direction or config.direction,
            "neutralize_industry": config.neutralize_industry,
            "neutralize_cap": config.neutralize_cap,
            "continuation_of": config.task_id,
            "previous_feedback_id": config.previous_feedback_id or f"{config.task_id}-feedback-{final_round.get('round_index', 0)}",
            "previous_expressions": _dedupe_strings(known_expressions),
            "previous_sota_expressions": [
                str(item.get("expression") or "").strip()
                for item in (final_round.get("sota_candidates") or [])
                if str(item.get("expression") or "").strip()
            ],
            "acceptance_policy": dict(config.acceptance_policy or {}),
        }
        return {
            "objective": payload["objective"],
            "summary": feedback.get("reason") or "根据上一轮反馈继续优化。",
            "payload": payload,
        }

    @staticmethod
    def _collect_top_factors(rounds: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for round_item in rounds:
            for candidate in list(round_item.get("candidates") or round_item.get("all_factors") or []):
                expression = str(candidate.get("expression") or "").strip()
                if not expression:
                    continue
                key = _normalize_expression_key(expression)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
        merged.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return merged[: max(int(limit or 0), 0)]

    @staticmethod
    def _raise_if_cancelled(config: RDAgentMiningConfig) -> None:
        if config.cancel_check is not None:
            config.cancel_check()


def _average_score(candidates: list[dict[str, Any]]) -> float:
    scores = [float(item.get("score") or 0.0) for item in candidates if item.get("score") is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _normalize_expression_key(expression: str) -> str:
    return "".join(ch.lower() for ch in expression if not ch.isspace())


def _tokenize_expression(expression: str) -> list[str]:
    normalized = _normalize_expression_key(expression)
    token = []
    tokens: list[str] = []
    for ch in normalized:
        if ch.isalnum() or ch == "_":
            token.append(ch)
            continue
        if token:
            tokens.append("".join(token))
            token.clear()
        tokens.append(ch)
    if token:
        tokens.append("".join(token))
    return tokens
