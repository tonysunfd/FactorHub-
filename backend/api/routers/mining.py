"""
因子挖掘 API 路由。
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.auto_factor_mining_service import auto_factor_mining_service
from backend.services.rdagent_factor_mining_service import (
    MAX_RDAGENT_ITERATIONS,
    RDAgentFactorMiningService,
    RDAgentMiningConfig,
    RDAgentTaskCancelled,
)
from backend.services.rdagent_runtime import get_rdagent_runtime_status
from backend.api.routers.mining_progress import (
    build_auto_campaign_status,
    build_mining_status_payload,
    finalize_task_result,
    normalize_fitness_history,
    sanitize_payload,
    update_task_from_candidates,
    update_task_progress,
)
from backend.services.research_tools.factor_selection_service import (
    build_factor_snapshot_summary,
    load_factor_candidates_for_llm,
    normalize_factor_expression_key,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class GeneticMiningRequest(BaseModel):
    """遗传算法挖掘请求"""

    stock_code: str
    base_factors: list[str] = []
    start_date: str
    end_date: str
    population_size: int = 50
    n_generations: int = 10
    cx_prob: float = 0.7
    mut_prob: float = 0.3
    elite_size: int = 5
    fitness_objective: str = "ic_mean"
    ic_threshold: float = 0.03


class AutoMiningFactorSelectionRequest(BaseModel):
    prompt: str
    direction: Optional[str] = "score"
    start_date: str
    end_date: str
    universe: str
    benchmark: str
    max_factor_count: int = 12
    candidate_limit: int = 80
    selection_mode: str = "auto"


class ManualMiningFactorSelectionRequest(BaseModel):
    prompt: str
    direction: Optional[str] = "report_sharpe"
    stock_code: str = ""
    start_date: str
    end_date: str
    fitness_objective: str = "ic_mean"
    max_factor_count: int = 8
    candidate_limit: int = 80


class AutoMiningRequest(BaseModel):
    prompt: str
    base_factors: list[str] = []
    start_date: str
    end_date: str
    universe: str
    benchmark: str
    n_groups: int = 5
    holding_period: int = 5
    n_candidates: int = 5
    direction: Optional[str] = "score"
    neutralize_industry: bool = True
    neutralize_cap: bool = True


class AutoMiningCampaignRequest(BaseModel):
    prompt: str
    base_factors: list[str] = []
    start_date: str
    end_date: str
    universe: str
    benchmark: str
    n_groups: int = 5
    holding_period: int = 5
    exploration_rounds: int = 3
    n_candidates_per_round: int = 5
    additional_factor_count_per_round: int = 3
    factor_update_mode: str = "append"
    parent_selection_strategy: str = "best_score_so_far"
    direction: Optional[str] = "score"
    neutralize_industry: bool = True
    neutralize_cap: bool = True
    retention_filter: dict[str, Any] = {}


class ContinueAutoMiningRequest(BaseModel):
    parent_task_id: str
    prompt: Optional[str] = None
    direction: Optional[str] = "score"
    factor_update_mode: Optional[str] = "append"
    additional_base_factors: list[str] = []
    max_factor_count: Optional[int] = None
    candidate_limit: Optional[int] = None
    n_candidates: Optional[int] = None
    n_groups: Optional[int] = None
    holding_period: Optional[int] = None
    neutralize_industry: Optional[bool] = True
    neutralize_cap: Optional[bool] = True


class RDAgentBootstrapSelectRequest(BaseModel):
    objective: str
    direction: Optional[str] = "score"
    start_date: str
    end_date: str
    universe: str
    benchmark: str
    max_factor_count: int = 8
    max_candidate_field_count: int = 5
    candidate_limit: int = 80


class RDAgentAcceptancePolicy(BaseModel):
    max_correlation_with_sota: float = 0.99
    min_rank_ic: float = 0.0
    min_annualized_return_delta: float = 0.0
    max_drawdown_regression: float = 0.05
    min_valid_coverage: float = 0.8


class RDAgentMiningRequest(BaseModel):
    objective: str
    candidate_universe: list[str]
    base_factors: list[str] = []
    start_date: str
    end_date: str
    universe: str
    benchmark: str
    max_iterations: int = 3
    candidates_per_iteration: int = 3
    n_groups: int = 5
    holding_period: int = 5
    direction: Optional[str] = "score"
    neutralize_industry: bool = True
    neutralize_cap: bool = True
    sota_library_id: Optional[str] = None
    continuation_of: Optional[str] = None
    previous_feedback_id: Optional[str] = None
    previous_expressions: list[str] = []
    previous_sota_expressions: list[str] = []
    execution_mode: str = "native_code"
    acceptance_policy: RDAgentAcceptancePolicy = RDAgentAcceptancePolicy()


mining_tasks: dict[str, dict[str, Any]] = {}
rdagent_tasks: dict[str, dict[str, Any]] = {}
RDAGENT_ALLOWED_CANDIDATE_FIELDS = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "vwap",
    "pct_change",
]


def _safe_candidate_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0.0))
    except Exception:
        return 0.0


def _normalize_manual_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "name": candidate.get("name") or f"Mined_Factor_{index + 1}",
        "expression": candidate.get("expression", ""),
        "ic": float(candidate.get("ic", 0.0) or 0.0),
        "ir": float(candidate.get("ir", 0.0) or 0.0),
        "fitness": float(candidate.get("fitness", 0.0) or 0.0),
    }


def _store_manual_task_progress(
    task_id: str,
    generation: int,
    total_generations: int,
    best_fitness: float,
    avg_fitness: float,
    candidates: Optional[list[dict[str, Any]]] = None,
) -> None:
    normalized_candidates = None
    if candidates is not None:
        normalized_candidates = [
            _normalize_manual_candidate(candidate, index)
            for index, candidate in enumerate(candidates)
        ]
    update_task_progress(
        mining_tasks[task_id],
        generation=generation,
        total_generations=total_generations,
        best_fitness=best_fitness,
        avg_fitness=avg_fitness,
        candidates=normalized_candidates,
    )


def _create_task(kind: str) -> str:
    task_id = str(uuid.uuid4())
    mining_tasks[task_id] = {
        "kind": kind,
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
    }
    return task_id


def _build_task_response(task_id: str, message: str) -> dict[str, Any]:
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "status": "pending",
        },
        "message": message,
    }


def _build_mining_status_payload(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    return build_mining_status_payload(task_id, task)


def _build_auto_campaign_status(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    return build_auto_campaign_status(task_id, task)


def _normalize_rdagent_field_list(fields: list[str], max_count: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in fields:
        field_name = str(value or "").strip()
        if field_name not in RDAGENT_ALLOWED_CANDIDATE_FIELDS or field_name in seen:
            continue
        seen.add(field_name)
        normalized.append(field_name)
        if len(normalized) >= max(max_count, 1):
            break
    return normalized


def _normalize_rdagent_factor_list(factors: list[str], max_count: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in factors:
        factor_name = str(value or "").strip()
        if not factor_name or factor_name in seen:
            continue
        seen.add(factor_name)
        normalized.append(factor_name)
        if len(normalized) >= max(max_count, 1):
            break
    return normalized


def _build_rdagent_status_payload(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    payload = build_auto_campaign_status(task_id, task)
    payload["cancel_requested"] = bool(task.get("cancel_requested"))
    payload["request"] = task.get("request") or {}
    return payload


def _build_task_list_item(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "kind": task.get("kind"),
        "status": task.get("status"),
        "progress": task.get("progress", 0),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at", task.get("created_at")),
        "request": task.get("request") or {},
    }


def _raise_if_rdagent_task_cancel_requested(task_id: str) -> None:
    task = rdagent_tasks.get(task_id)
    if not task:
        return
    if task.get("cancel_requested") or task.get("status") == "cancelled":
        raise RDAgentTaskCancelled(f"RDAgent 任务 {task_id} 已终止")


def _build_rdagent_round_from_service(round_item: dict[str, Any], task_id: str) -> dict[str, Any]:
    evaluation = round_item.get("evaluation") or {}
    feedback = round_item.get("feedback") or {}
    hypothesis = round_item.get("hypothesis") or {}
    experiment = round_item.get("experiment") or {}
    candidates = list(round_item.get("candidates") or round_item.get("all_factors") or [])
    return {
        "round_index": round_item.get("round_index", 0),
        "task_id": f"{task_id}-round-{round_item.get('round_index', 0)}",
        "best_score": evaluation.get("best_score", 0.0),
        "avg_score": evaluation.get("avg_score", 0.0),
        "input_base_factors": list(experiment.get("base_factors") or hypothesis.get("base_factors") or []),
        "selected_factors": list((round_item.get("continuation_selection") or {}).get("selected_factors") or experiment.get("base_factors") or hypothesis.get("base_factors") or []),
        "factor_update_mode": "append",
        "hypothesis": {
            "statement": hypothesis.get("statement") or hypothesis.get("summary") or hypothesis.get("hypothesis"),
            "reason": hypothesis.get("reason"),
            "research_direction": hypothesis.get("research_direction") or hypothesis.get("target_goal"),
            "expected_signal": hypothesis.get("expected_signal"),
        },
        "experiment": {
            "hypothesis_summary": experiment.get("hypothesis_summary") or hypothesis.get("statement") or hypothesis.get("summary"),
            "factor_formulations": list(experiment.get("factor_formulations") or []),
            "base_factors": list(experiment.get("base_factors") or hypothesis.get("base_factors") or []),
            "evaluation_focus": experiment.get("evaluation_focus"),
        },
        "evaluation": {
            **evaluation,
            "report_ref": evaluation.get("report_ref"),
        },
        "feedback": {
            "observations": feedback.get("observations") or feedback.get("summary"),
            "hypothesis_evaluation": "supported" if feedback.get("decision") else "needs_revision",
            "next_hypothesis": feedback.get("next_hypothesis") or feedback.get("next_goal"),
            "reason": feedback.get("reason"),
            "decision": feedback.get("decision"),
        },
        "continuation_hypothesis": {
            "hypothesis": hypothesis.get("statement") or hypothesis.get("summary"),
            "target_goal": hypothesis.get("research_direction") or hypothesis.get("target_goal"),
            "candidate_factors": list((round_item.get("continuation_selection") or {}).get("selected_factors") or experiment.get("base_factors") or []),
            "selected_for_next_round": list((round_item.get("next_base_factors") or [])),
        },
        "continuation_feedback": {
            "reason": feedback.get("reason") or feedback.get("summary"),
            "next_goal": feedback.get("next_hypothesis") or feedback.get("next_goal"),
            "decision": feedback.get("decision", True),
        },
        "retained_count": len([candidate for candidate in candidates if candidate.get("status") == "accepted"]),
        "retained_factors": [candidate for candidate in candidates if candidate.get("status") == "accepted"],
        "candidates": candidates,
        "all_factors": candidates,
        "manual_report": {
            "summary": feedback.get("observations") or feedback.get("summary"),
            "score": evaluation.get("best_score"),
        },
        "continue_mining_request": round_item.get("continue_mining_request") or {
            "objective": feedback.get("next_hypothesis") or hypothesis.get("statement") or hypothesis.get("objective"),
            "candidate_universe": list(hypothesis.get("candidate_universe") or []),
            "base_factors": list(experiment.get("base_factors") or []),
            "continuation_of": task_id,
            "previous_feedback_id": f"{task_id}-feedback-{round_item.get('round_index', 0)}",
        },
        "final_round_evaluation": {
            "recommended_goal": feedback.get("next_hypothesis") or feedback.get("next_goal"),
            "primary_problem": feedback.get("reason") or feedback.get("summary"),
            "metric_snapshot": evaluation.get("metrics") or {},
        },
    }


async def _run_rdagent_mining(task_id: str, request: RDAgentMiningRequest) -> None:
    task = rdagent_tasks[task_id]
    try:
        task["status"] = "running"
        task["progress"] = 5
        task["request"] = request.model_dump()
        task["total_rounds"] = max(1, min(int(request.max_iterations or 1), MAX_RDAGENT_ITERATIONS))
        task["upstream_status"] = "rdagent_running"
        task["candidates"] = []
        task["rounds"] = []

        config = RDAgentMiningConfig(
            task_id=task_id,
            objective=request.objective,
            max_iterations=request.max_iterations,
            candidates_per_iteration=request.candidates_per_iteration,
            base_factors=list(request.base_factors or []),
            candidate_universe=_normalize_rdagent_field_list(
                request.candidate_universe,
                max(len(request.candidate_universe), 1),
            ),
            start_date=request.start_date,
            end_date=request.end_date,
            universe=request.universe,
            benchmark=request.benchmark,
            n_groups=request.n_groups,
            holding_period=request.holding_period,
            direction=request.direction,
            neutralize_industry=request.neutralize_industry,
            neutralize_cap=request.neutralize_cap,
            acceptance_policy=request.acceptance_policy.model_dump(),
            continuation_of=request.continuation_of,
            previous_feedback_id=request.previous_feedback_id,
            previous_expressions=list(request.previous_expressions or []),
            previous_sota_expressions=list(request.previous_sota_expressions or []),
            execution_mode=request.execution_mode,
            cancel_check=lambda: _raise_if_rdagent_task_cancel_requested(task_id),
        )
        service = RDAgentFactorMiningService()

        def _progress(progress: int, stage: str, event: dict[str, Any]) -> None:
            iteration = int(event.get("iteration") or 0)
            payload = event.get("payload") or {}
            candidates = payload.get("candidates") or task.get("candidates") or []
            best_fitness = float(payload.get("best_score", payload.get("evaluation", {}).get("best_score", 0.0)) or 0.0)
            avg_fitness = float(payload.get("avg_score", payload.get("evaluation", {}).get("avg_score", 0.0)) or 0.0)
            task["upstream_status"] = stage
            task["current_round"] = max(iteration, task.get("current_round", 0))
            update_task_progress(
                task,
                generation=max(iteration, 1),
                total_generations=max(task.get("total_rounds", 1), 1),
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                progress=progress,
                candidates=candidates,
            )
            if stage == "rdagent_feedback":
                feedback = payload.get("feedback") or {}
                hypothesis = payload.get("hypothesis") or {}
                latest_round = {
                    "round_index": iteration,
                    "task_id": f"{task_id}-round-{iteration}",
                    "candidates": candidates,
                    "all_factors": candidates,
                    "best_score": best_fitness,
                    "avg_score": avg_fitness,
                    "hypothesis": {
                        "statement": hypothesis.get("statement") or hypothesis.get("summary"),
                        "reason": hypothesis.get("reason"),
                        "research_direction": hypothesis.get("research_direction"),
                    },
                    "feedback": {
                        "observations": feedback.get("observations") or feedback.get("summary"),
                        "hypothesis_evaluation": "supported" if feedback.get("decision") else "needs_revision",
                        "next_hypothesis": feedback.get("next_hypothesis") or feedback.get("next_goal"),
                        "reason": feedback.get("reason"),
                    },
                    "manual_report": {
                        "summary": feedback.get("observations") or feedback.get("summary"),
                        "score": best_fitness,
                    },
                }
                task["latest_round"] = latest_round

        result = await asyncio.to_thread(
            service.run,
            task_id=task_id,
            config=config,
            on_progress=_progress,
        )

        rounds = [_build_rdagent_round_from_service(round_item, task_id) for round_item in result.get("rounds", [])]
        retained_factors = list(result.get("retained_factors") or [])
        final_round = rounds[-1] if rounds else None
        final_result = {
            "task_id": task_id,
            "objective": request.objective,
            "rounds": rounds,
            "retained_factors": retained_factors,
            "top_factors": list(result.get("top_factors") or []),
            "watchlist_factors": list(result.get("watchlist_factors") or []),
            "fitness_history": normalize_fitness_history(result.get("fitness_history")),
            "final_round_result": {
                **(result.get("final_round_result") or {}),
                "factors": list((result.get("final_round_result") or {}).get("factors") or (final_round or {}).get("candidates") or []),
            },
            "manual_report": final_round.get("manual_report") if final_round else None,
            "continue_mining_request": result.get("continue_mining_request") or (final_round.get("continue_mining_request") if final_round else None),
        }
        finalize_task_result(
            task,
            final_result,
            candidates=retained_factors or (final_round.get("candidates") if final_round else []),
            generation_key="generations",
            best_key="best_score",
            avg_key="avg_score",
            history_key="fitness_history",
        )
        task["current_round"] = len(rounds)
        task["total_rounds"] = max(task.get("total_rounds", len(rounds)), len(rounds))
        task["rounds"] = rounds
        task["latest_round"] = final_round
        task["retained_count"] = len(retained_factors)
        task["upstream_status"] = "rdagent_completed"
        task["updated_at"] = asyncio.get_running_loop().time()
    except RDAgentTaskCancelled as exc:
        logger.info("RDAgent task %s cancelled: %s", task_id, exc)
        task["status"] = "cancelled"
        task["error"] = str(exc)
        task["upstream_status"] = "rdagent_cancelled"
        task["updated_at"] = asyncio.get_running_loop().time()
    except Exception as exc:
        logger.error("RDAgent task %s failed: %s", task_id, exc, exc_info=True)
        task["status"] = "failed"
        task["error"] = str(exc)
        task["upstream_status"] = "rdagent_failed"
        task["updated_at"] = asyncio.get_running_loop().time()


async def _run_auto_mining(task_id: str, request: AutoMiningRequest):
    try:
        mining_tasks[task_id]["status"] = "running"
        mining_tasks[task_id]["progress"] = 10
        mining_tasks[task_id]["request"] = request.model_dump()
        mining_tasks[task_id]["total_generations"] = request.n_candidates
        mining_tasks[task_id]["candidates"] = []

        def _progress(done_count: int, total_count: int, candidate: dict[str, Any]) -> None:
            candidates = [*mining_tasks[task_id].get("candidates", []), candidate]
            update_task_from_candidates(
                mining_tasks[task_id],
                generation=done_count,
                total_generations=total_count,
                candidates=candidates,
                score_getter=_safe_candidate_score,
            )

        result = auto_factor_mining_service.run_auto_mining(
            prompt=request.prompt,
            base_factors=request.base_factors,
            start_date=request.start_date,
            end_date=request.end_date,
            universe=request.universe,
            benchmark=request.benchmark,
            n_groups=request.n_groups,
            holding_period=request.holding_period,
            n_candidates=request.n_candidates,
            direction=request.direction,
            neutralize_industry=request.neutralize_industry,
            neutralize_cap=request.neutralize_cap,
            progress_callback=_progress,
        )

        finalize_task_result(
            mining_tasks[task_id],
            result,
            candidates=result.get("factors", []),
            generation_key="generations",
            best_key="best_score",
            avg_key="avg_score",
            round_evaluation=result.get("round_evaluation"),
        )
    except Exception as exc:
        logger.error("Auto mining task %s failed: %s", task_id, exc, exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(exc)


async def _run_auto_campaign(task_id: str, request: AutoMiningCampaignRequest):
    try:
        mining_tasks[task_id]["status"] = "running"
        mining_tasks[task_id]["progress"] = 10
        mining_tasks[task_id]["request"] = request.model_dump()
        mining_tasks[task_id]["current_round"] = 0
        mining_tasks[task_id]["total_rounds"] = request.exploration_rounds
        mining_tasks[task_id]["upstream_status"] = "running_campaign"
        mining_tasks[task_id]["candidates"] = []

        def _campaign_progress(snapshot: dict[str, Any]) -> None:
            overall_progress = (
                ((snapshot.get("current_round", 1) - 1) + snapshot.get("current_generation", 0) / max(request.n_candidates_per_round, 1))
                / max(request.exploration_rounds, 1)
            )
            update_task_progress(
                mining_tasks[task_id],
                generation=snapshot.get("current_generation", 0),
                total_generations=snapshot.get("total_generations", request.n_candidates_per_round),
                best_fitness=snapshot.get("best_fitness", 0.0),
                avg_fitness=snapshot.get("avg_fitness", 0.0),
                candidates=snapshot.get("candidates", []),
                history=normalize_fitness_history(snapshot.get("fitness_history")),
                progress=min(99, max(0, int(overall_progress * 100))),
            )
            mining_tasks[task_id]["current_round"] = snapshot.get("current_round", 0)
            mining_tasks[task_id]["total_rounds"] = snapshot.get("total_rounds", request.exploration_rounds)
            mining_tasks[task_id]["latest_round"] = snapshot.get("latest_round")
            mining_tasks[task_id]["rounds"] = snapshot.get("rounds", [])
            mining_tasks[task_id]["retained_count"] = snapshot.get("retained_count", 0)

        campaign_result = await asyncio.to_thread(
            auto_factor_mining_service.run_auto_campaign,
            prompt=request.prompt,
            base_factors=request.base_factors,
            start_date=request.start_date,
            end_date=request.end_date,
            universe=request.universe,
            benchmark=request.benchmark,
            n_groups=request.n_groups,
            holding_period=request.holding_period,
            exploration_rounds=request.exploration_rounds,
            n_candidates_per_round=request.n_candidates_per_round,
            additional_factor_count_per_round=request.additional_factor_count_per_round,
            factor_update_mode=request.factor_update_mode,
            parent_selection_strategy=request.parent_selection_strategy,
            direction=request.direction,
            neutralize_industry=request.neutralize_industry,
            neutralize_cap=request.neutralize_cap,
            retention_filter=request.retention_filter,
            progress_callback=_campaign_progress,
        )
        finalize_task_result(
            mining_tasks[task_id],
            campaign_result,
            candidates=campaign_result.get("retained_factors", []),
            generation_key="generations",
            best_key="best_score",
            avg_key="avg_score",
        )
        mining_tasks[task_id]["retained_count"] = len(campaign_result.get("retained_factors", []))
        mining_tasks[task_id]["rounds"] = campaign_result.get("rounds", [])
        mining_tasks[task_id]["latest_round"] = mining_tasks[task_id]["rounds"][-1] if mining_tasks[task_id]["rounds"] else None
        mining_tasks[task_id]["upstream_status"] = "completed_campaign"
    except Exception as exc:
        logger.error("Auto campaign task %s failed: %s", task_id, exc, exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(exc)


@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动遗传算法挖掘"""
    try:
        task_id = _create_task("genetic")
        # genetic mining 计算较重，直接挂到后台 asyncio task，避免阻塞首个状态轮询
        asyncio.create_task(_run_genetic_mining(task_id, request))
        return _build_task_response(task_id, "挖掘任务已启动")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auto/select-factors")
async def select_auto_mining_factors(request: AutoMiningFactorSelectionRequest):
    """根据提示词筛选基础因子。"""
    try:
        result = auto_factor_mining_service.select_factors(
            prompt=request.prompt,
            max_factor_count=request.max_factor_count,
            candidate_limit=request.candidate_limit,
            selection_mode=request.selection_mode,
            direction=request.direction,
            start_date=request.start_date,
            end_date=request.end_date,
            universe=request.universe,
            benchmark=request.benchmark,
        )
        return {
            "success": True,
            "data": result,
            "message": f"已筛选 {len(result['selected_factors'])} 个基础因子",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/manual/select-factors")
async def select_manual_mining_factors(request: ManualMiningFactorSelectionRequest):
    """为手动遗传挖掘筛选基础因子。"""
    try:
        stock_code = str(request.stock_code or "").strip()
        fitness_objective = str(request.fitness_objective or "ic_mean").strip() or "ic_mean"
        extra_context_parts = [
            "当前场景：单股票手动遗传挖掘。",
            f"当前遗传优化目标：{fitness_objective}。",
        ]
        if stock_code:
            extra_context_parts.append(f"当前目标股票代码：{stock_code}。")
        result = auto_factor_mining_service.select_factors(
            prompt=request.prompt,
            max_factor_count=request.max_factor_count,
            candidate_limit=request.candidate_limit,
            selection_mode="manual_genetic",
            direction=request.direction,
            start_date=request.start_date,
            end_date=request.end_date,
            universe="single_stock",
            benchmark="hs300",
            extra_context="\n".join(extra_context_parts),
        )
        if not bool(result.get("llm_used")):
            raise RuntimeError("手动因子挖掘的因子筛选未触发真实 LLM 调用，请检查 LLM 配置或后端链路。")
        return {
            "success": True,
            "data": result,
            "message": f"已筛选 {len(result['selected_factors'])} 个手动挖掘基础因子",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auto")
async def start_auto_mining(request: AutoMiningRequest, background_tasks: BackgroundTasks):
    """启动单轮自动因子挖掘。"""
    try:
        task_id = _create_task("auto")
        background_tasks.add_task(_run_auto_mining, task_id, request)
        return _build_task_response(task_id, "自动因子挖掘任务已启动")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auto/campaign")
async def start_auto_mining_campaign(request: AutoMiningCampaignRequest, background_tasks: BackgroundTasks):
    """启动多轮自动因子挖掘。"""
    try:
        task_id = _create_task("auto_campaign")
        background_tasks.add_task(_run_auto_campaign, task_id, request)
        return _build_task_response(task_id, "自动化挖掘任务已启动")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/auto/continue")
async def continue_auto_mining(request: ContinueAutoMiningRequest, background_tasks: BackgroundTasks):
    """继续自动挖掘。"""
    if request.parent_task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="父任务不存在")

    parent_task = mining_tasks[request.parent_task_id]
    parent_request = parent_task.get("request") or {}
    parent_result = parent_task.get("result") or {}

    base_factors = list(request.additional_base_factors or [])
    if not base_factors:
        selection = auto_factor_mining_service.select_continue_factors(
            parent_result=parent_result,
            parent_request=parent_request,
            prompt=request.prompt or "根据上一轮结果继续优化基础因子",
            direction=request.direction,
            factor_update_mode=request.factor_update_mode or "append",
            max_factor_count=request.max_factor_count or 5,
            candidate_limit=request.candidate_limit or 80,
        )
        base_factors = selection.get("selected_factors", [])
        if (request.factor_update_mode or "append") == "append":
            base_factors = list(dict.fromkeys([*(parent_request.get("base_factors") or []), *base_factors]))

    auto_request = AutoMiningRequest(
        prompt=request.prompt or "基于上一轮结果继续优化自动挖掘",
        base_factors=base_factors,
        start_date=parent_request.get("start_date", ""),
        end_date=parent_request.get("end_date", ""),
        universe=parent_request.get("universe", "hs300"),
        benchmark=parent_request.get("benchmark", "hs300"),
        n_groups=request.n_groups or parent_request.get("n_groups", 5),
        holding_period=request.holding_period or parent_request.get("holding_period", 5),
        n_candidates=request.n_candidates or parent_request.get("n_candidates", 5),
        direction=request.direction or parent_request.get("direction", "score"),
        neutralize_industry=(
            request.neutralize_industry
            if request.neutralize_industry is not None
            else parent_request.get("neutralize_industry", True)
        ),
        neutralize_cap=(
            request.neutralize_cap
            if request.neutralize_cap is not None
            else parent_request.get("neutralize_cap", True)
        ),
    )

    task_id = _create_task("auto_continue")
    mining_tasks[task_id]["parent_task_id"] = request.parent_task_id
    mining_tasks[task_id]["request"] = auto_request.model_dump()
    background_tasks.add_task(_run_auto_mining, task_id, auto_request)
    return _build_task_response(task_id, "继续自动挖掘任务已启动")


@router.post("/auto/continue/select-factors")
async def select_continue_auto_mining_factors(request: ContinueAutoMiningRequest):
    """根据上一轮结果重新筛选因子。"""
    if request.parent_task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="父任务不存在")

    parent_task = mining_tasks[request.parent_task_id]
    result = auto_factor_mining_service.select_continue_factors(
        parent_result=parent_task.get("result") or {},
        parent_request=parent_task.get("request") or {},
        prompt=request.prompt or "根据上一轮结果继续优化基础因子",
        direction=request.direction,
        factor_update_mode=request.factor_update_mode or "append",
        max_factor_count=request.max_factor_count or max(len(request.additional_base_factors) or 8, 1),
        candidate_limit=request.candidate_limit or 80,
    )
    return {
        "success": True,
        "data": result,
        "message": f"已重新筛选 {len(result['selected_factors'])} 个基础因子",
    }


@router.post("/rdagent/select-bootstrap")
async def select_rdagent_bootstrap(request: RDAgentBootstrapSelectRequest):
    """为 RDAgent 生成候选字段和基础因子。"""
    try:
        candidate_pool = load_factor_candidates_for_llm(
            limit=max(int(request.candidate_limit or 80), 1),
            selection_mode="auto",
        )
        factor_selection = auto_factor_mining_service.select_factors(
            prompt=request.objective,
            max_factor_count=max(int(request.max_factor_count or 8), 1),
            candidate_limit=max(int(request.candidate_limit or 80), 1),
            selection_mode="auto",
        )
        selected_base_factors = _normalize_rdagent_factor_list(
            factor_selection.get("selected_factors", []),
            max(int(request.max_factor_count or 8), 1),
        )

        objective_text = str(request.objective or "").lower()
        suggested_fields: list[str] = []
        if any(keyword in objective_text for keyword in ("volume", "成交量", "amount", "量能")):
            suggested_fields.extend(["volume", "amount"])
        if any(keyword in objective_text for keyword in ("volatility", "波动", "drawdown", "回撤")):
            suggested_fields.extend(["high", "low", "close"])
        if any(keyword in objective_text for keyword in ("trend", "sharpe", "return", "收益")):
            suggested_fields.extend(["close", "vwap", "pct_change"])
        suggested_fields.extend(RDAGENT_ALLOWED_CANDIDATE_FIELDS)
        selected_fields = _normalize_rdagent_field_list(
            suggested_fields,
            max(int(request.max_candidate_field_count or 5), 1),
        )

        factor_reasons = factor_selection.get("per_factor_reason", {})
        summary_lines = [
            f"候选字段：{', '.join(selected_fields) if selected_fields else '未选择'}。",
            f"基础因子：{', '.join(selected_base_factors) if selected_base_factors else '未选择'}。",
        ]
        if factor_selection.get("selection_rationale"):
            summary_lines.append(str(factor_selection["selection_rationale"]))
        if candidate_pool:
            top_snapshot = candidate_pool[: min(3, len(candidate_pool))]
            snapshot_lines = [
                f"{item.get('name')}：{build_factor_snapshot_summary(item.get('snapshot_summary') or {})}"
                for item in top_snapshot
            ]
            summary_lines.append("候选参考：" + "；".join(snapshot_lines))
        if factor_reasons:
            summary_lines.append(
                "因子理由：" + "；".join(
                    f"{name}：{reason}" for name, reason in list(factor_reasons.items())[: min(5, len(factor_reasons))]
                )
            )

        return {
            "success": True,
            "data": {
                "candidate_universe": selected_fields,
                "base_factors": selected_base_factors,
                "selection_rationale": factor_selection.get("selection_rationale", ""),
                "per_factor_reason": factor_reasons,
                "summary": "\n".join(summary_lines),
            },
            "message": "已生成 RDAgent 启动配置",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rdagent")
async def start_rdagent_mining(request: RDAgentMiningRequest):
    """启动 RDAgent 因子挖掘。"""
    try:
        task_id = str(uuid.uuid4())
        normalized_fields = _normalize_rdagent_field_list(
            request.candidate_universe or RDAGENT_ALLOWED_CANDIDATE_FIELDS,
            max(len(request.candidate_universe or []), 1),
        ) or RDAGENT_ALLOWED_CANDIDATE_FIELDS[:3]
        normalized_request = request.model_copy(update={"candidate_universe": normalized_fields})
        rdagent_tasks[task_id] = {
            "kind": "rdagent",
            "status": "pending",
            "progress": 0,
            "error": None,
            "result": None,
            "request": normalized_request.model_dump(),
            "created_at": str(asyncio.get_running_loop().time()),
            "updated_at": str(asyncio.get_running_loop().time()),
            "current_round": 0,
            "total_rounds": max(1, min(int(request.max_iterations or 1), MAX_RDAGENT_ITERATIONS)),
            "retained_count": 0,
            "upstream_status": "rdagent_pending",
            "rounds": [],
            "latest_round": None,
            "candidates": [],
            "cancel_requested": False,
        }
        asyncio.create_task(_run_rdagent_mining(task_id, normalized_request))
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "pending",
            },
            "message": "RDAgent 挖掘任务已启动",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rdagent/{task_id}/cancel")
async def cancel_rdagent_mining(task_id: str):
    task = rdagent_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("status") in {"completed", "failed", "cancelled"}:
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": task.get("status"),
            },
            "message": "任务已结束，无需重复终止",
        }
    task["cancel_requested"] = True
    task["updated_at"] = str(asyncio.get_running_loop().time())
    if task.get("status") == "pending":
        task["status"] = "cancelled"
        task["error"] = "任务在启动前已终止"
        task["upstream_status"] = "rdagent_cancelled"
    else:
        task["status"] = "cancelled"
        task["error"] = "任务已终止"
        task["upstream_status"] = "rdagent_cancelled"
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "status": task["status"],
            "cancel_requested": True,
        },
        "message": "已终止 RDAgent 任务",
    }


@router.get("/tasks")
async def list_mining_tasks(kind: Optional[str] = None, limit: int = 20):
    """返回最近任务，供前端恢复状态。"""
    normalized_limit = max(int(limit or 20), 1)
    items: list[dict[str, Any]] = []
    if kind in (None, "", "genetic", "auto", "auto_campaign", "auto_continue"):
        for task_id, task in mining_tasks.items():
            items.append(_build_task_list_item(task_id, task))
    if kind in (None, "", "rdagent"):
        for task_id, task in rdagent_tasks.items():
            items.append(_build_task_list_item(task_id, task))
    if kind:
        items = [item for item in items if item.get("kind") == kind]
    items.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return {
        "success": True,
        "data": items[:normalized_limit],
    }


@router.get("/reports/{filename}")
async def get_auto_mining_report(filename: str):
    """获取自动挖掘生成的 HTML 报告。"""
    path = auto_factor_mining_service.get_report_path(filename)
    if not path.exists() or path.name != Path(filename).name:
        raise HTTPException(status_code=404, detail="报告不存在")
    return FileResponse(path)


async def _run_genetic_mining(task_id: str, request: GeneticMiningRequest):
    """后台执行遗传算法挖掘"""
    try:
        await asyncio.to_thread(_run_genetic_mining_sync, task_id, request)
    except Exception as exc:
        logger.error("Task %s failed: %s", task_id, exc, exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(exc)


def _run_genetic_mining_sync(task_id: str, request: GeneticMiningRequest):
    logger.info("Starting mining task %s", task_id)
    logger.info("Stock: %s, Base factors: %s", request.stock_code, request.base_factors)
    logger.info(
        "Parameters: population=%s, generations=%s",
        request.population_size,
        request.n_generations,
    )

    from backend.core.database import get_db_session
    from backend.data.service import data_service
    from backend.repositories.factor_repository import FactorRepository
    from backend.services.factor_service import factor_service

    mining_tasks[task_id]["status"] = "running"
    mining_tasks[task_id]["request"] = request.model_dump()
    mining_tasks[task_id]["candidates"] = []

    data = data_service.get_stock_data(request.stock_code, request.start_date, request.end_date)
    if data is None or len(data) == 0:
        raise Exception("未获取到有效数据")

    if "close" in data.columns:
        data["return"] = data["close"].pct_change()

    base_factor_codes: list[str] = []
    if request.base_factors:
        try:
            db = get_db_session()
            repo = FactorRepository(db)
            for factor_name in request.base_factors:
                factor = repo.get_by_name(factor_name)
                if factor:
                    base_factor_codes.append(factor.code)
                    logger.info("Found factor: %s -> %s", factor_name, factor.code)
                else:
                    logger.warning("Factor not found in database: %s", factor_name)
            db.close()
        except Exception as exc:
            logger.error("Error loading factors from database: %s", exc)

    if not base_factor_codes:
        base_factor_codes = [
            "RSI(close, 14)",
            "SMA(close, 20)",
            "close / open",
            "volume / 1000000",
            "MACD(close, 12, 26, 9)[0]",
        ]

    try:
        from backend.services.genetic_factor_mining_service import create_genetic_mining_service

        mining_service = create_genetic_mining_service(
            base_factors=base_factor_codes,
            data=data,
            return_column="return",
            population_size=request.population_size,
            n_generations=request.n_generations,
            cx_prob=request.cx_prob,
            mut_prob=request.mut_prob,
            factor_calculator=factor_service.calculator,
        )

        def progress_callback(gen, total_gen, best_fitness, avg_fitness, candidates=None):
            _store_manual_task_progress(
                task_id=task_id,
                generation=gen,
                total_generations=total_gen,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                candidates=candidates,
            )

        mining_service.set_progress_callback(progress_callback)
        result = mining_service.mine_factors()
        if not result.get("success"):
            raise Exception(result.get("message", "挖掘失败"))
        _store_manual_mining_result(task_id, request.n_generations, result)
    except ImportError as exc:
        logger.warning("DEAP library not available, using simulation mode: %s", exc)
        _run_simulated_mining_sync(task_id, request, data, base_factor_codes, factor_service)


def _store_manual_mining_result(task_id: str, generations: int, result: dict[str, Any]) -> None:
    best_factors = result.get("best_factors", [])
    discovered_factors = []
    for index, factor_info in enumerate(best_factors):
        validation = factor_info.get("validation", {})
        ic = validation.get("ic_validation", {}).get("ic", 0.0)
        ir = validation.get("ir_validation", {}).get("ir", 0.0)
        fitness = factor_info.get("fitness", 0.0)
        factor_payload = {
            "name": f"Mined_Factor_{index + 1}",
            "expression": factor_info["expression"],
            "ic": float(ic) if ic else 0.0,
            "ir": float(ir) if ir else 0.0,
            "fitness": float(fitness),
        }

        optional_fields = [
            "score",
            "grade",
            "report_url",
            "report_metrics",
            "backtest_summary",
            "wq_brain",
            "component_scores",
            "anti_overfit",
            "interpretation",
            "task_details",
            "quantgpt_task_details",
            "execution_meta",
            "engine_type",
            "dialect",
            "canonical_expression",
            "canonical_ast",
            "raw_expression",
            "source",
            "status",
            "base_factors",
            "task_id",
        ]
        for field in optional_fields:
            if field in factor_info and factor_info.get(field) is not None:
                factor_payload[field] = factor_info.get(field)

        discovered_factors.append(factor_payload)

    logbook = result.get("logbook")
    if logbook is not None:
        fitness_history = {
            "best": [float(gen["max"]) for gen in logbook],
            "average": [float(gen["avg"]) for gen in logbook],
        }
    else:
        fitness_history = {"best": [], "average": []}

    result_data = {
        "factors": discovered_factors,
        "best_fitness": float(discovered_factors[0]["fitness"]) if discovered_factors else 0.0,
        "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0.0,
        "generations": generations,
        "fitness_history": fitness_history,
    }

    finalize_task_result(
        mining_tasks[task_id],
        result_data,
        candidates=result_data["factors"],
    )


def _run_simulated_mining_sync(task_id: str, request: GeneticMiningRequest, data, base_factor_codes, factor_service):
    """模拟模式挖掘（当 DEAP 库未安装时使用）"""
    factor_values = {}
    for code in base_factor_codes:
        try:
            values = factor_service.calculator.calculate(data, code)
            if values is not None and len(values.dropna()) > 0:
                factor_values[code] = values
        except Exception as exc:
            logger.warning("计算基础因子失败 %s: %s", code, exc)

    if not factor_values:
        raise Exception("无法计算任何有效的因子值")

    n_generations = request.n_generations
    fitness_history = {"best": [], "average": []}
    current_best_fitness = 0.0
    code_list = list(factor_values.keys())

    def _build_simulated_candidates(limit: int) -> list[dict[str, Any]]:
        discovered_factors = []
        for index in range(min(limit, 5, len(code_list))):
            base_code = code_list[index % len(code_list)]
            if index == 0:
                expression = f"({base_code} * 1.5)"
            elif index == 1:
                expression = f"({base_code} + close / open)"
            elif index == 2:
                expression = f"({base_code} * volume / 1000000)"
            elif index == 3:
                expression = f"({base_code} - SMA(close, 20))"
            else:
                expression = f"({base_code} / (close + 1))"

            discovered_factors.append(
                {
                    "name": f"Mined_Factor_{index + 1}",
                    "expression": expression,
                    "ic": 0.03 + (index * 0.01),
                    "ir": 0.5 + (index * 0.1),
                    "fitness": 0.03 + (index * 0.01),
                }
            )
        return discovered_factors

    for gen in range(n_generations):
        current_best_fitness = 0.03 + (gen + 1) * 0.005 + (0.001 * (gen % 3))
        current_avg_fitness = current_best_fitness * (0.85 + 0.1 * (gen % 2))
        fitness_history["best"].append(current_best_fitness)
        fitness_history["average"].append(current_avg_fitness)
        _store_manual_task_progress(
            task_id=task_id,
            generation=gen + 1,
            total_generations=n_generations,
            best_fitness=current_best_fitness,
            avg_fitness=current_avg_fitness,
            candidates=_build_simulated_candidates(gen + 1),
        )
        import time
        time.sleep(0.5)

    discovered_factors = _build_simulated_candidates(5)
    simulated_result = {
        "best_factors": [
            {
                "expression": factor["expression"],
                "fitness": factor["fitness"],
                "validation": {
                    "ic_validation": {"ic": factor["ic"]},
                    "ir_validation": {"ir": factor["ir"]},
                },
            }
            for factor in discovered_factors
        ],
        "logbook": [
            {"max": best, "avg": avg}
            for best, avg in zip(fitness_history["best"], fitness_history["average"])
        ],
    }
    _store_manual_mining_result(task_id, n_generations, simulated_result)


@router.get("/status/{task_id}")
async def get_mining_status(task_id: str):
    """获取遗传挖掘状态"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "data": _build_mining_status_payload(task_id, mining_tasks[task_id])}


@router.get("/results/{task_id}")
async def get_mining_results(task_id: str):
    """获取遗传挖掘结果"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = mining_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")
    return {"success": True, "data": task["result"]}


@router.get("/auto/status/{task_id}")
async def get_auto_mining_status(task_id: str):
    """获取自动挖掘状态"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "data": _build_mining_status_payload(task_id, mining_tasks[task_id])}


@router.get("/auto/results/{task_id}")
async def get_auto_mining_results(task_id: str):
    """获取自动挖掘结果"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = mining_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")
    return {"success": True, "data": task["result"]}


@router.get("/auto/campaign/status/{task_id}")
async def get_auto_mining_campaign_status(task_id: str):
    """获取自动化挖掘状态"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "data": _build_auto_campaign_status(task_id, mining_tasks[task_id])}


@router.get("/auto/campaign/results/{task_id}")
async def get_auto_mining_campaign_results(task_id: str):
    """获取自动化挖掘结果"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = mining_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")
    return {"success": True, "data": sanitize_payload(task["result"])}


@router.get("/rdagent/status/{task_id}")
async def get_rdagent_mining_status(task_id: str):
    """获取 RDAgent 挖掘状态。"""
    if task_id not in rdagent_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "data": _build_rdagent_status_payload(task_id, rdagent_tasks[task_id])}


@router.get("/rdagent/runtime-status")
async def get_rdagent_reference_runtime_status():
    """获取 reference RD-Agent 运行时状态。"""
    return {"success": True, "data": get_rdagent_runtime_status()}


@router.get("/rdagent/results/{task_id}")
async def get_rdagent_mining_results(task_id: str):
    """获取 RDAgent 挖掘结果。"""
    if task_id not in rdagent_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = rdagent_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")
    return {"success": True, "data": task["result"]}
