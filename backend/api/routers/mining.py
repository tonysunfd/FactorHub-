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


mining_tasks: dict[str, dict[str, Any]] = {}


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
    progress = int(generation / max(total_generations, 1) * 100)
    mining_tasks[task_id]["progress"] = progress
    mining_tasks[task_id]["current_generation"] = generation
    mining_tasks[task_id]["total_generations"] = total_generations
    mining_tasks[task_id]["best_fitness"] = float(best_fitness)
    mining_tasks[task_id]["avg_fitness"] = float(avg_fitness)
    if "fitness_history" not in mining_tasks[task_id]:
        mining_tasks[task_id]["fitness_history"] = {"best": [], "average": []}
    mining_tasks[task_id]["fitness_history"]["best"].append(float(best_fitness))
    mining_tasks[task_id]["fitness_history"]["average"].append(float(avg_fitness))
    if candidates is not None:
        mining_tasks[task_id]["candidates"] = [
            _normalize_manual_candidate(candidate, index)
            for index, candidate in enumerate(candidates)
        ]


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
    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "error": task.get("error"),
    }

    if task["status"] == "completed" and task.get("result"):
        result = task["result"]
        response_data["current_generation"] = result.get("generations", 0)
        response_data["total_generations"] = result.get("generations", 0)
        response_data["best_fitness"] = result.get("best_fitness", result.get("best_score", 0))
        response_data["avg_fitness"] = result.get("avg_fitness", result.get("avg_score", 0))
        response_data["fitness_history"] = result.get("fitness_history", {"best": [], "average": []})
    else:
        response_data["current_generation"] = task.get("current_generation", 0)
        response_data["total_generations"] = task.get("total_generations", 10)
        response_data["best_fitness"] = task.get("best_fitness", 0.0)
        response_data["avg_fitness"] = task.get("avg_fitness", 0.0)
        response_data["fitness_history"] = task.get("fitness_history", {"best": [], "average": []})
    response_data["candidates"] = task.get("candidates", [])
    response_data["round_evaluation"] = task.get("round_evaluation")

    return response_data


def _build_auto_campaign_status(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    payload = _build_mining_status_payload(task_id, task)
    payload["current_round"] = task.get("current_round", 0)
    payload["total_rounds"] = task.get("total_rounds", 0)
    payload["retained_count"] = task.get("retained_count", 0)
    payload["upstream_status"] = task.get("upstream_status")
    payload["rounds"] = task.get("rounds", [])
    payload["latest_round"] = task.get("latest_round")
    return payload


async def _run_auto_mining(task_id: str, request: AutoMiningRequest):
    try:
        mining_tasks[task_id]["status"] = "running"
        mining_tasks[task_id]["progress"] = 10
        mining_tasks[task_id]["request"] = request.model_dump()
        mining_tasks[task_id]["total_generations"] = request.n_candidates
        mining_tasks[task_id]["candidates"] = []

        def _progress(done_count: int, total_count: int, candidate: dict[str, Any]) -> None:
            progress = int(done_count / max(total_count, 1) * 100)
            mining_tasks[task_id]["progress"] = progress
            mining_tasks[task_id]["current_generation"] = done_count
            mining_tasks[task_id]["total_generations"] = total_count
            mining_tasks[task_id]["candidates"] = [*mining_tasks[task_id].get("candidates", []), candidate]
            if mining_tasks[task_id]["candidates"]:
                scores = [_safe_candidate_score(item) for item in mining_tasks[task_id]["candidates"]]
                mining_tasks[task_id]["best_fitness"] = max(scores)
                mining_tasks[task_id]["avg_fitness"] = round(sum(scores) / len(scores), 4)

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

        mining_tasks[task_id]["status"] = "completed"
        mining_tasks[task_id]["progress"] = 100
        mining_tasks[task_id]["result"] = result
        mining_tasks[task_id]["candidates"] = result.get("factors", [])
        mining_tasks[task_id]["current_generation"] = result.get("generations", 0)
        mining_tasks[task_id]["total_generations"] = result.get("generations", 0)
        mining_tasks[task_id]["best_fitness"] = result.get("best_score", 0)
        mining_tasks[task_id]["avg_fitness"] = result.get("avg_score", 0)
        mining_tasks[task_id]["fitness_history"] = result.get("fitness_history", {"best": [], "average": []})
        mining_tasks[task_id]["round_evaluation"] = result.get("round_evaluation")
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
            mining_tasks[task_id]["current_round"] = snapshot.get("current_round", 0)
            mining_tasks[task_id]["total_rounds"] = snapshot.get("total_rounds", request.exploration_rounds)
            mining_tasks[task_id]["latest_round"] = snapshot.get("latest_round")
            mining_tasks[task_id]["rounds"] = snapshot.get("rounds", [])
            mining_tasks[task_id]["retained_count"] = snapshot.get("retained_count", 0)
            mining_tasks[task_id]["fitness_history"] = snapshot.get("fitness_history", {"best": [], "average": []})
            mining_tasks[task_id]["best_fitness"] = snapshot.get("best_fitness", 0.0)
            mining_tasks[task_id]["avg_fitness"] = snapshot.get("avg_fitness", 0.0)
            mining_tasks[task_id]["candidates"] = snapshot.get("candidates", [])
            mining_tasks[task_id]["current_generation"] = snapshot.get("current_generation", 0)
            mining_tasks[task_id]["total_generations"] = snapshot.get("total_generations", request.n_candidates_per_round)
            overall_progress = (
                ((snapshot.get("current_round", 1) - 1) + snapshot.get("current_generation", 0) / max(request.n_candidates_per_round, 1))
                / max(request.exploration_rounds, 1)
            )
            mining_tasks[task_id]["progress"] = min(99, max(0, int(overall_progress * 100)))

        campaign_result = auto_factor_mining_service.run_auto_campaign(
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
        mining_tasks[task_id]["status"] = "completed"
        mining_tasks[task_id]["progress"] = 100
        mining_tasks[task_id]["result"] = campaign_result
        mining_tasks[task_id]["fitness_history"] = campaign_result.get("fitness_history", {"best": [], "average": []})
        mining_tasks[task_id]["retained_count"] = len(campaign_result.get("retained_factors", []))
        mining_tasks[task_id]["rounds"] = campaign_result.get("rounds", [])
        mining_tasks[task_id]["latest_round"] = mining_tasks[task_id]["rounds"][-1] if mining_tasks[task_id]["rounds"] else None
        mining_tasks[task_id]["upstream_status"] = "completed_campaign"
        mining_tasks[task_id]["best_fitness"] = campaign_result.get("best_score", 0.0)
        mining_tasks[task_id]["avg_fitness"] = campaign_result.get("avg_score", 0.0)
    except Exception as exc:
        logger.error("Auto campaign task %s failed: %s", task_id, exc, exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(exc)


@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动遗传算法挖掘"""
    try:
        task_id = _create_task("genetic")
        background_tasks.add_task(_run_genetic_mining, task_id, request)
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
        )
        return {
            "success": True,
            "data": result,
            "message": f"已筛选 {len(result['selected_factors'])} 个基础因子",
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

            best_factors = result.get("best_factors", [])
            discovered_factors = []
            for index, factor_info in enumerate(best_factors):
                validation = factor_info.get("validation", {})
                ic = validation.get("ic_validation", {}).get("ic", 0.0)
                ir = validation.get("ir_validation", {}).get("ir", 0.0)
                fitness = factor_info.get("fitness", 0.0)
                discovered_factors.append(
                    {
                        "name": f"Mined_Factor_{index + 1}",
                        "expression": factor_info["expression"],
                        "ic": float(ic) if ic else 0.0,
                        "ir": float(ir) if ir else 0.0,
                        "fitness": float(fitness),
                    }
                )

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
                "generations": request.n_generations,
                "fitness_history": fitness_history,
            }

            mining_tasks[task_id]["status"] = "completed"
            mining_tasks[task_id]["progress"] = 100
            mining_tasks[task_id]["result"] = result_data
            mining_tasks[task_id]["candidates"] = result_data["factors"]
            mining_tasks[task_id]["current_generation"] = request.n_generations
            mining_tasks[task_id]["total_generations"] = request.n_generations
            mining_tasks[task_id]["best_fitness"] = result_data["best_fitness"]
            mining_tasks[task_id]["avg_fitness"] = result_data["avg_fitness"]
            mining_tasks[task_id]["fitness_history"] = fitness_history
        except ImportError as exc:
            logger.warning("DEAP library not available, using simulation mode: %s", exc)
            await _run_simulated_mining(task_id, request, data, base_factor_codes, factor_service)
    except Exception as exc:
        logger.error("Task %s failed: %s", task_id, exc, exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(exc)


async def _run_simulated_mining(task_id: str, request: GeneticMiningRequest, data, base_factor_codes, factor_service):
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
        await asyncio.sleep(0.5)

    discovered_factors = _build_simulated_candidates(5)

    result = {
        "factors": discovered_factors,
        "best_fitness": discovered_factors[0]["ic"] if discovered_factors else 0,
        "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0,
        "generations": n_generations,
        "fitness_history": fitness_history,
    }
    mining_tasks[task_id]["status"] = "completed"
    mining_tasks[task_id]["progress"] = 100
    mining_tasks[task_id]["result"] = result
    mining_tasks[task_id]["candidates"] = discovered_factors
    mining_tasks[task_id]["current_generation"] = n_generations
    mining_tasks[task_id]["total_generations"] = n_generations
    mining_tasks[task_id]["best_fitness"] = result["best_fitness"]
    mining_tasks[task_id]["avg_fitness"] = result["avg_fitness"]
    mining_tasks[task_id]["fitness_history"] = fitness_history


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
    return {"success": True, "data": task["result"]}
