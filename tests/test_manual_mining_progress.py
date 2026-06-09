from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import BackgroundTasks

from backend.api.routers import mining
from backend.api.routers import mining_progress


class _DummyDBSession:
    def close(self) -> None:
        return None


class _DummyFactorRepository:
    def __init__(self, db_session) -> None:
        self.db_session = db_session

    def get_by_name(self, factor_name: str):
        return SimpleNamespace(code=f"{factor_name}_code")


class _FakeGeneticMiningService:
    def __init__(self, task_id: str, captured: dict[str, object]) -> None:
        self.task_id = task_id
        self.captured = captured
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def mine_factors(self):
        assert self._progress_callback is not None
        self._progress_callback(
            1,
            3,
            0.21,
            0.11,
            [
                {
                    "name": "Snapshot_1",
                    "expression": "(Alpha1_code + Alpha2_code)",
                    "fitness": 0.21,
                    "ic": 0.03,
                    "ir": 0.4,
                }
            ],
        )
        self.captured["running_candidates"] = list(mining.mining_tasks[self.task_id].get("candidates", []))
        return {
            "success": True,
            "best_factors": [
                {
                    "expression": "(Alpha1_code + Alpha2_code)",
                    "fitness": 0.21,
                    "score": 82.5,
                    "grade": "A",
                    "report_url": "/api/mining/reports/manual-factor.html",
                    "report_metrics": {"sharpe": 1.18, "cagr": 0.22},
                    "backtest_summary": {
                        "ic_mean": 0.03,
                        "ic_ir": 0.4,
                        "rank_ic_mean": 0.028,
                        "long_short_sharpe": 1.18,
                        "wq_fitness": 0.21,
                    },
                    "wq_brain": {"wq_rating": "A", "wq_fitness": 0.21},
                    "component_scores": {"total_score": 82.5},
                    "anti_overfit": {"score": 75.0, "recommendation": "推荐", "tests": []},
                    "interpretation": {"summary": "manual ok", "weaknesses": [], "next_steps": []},
                    "task_details": {
                        "report_url": "/api/mining/reports/manual-factor.html",
                        "report_metrics": {"sharpe": 1.18, "cagr": 0.22},
                        "backtest_summary": {
                            "ic_mean": 0.03,
                            "ic_ir": 0.4,
                            "rank_ic_mean": 0.028,
                            "long_short_sharpe": 1.18,
                            "wq_fitness": 0.21,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_fitness": 0.21},
                    },
                    "validation": {
                        "ic_validation": {"ic": 0.03},
                        "ir_validation": {"ir": 0.4},
                    },
                }
            ],
            "logbook": [
                {"max": 0.21, "avg": 0.11},
            ],
        }


def test_run_genetic_mining_persists_running_candidates(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sample_data = pd.DataFrame(
        {
            "close": [10.0, 10.5, 10.2, 10.8],
            "open": [9.8, 10.1, 10.0, 10.4],
            "volume": [100, 120, 110, 130],
        }
    )

    monkeypatch.setattr("backend.core.database.get_db_session", lambda: _DummyDBSession())
    monkeypatch.setattr("backend.repositories.factor_repository.FactorRepository", _DummyFactorRepository)
    monkeypatch.setattr(
        "backend.data.service.data_service.get_stock_data",
        lambda stock_code, start_date, end_date: sample_data.copy(),
    )

    task_id = mining._create_task("genetic")
    monkeypatch.setattr(
        "backend.services.genetic_factor_mining_service.create_genetic_mining_service",
        lambda **kwargs: _FakeGeneticMiningService(task_id, captured),
    )
    request = mining.GeneticMiningRequest(
        stock_code="000001.SZ",
        base_factors=["Alpha1", "Alpha2"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        population_size=8,
        n_generations=3,
        cx_prob=0.6,
        mut_prob=0.2,
    )

    asyncio.run(mining._run_genetic_mining(task_id, request))

    task = mining.mining_tasks[task_id]
    assert task["status"] == "completed"
    assert captured["running_candidates"] == [
        {
            "name": "Snapshot_1",
            "expression": "(Alpha1_code + Alpha2_code)",
            "fitness": 0.21,
            "ic": 0.03,
            "ir": 0.4,
        }
    ]
    assert task["current_generation"] == 3
    assert task["candidates"] == [
        {
            "name": "Mined_Factor_1",
            "expression": "(Alpha1_code + Alpha2_code)",
            "ic": 0.03,
            "ir": 0.4,
            "fitness": 0.21,
            "score": 82.5,
            "grade": "A",
            "report_url": "/api/mining/reports/manual-factor.html",
            "report_metrics": {"sharpe": 1.18, "cagr": 0.22},
            "backtest_summary": {
                "ic_mean": 0.03,
                "ic_ir": 0.4,
                "rank_ic_mean": 0.028,
                "long_short_sharpe": 1.18,
                "wq_fitness": 0.21,
            },
            "wq_brain": {"wq_rating": "A", "wq_fitness": 0.21},
            "component_scores": {"total_score": 82.5},
            "anti_overfit": {"score": 75.0, "recommendation": "推荐", "tests": []},
            "interpretation": {"summary": "manual ok", "weaknesses": [], "next_steps": []},
            "task_details": {
                "report_url": "/api/mining/reports/manual-factor.html",
                "report_metrics": {"sharpe": 1.18, "cagr": 0.22},
                "backtest_summary": {
                    "ic_mean": 0.03,
                    "ic_ir": 0.4,
                    "rank_ic_mean": 0.028,
                    "long_short_sharpe": 1.18,
                    "wq_fitness": 0.21,
                },
                "wq_brain": {"wq_rating": "A", "wq_fitness": 0.21},
            },
        }
    ]
    assert task["result"]["factors"] == [
        {
            "name": "Mined_Factor_1",
            "expression": "(Alpha1_code + Alpha2_code)",
            "ic": 0.03,
            "ir": 0.4,
            "fitness": 0.21,
            "score": 82.5,
            "grade": "A",
            "report_url": "/api/mining/reports/manual-factor.html",
            "report_metrics": {"sharpe": 1.18, "cagr": 0.22},
            "backtest_summary": {
                "ic_mean": 0.03,
                "ic_ir": 0.4,
                "rank_ic_mean": 0.028,
                "long_short_sharpe": 1.18,
                "wq_fitness": 0.21,
            },
            "wq_brain": {"wq_rating": "A", "wq_fitness": 0.21},
            "component_scores": {"total_score": 82.5},
            "anti_overfit": {"score": 75.0, "recommendation": "推荐", "tests": []},
            "interpretation": {"summary": "manual ok", "weaknesses": [], "next_steps": []},
            "task_details": {
                "report_url": "/api/mining/reports/manual-factor.html",
                "report_metrics": {"sharpe": 1.18, "cagr": 0.22},
                "backtest_summary": {
                    "ic_mean": 0.03,
                    "ic_ir": 0.4,
                    "rank_ic_mean": 0.028,
                    "long_short_sharpe": 1.18,
                    "wq_fitness": 0.21,
                },
                "wq_brain": {"wq_rating": "A", "wq_fitness": 0.21},
            },
        }
    ]


def test_start_genetic_mining_returns_without_waiting_for_completion(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_run(task_id: str, request) -> None:
        mining.mining_tasks[task_id]["status"] = "running"
        mining.mining_tasks[task_id]["current_generation"] = 1
        mining.mining_tasks[task_id]["total_generations"] = 8
        mining.mining_tasks[task_id]["best_fitness"] = 0.12
        mining.mining_tasks[task_id]["avg_fitness"] = 0.08
        mining.mining_tasks[task_id]["fitness_history"] = {"best": [0.12], "average": [0.08]}
        started.set()
        await release.wait()
        mining.mining_tasks[task_id]["status"] = "completed"

    created_tasks: list[asyncio.Task] = []
    original_create_task = asyncio.create_task

    def tracked_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(mining, "_run_genetic_mining", fake_run)
    monkeypatch.setattr(mining.asyncio, "create_task", tracked_create_task)

    request = mining.GeneticMiningRequest(
        stock_code="000001.SZ",
        base_factors=["Alpha1", "Alpha2"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        population_size=8,
        n_generations=8,
        cx_prob=0.6,
        mut_prob=0.2,
    )

    async def scenario() -> None:
        response = await mining.start_genetic_mining(request, BackgroundTasks())
        task_id = response["data"]["task_id"]
        assert response["data"]["status"] == "pending"

        await asyncio.wait_for(started.wait(), timeout=1)
        status_payload = mining._build_mining_status_payload(task_id, mining.mining_tasks[task_id])
        assert status_payload["status"] == "running"
        assert status_payload["current_generation"] == 1
        assert status_payload["fitness_history"]["best"] == [0.12]

        release.set()
        await asyncio.gather(*created_tasks)

    asyncio.run(scenario())


def test_update_task_from_candidates_appends_fitness_history() -> None:
    task = {"status": "running", "fitness_history": {"best": [], "average": []}}

    mining_progress.update_task_from_candidates(
        task,
        generation=2,
        total_generations=5,
        candidates=[
            {"score": 1.2},
            {"score": 0.8},
        ],
        score_getter=lambda candidate: candidate["score"],
    )

    assert task["progress"] == 40
    assert task["current_generation"] == 2
    assert task["best_fitness"] == 1.2
    assert task["avg_fitness"] == 1.0
    assert task["fitness_history"] == {"best": [1.2], "average": [1.0]}


def test_run_auto_mining_tracks_incremental_fitness_history(monkeypatch) -> None:
    task_id = mining._create_task("auto")

    def fake_run_auto_mining(**kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback(1, 3, {"name": "Factor_A", "score": 1.0})
        progress_callback(2, 3, {"name": "Factor_B", "score": 2.0})
        return {
            "factors": [
                {"name": "Factor_A", "score": 1.0},
                {"name": "Factor_B", "score": 2.0},
            ],
            "best_score": 2.0,
            "avg_score": 1.5,
            "generations": 3,
            "fitness_history": {"best": [1.0, 2.0], "average": [1.0, 1.5]},
            "round_evaluation": {"summary": "ok"},
        }

    monkeypatch.setattr(mining.auto_factor_mining_service, "run_auto_mining", fake_run_auto_mining)

    request = mining.AutoMiningRequest(
        prompt="improve factors",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        n_candidates=3,
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
    )

    asyncio.run(mining._run_auto_mining(task_id, request))

    task = mining.mining_tasks[task_id]
    assert task["status"] == "completed"
    assert task["fitness_history"] == {"best": [1.0, 2.0], "average": [1.0, 1.5]}
    assert task["round_evaluation"] == {"summary": "ok"}
    assert task["candidates"] == [
        {"name": "Factor_A", "score": 1.0},
        {"name": "Factor_B", "score": 2.0},
    ]


def test_run_auto_campaign_uses_to_thread_and_persists_round_state(monkeypatch) -> None:
    task_id = mining._create_task("auto_campaign")
    captured: dict[str, object] = {}

    async def fake_to_thread(func, /, *args, **kwargs):
        captured["func"] = func
        captured["kwargs"] = kwargs
        progress_callback = kwargs["progress_callback"]
        progress_callback(
            {
                "current_round": 1,
                "total_rounds": 2,
                "current_generation": 1,
                "total_generations": 1,
                "best_fitness": 71.0,
                "avg_fitness": 68.0,
                "fitness_history": {"best": [71.0], "average": [68.0]},
                "candidates": [{"name": "Candidate_1", "score": 71.0}],
                "rounds": [],
                "latest_round": {"round_index": 1, "input_base_factors": ["Alpha1"]},
                "retained_count": 0,
            }
        )
        return {
            "rounds": [
                {
                    "round_index": 1,
                    "task_id": "campaign-round-1",
                    "input_base_factors": ["Alpha1"],
                    "continuation_hypothesis": {"target_goal": "优化 L/S Sharpe"},
                    "continuation_feedback": {"reason": "继续优化"},
                    "retained_count": 1,
                    "retained_factors": [{"name": "Candidate_1", "score": 71.0}],
                    "all_factors": [{"name": "Candidate_1", "score": 71.0}],
                }
            ],
            "retained_factors": [{"name": "Candidate_1", "score": 71.0}],
            "final_round_task_id": "campaign-round-1",
            "final_round_result": {"factors": [{"name": "Candidate_1", "score": 71.0}]},
            "best_score": 71.0,
            "avg_score": 68.0,
            "fitness_history": {"best": [71.0], "average": [68.0]},
            "selection_mode": "any",
            "retention_filter": {"match_mode": "any", "score_min": 0},
        }

    monkeypatch.setattr(mining.asyncio, "to_thread", fake_to_thread)

    request = mining.AutoMiningCampaignRequest(
        prompt="improve factors",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "any", "score_min": 0},
    )

    asyncio.run(mining._run_auto_campaign(task_id, request))

    task = mining.mining_tasks[task_id]
    assert getattr(captured["func"], "__name__", "") == "run_auto_campaign"
    assert captured["kwargs"]["prompt"] == "improve factors"
    assert task["status"] == "completed"
    assert task["current_round"] == 1
    assert task["retained_count"] == 1
    assert task["latest_round"]["input_base_factors"] == ["Alpha1"]
    assert task["rounds"][0]["continuation_hypothesis"]["target_goal"] == "优化 L/S Sharpe"
    assert task["fitness_history"] == {"best": [71.0], "average": [68.0]}


def test_auto_campaign_status_and_results_strip_control_chars(monkeypatch) -> None:
    task_id = mining._create_task("auto_campaign")
    dirty_expression = "rank(close)\x00\x1f"
    mining.mining_tasks[task_id].update(
        {
            "status": "completed",
            "progress": 100,
            "current_round": 1,
            "total_rounds": 1,
            "retained_count": 1,
            "rounds": [
                {
                    "round_index": 1,
                    "task_id": "campaign-round-1",
                    "all_factors": [{"name": "Candidate_1", "expression": dirty_expression}],
                    "retained_factors": [{"name": "Candidate_1", "expression": dirty_expression}],
                }
            ],
            "latest_round": {
                "round_index": 1,
                "task_id": "campaign-round-1",
                "all_factors": [{"name": "Candidate_1", "expression": dirty_expression}],
            },
            "result": {
                "rounds": [
                    {
                        "round_index": 1,
                        "task_id": "campaign-round-1",
                        "all_factors": [{"name": "Candidate_1", "expression": dirty_expression}],
                        "retained_factors": [{"name": "Candidate_1", "expression": dirty_expression}],
                    }
                ],
                "retained_factors": [{"name": "Candidate_1", "expression": dirty_expression}],
                "final_round_result": {"factors": [{"name": "Candidate_1", "expression": dirty_expression}]},
                "best_score": 1.0,
                "avg_score": 1.0,
                "fitness_history": {"best": [1.0], "average": [1.0]},
            },
        }
    )

    status_payload = asyncio.run(mining.get_auto_mining_campaign_status(task_id))
    result_payload = asyncio.run(mining.get_auto_mining_campaign_results(task_id))

    assert "\x00" not in status_payload["data"]["latest_round"]["all_factors"][0]["expression"]
    assert "\x1f" not in status_payload["data"]["latest_round"]["all_factors"][0]["expression"]
    assert "\x00" not in result_payload["data"]["retained_factors"][0]["expression"]
    assert "\x1f" not in result_payload["data"]["final_round_result"]["factors"][0]["expression"]


def test_select_manual_mining_factors_uses_manual_genetic_llm_path(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_select_factors(**kwargs):
        captured.update(kwargs)
        return {
            "selected_factors": ["AlphaClose", "AlphaVolume"],
            "selection_rationale": "已按手动遗传挖掘场景完成筛选。",
            "per_factor_reason": {
                "AlphaClose": "价格因子适合作为单股票 seed factor。",
                "AlphaVolume": "量能因子可增强可解释性。",
            },
            "llm_used": True,
            "llm_call_mode": "live_api",
            "llm_model": "deepseek-chat",
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://127.0.0.1:4000/v1",
            "llm_evidence": {
                "call_mode": "live_api",
                "provider": "openai_compatible",
                "model": "deepseek-chat",
                "base_url": "http://127.0.0.1:4000/v1",
                "response_id": "resp-manual-1",
            },
            "candidate_count": 12,
        }

    monkeypatch.setattr(mining.auto_factor_mining_service, "select_factors", fake_select_factors)

    request = mining.ManualMiningFactorSelectionRequest(
        prompt="为当前股票选择适合手动遗传挖掘的基础因子",
        direction="report_sharpe",
        stock_code="000001",
        start_date="2024-01-01",
        end_date="2024-01-31",
        fitness_objective="sharpe",
        max_factor_count=2,
        candidate_limit=12,
    )

    result = asyncio.run(mining.select_manual_mining_factors(request))

    assert captured["selection_mode"] == "manual_genetic"
    assert captured["universe"] == "single_stock"
    assert captured["benchmark"] == "hs300"
    assert "当前场景：单股票手动遗传挖掘。" in captured["extra_context"]
    assert "当前遗传优化目标：sharpe。" in captured["extra_context"]
    assert "当前目标股票代码：000001。" in captured["extra_context"]
    assert result["success"] is True
    assert result["data"]["selected_factors"] == ["AlphaClose", "AlphaVolume"]
    assert result["data"]["llm_used"] is True
    assert result["data"]["llm_call_mode"] == "live_api"
    assert result["data"]["llm_provider"] == "openai_compatible"
    assert result["data"]["llm_evidence"] == {
        "call_mode": "live_api",
        "provider": "openai_compatible",
        "model": "deepseek-chat",
        "base_url": "http://127.0.0.1:4000/v1",
        "response_id": "resp-manual-1",
    }


def test_select_manual_mining_factors_requires_real_llm_result(monkeypatch) -> None:
    def fake_select_factors(**kwargs):
        return {
            "selected_factors": ["AlphaClose"],
            "selection_rationale": "仅返回筛选结果，但没有真实 LLM 证据。",
            "per_factor_reason": {"AlphaClose": "价格因子。"},
            "llm_used": False,
        }

    monkeypatch.setattr(mining.auto_factor_mining_service, "select_factors", fake_select_factors)

    request = mining.ManualMiningFactorSelectionRequest(
        prompt="为当前股票选择适合手动遗传挖掘的基础因子",
        direction="report_sharpe",
        stock_code="000001",
        start_date="2024-01-01",
        end_date="2024-01-31",
        fitness_objective="sharpe",
        max_factor_count=1,
        candidate_limit=12,
    )

    with pytest.raises(Exception, match="未触发真实 LLM 调用"):
        asyncio.run(mining.select_manual_mining_factors(request))
