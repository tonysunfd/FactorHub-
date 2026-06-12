from __future__ import annotations

import asyncio

from backend.api.routers import mining


def test_select_rdagent_bootstrap_returns_candidate_universe(monkeypatch) -> None:
    monkeypatch.setattr(
        mining,
        "load_factor_candidates_for_llm",
        lambda limit, selection_mode: [
            {
                "name": "AlphaVol",
                "snapshot_summary": {
                    "report_metrics": {"sharpe": 1.2},
                    "backtest_summary": {"rank_ic_mean": 0.03},
                },
            }
        ],
    )

    class _FakeAutoMiningService:
        def select_factors(self, **kwargs):
            return {
                "selected_factors": ["AlphaVol", "AlphaTrend"],
                "selection_rationale": "基于目标筛选。",
                "per_factor_reason": {"AlphaVol": "量价配合", "AlphaTrend": "趋势增强"},
            }

    monkeypatch.setattr(mining, "auto_factor_mining_service", _FakeAutoMiningService())

    request = mining.RDAgentBootstrapSelectRequest(
        objective="提升趋势收益并兼顾量能",
        start_date="2024-01-01",
        end_date="2024-03-31",
        universe="hs300",
        benchmark="000300.SH",
        max_factor_count=2,
        max_candidate_field_count=3,
    )

    response = asyncio.run(mining.select_rdagent_bootstrap(request))

    assert response["success"] is True
    assert response["data"]["base_factors"] == ["AlphaVol", "AlphaTrend"]
    assert "close" in response["data"]["candidate_universe"]
    assert "volume" in response["data"]["candidate_universe"]


def test_rdagent_start_status_results_and_tasks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run(task_id: str, request) -> None:
        captured["task_id"] = task_id
        mining.rdagent_tasks[task_id]["status"] = "completed"
        mining.rdagent_tasks[task_id]["progress"] = 100
        mining.rdagent_tasks[task_id]["current_round"] = 2
        mining.rdagent_tasks[task_id]["total_rounds"] = 2
        mining.rdagent_tasks[task_id]["retained_count"] = 1
        mining.rdagent_tasks[task_id]["upstream_status"] = "rdagent_completed"
        mining.rdagent_tasks[task_id]["fitness_history"] = {"best": [72.0, 81.0], "average": [65.0, 74.0]}
        mining.rdagent_tasks[task_id]["candidates"] = [
            {"name": "Candidate_1", "expression": "rank(close)", "status": "accepted"}
        ]
        mining.rdagent_tasks[task_id]["rounds"] = [
            {
                "round_index": 1,
                "task_id": f"{task_id}-round-1",
                "hypothesis": {"statement": "量价共振因子可提升综合分数"},
                "feedback": {"observations": "本轮有 1 个候选通过筛选", "hypothesis_evaluation": "supported"},
                "evaluation": {"best_score": 81.0, "avg_score": 74.0},
                "continuation_selection": {"selected_factors": ["AlphaContinue"]},
                "next_base_factors": ["Alpha1", "AlphaContinue"],
                "candidates": [{"name": "Candidate_1", "expression": "rank(close)", "status": "accepted"}],
                "all_factors": [{"name": "Candidate_1", "expression": "rank(close)", "status": "accepted"}],
            }
        ]
        mining.rdagent_tasks[task_id]["latest_round"] = mining.rdagent_tasks[task_id]["rounds"][0]
        mining.rdagent_tasks[task_id]["result"] = {
            "task_id": task_id,
            "rounds": [
                {
                    **mining.rdagent_tasks[task_id]["rounds"][0],
                    "continuation_hypothesis": {
                        "selected_for_next_round": ["Alpha1", "AlphaContinue"],
                    },
                }
            ],
            "retained_factors": mining.rdagent_tasks[task_id]["candidates"],
            "fitness_history": mining.rdagent_tasks[task_id]["fitness_history"],
            "final_round_result": {"factors": mining.rdagent_tasks[task_id]["candidates"]},
            "continue_mining_request": {
                "objective": "继续优化量价共振因子",
                "payload": {
                    "previous_sota_expressions": ["rank(close)"],
                },
            },
        }

    created_tasks: list[asyncio.Task] = []
    original_create_task = asyncio.create_task

    def tracked_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(mining, "_run_rdagent_mining", fake_run)
    monkeypatch.setattr(mining.asyncio, "create_task", tracked_create_task)

    request = mining.RDAgentMiningRequest(
        objective="提升 score",
        candidate_universe=["close", "volume"],
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-03-31",
        universe="hs300",
        benchmark="000300.SH",
        max_iterations=2,
        candidates_per_iteration=2,
    )

    async def scenario() -> None:
        response = await mining.start_rdagent_mining(request)
        task_id = response["data"]["task_id"]
        await asyncio.gather(*created_tasks)

        status = await mining.get_rdagent_mining_status(task_id)
        result = await mining.get_rdagent_mining_results(task_id)
        tasks = await mining.list_mining_tasks(kind="rdagent", limit=10)

        assert status["data"]["status"] == "completed"
        assert status["data"]["retained_count"] == 1
        assert status["data"]["request"]["max_iterations"] == 2
        assert status["data"]["request"]["candidate_universe"] == ["close", "volume"]
        assert result["data"]["final_round_result"]["factors"][0]["expression"] == "rank(close)"
        assert result["data"]["continue_mining_request"]["objective"] == "继续优化量价共振因子"
        assert result["data"]["continue_mining_request"]["payload"]["previous_sota_expressions"] == ["rank(close)"]
        assert result["data"]["rounds"][0]["continuation_hypothesis"]["selected_for_next_round"] == ["Alpha1", "AlphaContinue"]
        assert tasks["data"][0]["task_id"] == task_id
        assert tasks["data"][0]["kind"] == "rdagent"

    asyncio.run(scenario())
    assert "task_id" in captured


def test_cancel_rdagent_task_marks_cancelled() -> None:
    task_id = "rdagent-cancel-test"
    mining.rdagent_tasks[task_id] = {
        "kind": "rdagent",
        "status": "running",
        "progress": 30,
        "request": {},
        "created_at": "1",
        "updated_at": "1",
        "rounds": [],
    }

    response = asyncio.run(mining.cancel_rdagent_mining(task_id))
    status = asyncio.run(mining.get_rdagent_mining_status(task_id))

    assert response["success"] is True
    assert status["data"]["status"] == "cancelled"
    assert status["data"]["cancel_requested"] is True
