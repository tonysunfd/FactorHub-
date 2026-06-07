from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd

from backend.api.routers import mining


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
        }
    ]
    assert task["result"]["factors"] == [
        {
            "name": "Mined_Factor_1",
            "expression": "(Alpha1_code + Alpha2_code)",
            "ic": 0.03,
            "ir": 0.4,
            "fitness": 0.21,
        }
    ]
