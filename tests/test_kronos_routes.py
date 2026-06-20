from __future__ import annotations

import asyncio

from backend.api.routers import kronos


class _FakeKronosTaskService:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, dict]] = []

    def list_data_files(self):
        return [{"key": "factorhub_stock", "name": "Factorhub 单票"}]

    def load_dataset(self, payload):
        return type(
            "Prepared",
            (),
            {
                "source_type": payload.get("source_type", "factorhub_stock"),
                "title": payload.get("stock_code") or payload.get("universe") or "preview",
                "data_preview": {"rows": 123, "columns": ["open", "high", "low", "close"]},
            },
        )()

    def enqueue_task(self, task_type: str, payload: dict):
        self.enqueued.append((task_type, payload))
        return {"task_id": f"{task_type}-1", "task_type": task_type, "status": "pending", "request_payload": payload}

    def list_tasks(self, limit: int = 20, task_type: str | None = None):
        return [{"task_id": "task-1", "task_type": task_type or "single_predict", "status": "completed"}]

    def get_task(self, task_id: str):
        return {"task_id": task_id, "status": "completed", "result_payload": {"forecast_return": 0.12}}

    def cancel_task(self, task_id: str):
        return {"task_id": task_id, "status": "cancelled"}

    def get_runtime_status(self):
        return {"device": "cpu", "phase": "cpu", "queue": {"redis_connected": True}}


def test_list_kronos_data_files(monkeypatch) -> None:
    fake_service = _FakeKronosTaskService()
    monkeypatch.setattr(kronos, "kronos_task_service", fake_service)

    response = asyncio.run(kronos.list_data_files())
    assert response["success"] is True
    assert response["data"][0]["key"] == "factorhub_stock"


def test_create_kronos_predict_task(monkeypatch) -> None:
    fake_service = _FakeKronosTaskService()
    monkeypatch.setattr(kronos, "kronos_task_service", fake_service)

    request = kronos.PredictTaskRequest(
        source_type="factorhub_stock",
        stock_code="000001",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    response = asyncio.run(kronos.create_predict_task(request))
    assert response["success"] is True
    assert response["data"]["task_id"] == "single_predict-1"
    assert fake_service.enqueued[0][0] == "single_predict"


def test_create_batch_predict_task(monkeypatch) -> None:
    fake_service = _FakeKronosTaskService()
    monkeypatch.setattr(kronos, "kronos_task_service", fake_service)

    request = kronos.BatchPredictTaskRequest(
        source_type="factorhub_universe",
        universe="hs300",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    response = asyncio.run(kronos.create_batch_predict_task(request))
    assert response["success"] is True
    assert response["data"]["task_type"] == "batch_predict"


def test_get_kronos_runtime_status(monkeypatch) -> None:
    fake_service = _FakeKronosTaskService()
    monkeypatch.setattr(kronos, "kronos_task_service", fake_service)

    response = asyncio.run(kronos.get_runtime_status())
    assert response["success"] is True
    assert response["data"]["device"] == "cpu"
