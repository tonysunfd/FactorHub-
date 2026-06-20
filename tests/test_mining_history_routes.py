from __future__ import annotations

from types import SimpleNamespace

from backend.api.routers import mining


class _FakeMiningHistoryService:
    def __init__(self) -> None:
        self.saved_entries: list[dict] = []
        self.list_response: list[dict] = []
        self.deleted_ids: list[int] = []

    def save_entry(self, **kwargs):
        self.saved_entries.append(kwargs)
        return {"id": len(self.saved_entries), **kwargs}

    def list_entries(self, *, limit: int = 20, kind: str | None = None):
        return self.list_response[:limit]

    def delete_entry(self, history_id: int) -> bool:
        self.deleted_ids.append(history_id)
        return history_id != 404


def test_persist_mining_history_for_rdagent(monkeypatch) -> None:
    fake_service = _FakeMiningHistoryService()
    monkeypatch.setattr(mining, "mining_history_service", fake_service)

    task = {
      "kind": "rdagent",
      "status": "completed",
      "request": {"objective": "提升 score", "candidate_universe": ["close", "volume"]},
      "result": {
          "best_score": 88.5,
          "retained_factors": [{"name": "Alpha_1", "expression": "rank(close)"}],
      },
    }

    mining._persist_mining_history("task-rdagent-1", task)

    assert len(fake_service.saved_entries) == 1
    entry = fake_service.saved_entries[0]
    assert entry["task_id"] == "task-rdagent-1"
    assert entry["kind"] == "rdagent"
    assert entry["status"] == "completed"
    assert entry["title"] == "提升 score"
    assert "保留 1 个候选" in entry["summary"]
    assert entry["request_payload"]["candidate_universe"] == ["close", "volume"]


async def _list_history(kind: str | None = None, limit: int = 20):
    return await mining.list_mining_history(kind=kind, limit=limit)


async def _delete_history(history_id: int):
    return await mining.delete_mining_history(history_id)


def test_list_and_delete_mining_history(monkeypatch) -> None:
    fake_service = _FakeMiningHistoryService()
    fake_service.list_response = [
        {
            "id": 1,
            "task_id": "task-1",
            "kind": "auto",
            "status": "completed",
            "title": "自动挖掘记录",
            "summary": "自动挖掘完成，生成 3 个因子，最佳分数 77.10",
            "request_payload": {"prompt": "优化 score"},
            "result_payload": {"factors": [{"name": "A1", "expression": "rank(close)"}]},
            "created_at": "2026-06-20T10:00:00",
            "updated_at": "2026-06-20T10:00:00",
        }
    ]
    monkeypatch.setattr(mining, "mining_history_service", fake_service)

    list_response = mining.asyncio.run(_list_history(limit=10))
    assert list_response["success"] is True
    assert list_response["data"][0]["task_id"] == "task-1"

    delete_response = mining.asyncio.run(_delete_history(1))
    assert delete_response["success"] is True
    assert fake_service.deleted_ids == [1]
