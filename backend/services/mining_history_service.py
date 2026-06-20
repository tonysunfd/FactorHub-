"""
因子挖掘历史记录服务
"""
from __future__ import annotations

from backend.core.database import get_db_session
from backend.models.mining_history import MiningHistoryModel
from backend.repositories.mining_history_repository import MiningHistoryRepository


class MiningHistoryService:
    """因子挖掘历史记录服务"""

    def save_entry(
        self,
        *,
        task_id: str,
        kind: str,
        status: str,
        title: str,
        summary: str,
        request_payload: dict,
        result_payload: dict,
    ) -> dict:
        db = get_db_session()
        try:
            repo = MiningHistoryRepository(db)
            existing = repo.get_by_task_id(task_id)
            if existing:
                existing.kind = kind
                existing.status = status
                existing.title = title
                existing.summary = summary
                existing.request_payload = request_payload or {}
                existing.result_payload = result_payload or {}
                saved = repo.update(existing)
            else:
                saved = repo.create(
                    MiningHistoryModel(
                        task_id=task_id,
                        kind=kind,
                        status=status,
                        title=title,
                        summary=summary,
                        request_payload=request_payload or {},
                        result_payload=result_payload or {},
                    )
                )
            return saved.to_dict()
        finally:
            db.close()

    def list_entries(self, *, limit: int = 20, kind: str | None = None) -> list[dict]:
        db = get_db_session()
        try:
            repo = MiningHistoryRepository(db)
            return [item.to_dict() for item in repo.list(limit=limit, kind=kind)]
        finally:
            db.close()

    def delete_entry(self, history_id: int) -> bool:
        db = get_db_session()
        try:
            repo = MiningHistoryRepository(db)
            return repo.delete(history_id)
        finally:
            db.close()


mining_history_service = MiningHistoryService()
