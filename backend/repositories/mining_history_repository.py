"""
因子挖掘历史记录数据访问层
"""
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.models.mining_history import MiningHistoryModel


class MiningHistoryRepository:
    """因子挖掘历史记录数据访问类"""

    def __init__(self, db: Session):
        self.db = db

    def get_by_task_id(self, task_id: str) -> Optional[MiningHistoryModel]:
        return self.db.scalar(select(MiningHistoryModel).where(MiningHistoryModel.task_id == task_id))

    def list(self, limit: int = 20, kind: str | None = None) -> list[MiningHistoryModel]:
        query = select(MiningHistoryModel)
        if kind:
            query = query.where(MiningHistoryModel.kind == kind)
        query = query.order_by(MiningHistoryModel.updated_at.desc(), MiningHistoryModel.id.desc()).limit(limit)
        return list(self.db.scalars(query).all())

    def create(self, history: MiningHistoryModel) -> MiningHistoryModel:
        self.db.add(history)
        self.db.commit()
        self.db.refresh(history)
        return history

    def update(self, history: MiningHistoryModel) -> MiningHistoryModel:
        self.db.commit()
        self.db.refresh(history)
        return history

    def delete(self, history_id: int) -> bool:
        item = self.db.get(MiningHistoryModel, history_id)
        if not item:
            return False
        self.db.delete(item)
        self.db.commit()
        return True

    def clear_by_task_id(self, task_id: str) -> int:
        result = self.db.execute(delete(MiningHistoryModel).where(MiningHistoryModel.task_id == task_id))
        self.db.commit()
        return int(result.rowcount or 0)
