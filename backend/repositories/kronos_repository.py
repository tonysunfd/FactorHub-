"""
Kronos 预测任务数据访问层
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.kronos_task import (
    KronosPredictionItemModel,
    KronosPredictionRunModel,
    KronosTaskModel,
)


class KronosRepository:
    """Kronos 任务与结果持久化仓储"""

    def __init__(self, db: Session):
        self.db = db

    def get_task(self, task_id: str) -> KronosTaskModel | None:
        return self.db.scalar(select(KronosTaskModel).where(KronosTaskModel.task_id == task_id))

    def list_tasks(self, limit: int = 20, task_type: str | None = None) -> list[KronosTaskModel]:
        query = select(KronosTaskModel)
        if task_type:
            query = query.where(KronosTaskModel.task_type == task_type)
        query = query.order_by(KronosTaskModel.updated_at.desc(), KronosTaskModel.id.desc()).limit(limit)
        return list(self.db.scalars(query).all())

    def create_task(self, item: KronosTaskModel) -> KronosTaskModel:
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item

    def update_task(self, item: KronosTaskModel) -> KronosTaskModel:
        self.db.commit()
        self.db.refresh(item)
        return item

    def get_run(self, task_id: str) -> KronosPredictionRunModel | None:
        return self.db.scalar(select(KronosPredictionRunModel).where(KronosPredictionRunModel.task_id == task_id))

    def upsert_run(self, item: KronosPredictionRunModel) -> KronosPredictionRunModel:
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item

    def list_items(self, task_id: str) -> list[KronosPredictionItemModel]:
        query = (
            select(KronosPredictionItemModel)
            .where(KronosPredictionItemModel.task_id == task_id)
            .order_by(KronosPredictionItemModel.stock_code.asc())
        )
        return list(self.db.scalars(query).all())

    def replace_items(self, task_id: str, items: list[KronosPredictionItemModel]) -> list[KronosPredictionItemModel]:
        existing = self.list_items(task_id)
        for item in existing:
            self.db.delete(item)
        for item in items:
            self.db.add(item)
        self.db.commit()
        return self.list_items(task_id)
