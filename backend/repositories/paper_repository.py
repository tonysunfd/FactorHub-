from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.models.paper import PaperOrderModel, PaperSnapshotModel, PaperStrategyModel


class PaperRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_strategy(self, strategy: PaperStrategyModel) -> PaperStrategyModel:
        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)
        return strategy

    def list_strategies(self, include_stopped: bool = False):
        query = self.db.query(PaperStrategyModel)
        if not include_stopped:
            query = query.filter(PaperStrategyModel.status != "stopped")
        return query.order_by(desc(PaperStrategyModel.created_at)).all()

    def get_strategy(self, strategy_id: int):
        return self.db.query(PaperStrategyModel).filter(PaperStrategyModel.id == strategy_id).first()

    def update_strategy_status(self, strategy_id: int, status: str):
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return None
        strategy.status = status
        self.db.commit()
        self.db.refresh(strategy)
        return strategy

    def get_snapshots(self, strategy_id: int):
        return (
            self.db.query(PaperSnapshotModel)
            .filter(PaperSnapshotModel.strategy_id == strategy_id)
            .order_by(PaperSnapshotModel.date.asc())
            .all()
        )

    def get_latest_snapshot(self, strategy_id: int):
        return (
            self.db.query(PaperSnapshotModel)
            .filter(PaperSnapshotModel.strategy_id == strategy_id)
            .order_by(PaperSnapshotModel.date.desc())
            .first()
        )

    def get_snapshot_by_date(self, strategy_id: int, date: str):
        return (
            self.db.query(PaperSnapshotModel)
            .filter(
                PaperSnapshotModel.strategy_id == strategy_id,
                PaperSnapshotModel.date == date,
            )
            .first()
        )

    def get_orders(self, strategy_id: int, limit: int = 50):
        return (
            self.db.query(PaperOrderModel)
            .filter(PaperOrderModel.strategy_id == strategy_id)
            .order_by(PaperOrderModel.date.desc(), PaperOrderModel.id.desc())
            .limit(limit)
            .all()
        )
