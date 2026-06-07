from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.core.database import get_db_session
from backend.models.paper import PaperStrategyModel
from backend.repositories.backtest_repository import BacktestRepository
from backend.repositories.paper_repository import PaperRepository
from backend.services.paper_trading_service import PaperTradingService

router = APIRouter()


class CreatePaperStrategyRequest(BaseModel):
    backtest_id: int
    name: str | None = None


class UpdatePaperStrategyRequest(BaseModel):
    status: str


@router.post("/strategies")
async def create_paper_strategy(request: CreatePaperStrategyRequest):
    db = get_db_session()
    try:
        backtest_repo = BacktestRepository()
        paper_repo = PaperRepository(db)

        backtest = backtest_repo.get_by_id(request.backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail="回测记录不存在")

        strategy_config = getattr(backtest, "strategy_config", None)
        if not strategy_config:
            raise HTTPException(status_code=400, detail="该回测未保存结构化策略配置，无法上模拟盘")

        initial_capital = strategy_config.get("initial_capital", backtest.initial_capital or 1_000_000)

        strategy = PaperStrategyModel(
            name=request.name or backtest.strategy_name,
            backtest_id=backtest.id,
            strategy_config=strategy_config,
            initial_capital=initial_capital,
            current_value=initial_capital,
            commission_rate=strategy_config.get("commission_rate", 0.0003),
            slippage_rate=strategy_config.get("slippage", 0.0),
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        created = paper_repo.create_strategy(strategy)
        service = PaperTradingService(db)
        created = service.settle_strategy(created.id, force=True)

        return {"success": True, "data": _strategy_summary(created)}
    finally:
        try:
            backtest_repo.close()
        except Exception:
            pass
        db.close()


@router.get("/strategies")
async def list_paper_strategies(include_stopped: bool = Query(False)):
    db = get_db_session()
    try:
        repo = PaperRepository(db)
        items = repo.list_strategies(include_stopped=include_stopped)
        return {"success": True, "data": [_strategy_summary(s) for s in items]}
    finally:
        db.close()


@router.get("/strategies/{strategy_id}")
async def get_paper_strategy(strategy_id: int):
    db = get_db_session()
    try:
        repo = PaperRepository(db)
        strategy = repo.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="模拟盘策略不存在")

        snapshots = repo.get_snapshots(strategy_id)
        payload = _strategy_summary(strategy)
        payload["strategy_config"] = strategy.strategy_config
        payload["nav_curve"] = [
            {
                "date": s.date,
                "value": s.portfolio_value,
                "daily_return": s.daily_return,
            }
            for s in snapshots
        ]
        return {"success": True, "data": payload}
    finally:
        db.close()


@router.get("/strategies/{strategy_id}/orders")
async def get_paper_orders(strategy_id: int, limit: int = Query(50, ge=1, le=200)):
    db = get_db_session()
    try:
        repo = PaperRepository(db)
        strategy = repo.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="模拟盘策略不存在")
        orders = repo.get_orders(strategy_id, limit=limit)
        return {
            "success": True,
            "data": [
                {
                    "id": o.id,
                    "date": o.date,
                    "stock_code": o.stock_code,
                    "direction": o.direction,
                    "shares": o.shares,
                    "price": o.price,
                    "amount": o.amount,
                    "commission": o.commission,
                    "slippage": o.slippage,
                }
                for o in orders
            ],
        }
    finally:
        db.close()


@router.post("/strategies/{strategy_id}/settle")
async def settle_paper_strategy(strategy_id: int, force: bool = Query(False)):
    db = get_db_session()
    try:
        repo = PaperRepository(db)
        strategy = repo.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="模拟盘策略不存在")
        service = PaperTradingService(db)
        updated = service.settle_strategy(strategy_id, force=force)
        return {"success": True, "data": _strategy_summary(updated)}
    finally:
        db.close()


@router.post("/settle")
async def settle_all_paper_strategies():
    db = get_db_session()
    try:
        service = PaperTradingService(db)
        updated = service.settle_all_active_strategies()
        return {"success": True, "data": [_strategy_summary(item) for item in updated if item]}
    finally:
        db.close()


@router.patch("/strategies/{strategy_id}")
async def update_paper_strategy(strategy_id: int, request: UpdatePaperStrategyRequest):
    if request.status not in ("active", "paused", "stopped"):
        raise HTTPException(status_code=400, detail="状态只能是 active / paused / stopped")

    db = get_db_session()
    try:
        repo = PaperRepository(db)
        updated = repo.update_strategy_status(strategy_id, request.status)
        if not updated:
            raise HTTPException(status_code=404, detail="模拟盘策略不存在")
        return {"success": True, "data": _strategy_summary(updated)}
    finally:
        db.close()


def _strategy_summary(s: PaperStrategyModel) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "status": s.status,
        "current_value": s.current_value,
        "initial_capital": s.initial_capital,
        "total_return": (s.current_value / s.initial_capital - 1) if s.initial_capital else 0,
        "last_rebalance_date": s.last_rebalance_date,
        "next_rebalance_date": s.next_rebalance_date,
        "backtest_id": s.backtest_id,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }
