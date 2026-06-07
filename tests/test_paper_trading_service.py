from __future__ import annotations

from datetime import datetime

import pandas as pd

from backend.core.database import init_db, get_db_session
from backend.models.backtest import BacktestResultModel
from backend.models.paper import PaperOrderModel, PaperSnapshotModel, PaperStrategyModel
from backend.services.paper_trading_service import PaperTradingService


def _clear_tables() -> None:
    db = get_db_session()
    try:
        db.query(PaperOrderModel).delete()
        db.query(PaperSnapshotModel).delete()
        db.query(PaperStrategyModel).delete()
        db.query(BacktestResultModel).delete()
        db.commit()
    finally:
        db.close()


def _build_backtest(strategy_config: dict) -> int:
    db = get_db_session()
    try:
        item = BacktestResultModel(
            strategy_name="paper_test_factor",
            factor_combination="paper_test_factor",
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_capital=1_000_000,
            final_capital=1_050_000,
            total_return=0.05,
            annual_return=0.1,
            volatility=0.2,
            sharpe_ratio=1.0,
            max_drawdown=0.1,
            equity_curve={},
            quantile_returns={},
            trades_count=2,
            strategy_config=strategy_config,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item.id
    finally:
        db.close()


def _build_strategy(backtest_id: int, strategy_config: dict) -> int:
    db = get_db_session()
    try:
        item = PaperStrategyModel(
            name="paper-test",
            backtest_id=backtest_id,
            strategy_config=strategy_config,
            initial_capital=1_000_000,
            current_value=1_000_000,
            commission_rate=0.0003,
            slippage_rate=0.0,
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item.id
    finally:
        db.close()


def setup_module():
    init_db()
    _clear_tables()


def teardown_function():
    _clear_tables()


def test_settle_strategy_creates_positions_and_next_rebalance(monkeypatch):
    strategy_config = {
        "strategy_class": "factor_rotation",
        "stock_codes": ["000001", "000002"],
        "factor_name": "alpha_a",
        "factor_names": ["alpha_a"],
        "strategy_type": "single_factor",
        "direction": "long",
        "weight_method": "equal_weight",
        "shares_per_trade": 100,
        "holding_period": 7,
        "n_groups": 2,
    }
    backtest_id = _build_backtest(strategy_config)
    strategy_id = _build_strategy(backtest_id, strategy_config)

    def fake_calculate_factors_for_stocks(stock_codes, factor_names, start_date, end_date, rolling_window=None):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 10.5],
                    "close": [10.2, 10.7],
                    "alpha_a": [1.0, 2.0],
                }
            ),
            "000002": pd.DataFrame(
                {
                    "date": dates,
                    "open": [20.0, 19.8],
                    "close": [20.1, 19.7],
                    "alpha_a": [0.5, 0.4],
                }
            ),
        }

    monkeypatch.setattr(
        "backend.services.paper_trading_service.factor_service.calculate_factors_for_stocks",
        fake_calculate_factors_for_stocks,
    )

    db = get_db_session()
    try:
        service = PaperTradingService(db)
        updated = service.settle_strategy(strategy_id, force=True)
        assert updated is not None
        assert updated.last_rebalance_date is not None
        assert updated.next_rebalance_date is not None
        assert updated.current_value > 0

        snapshot = (
            db.query(PaperSnapshotModel)
            .filter(PaperSnapshotModel.strategy_id == strategy_id)
            .order_by(PaperSnapshotModel.id.desc())
            .first()
        )
        assert snapshot is not None
        assert snapshot.positions
        assert "000001" in snapshot.positions

        orders = (
            db.query(PaperOrderModel)
            .filter(PaperOrderModel.strategy_id == strategy_id)
            .all()
        )
        assert len(orders) == 1
        assert orders[0].direction == "buy"
        assert orders[0].stock_code == "000001"
    finally:
        db.close()


def test_settle_strategy_hold_day_marks_to_market(monkeypatch):
    strategy_config = {
        "strategy_class": "factor_rotation",
        "stock_codes": ["000001"],
        "factor_name": "alpha_a",
        "factor_names": ["alpha_a"],
        "strategy_type": "single_factor",
        "direction": "long",
        "weight_method": "equal_weight",
        "shares_per_trade": 100,
        "holding_period": 5,
        "n_groups": 2,
    }
    backtest_id = _build_backtest(strategy_config)
    strategy_id = _build_strategy(backtest_id, strategy_config)

    db = get_db_session()
    try:
        strategy = db.query(PaperStrategyModel).filter(PaperStrategyModel.id == strategy_id).first()
        strategy.last_rebalance_date = "2024-01-02"
        strategy.next_rebalance_date = "2999-01-01"
        db.add(
            PaperSnapshotModel(
                strategy_id=strategy_id,
                date="2024-01-02",
                portfolio_value=1_000_000,
                cash=500_000,
                market_value=500_000,
                daily_return=0.0,
                positions={"000001": {"shares": 1000, "entry_price": 10.0, "entry_date": "2024-01-02"}},
            )
        )
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(
        "backend.services.paper_trading_service.data_service.get_stock_data",
        lambda stock_code, start_date, end_date: pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03"]),
                "close": [12.0],
            }
        ),
    )

    db = get_db_session()
    try:
        service = PaperTradingService(db)
        updated = service.settle_strategy(strategy_id, force=False)
        assert updated is not None
        assert updated.current_value == 512000.0

        snapshots = (
            db.query(PaperSnapshotModel)
            .filter(PaperSnapshotModel.strategy_id == strategy_id)
            .order_by(PaperSnapshotModel.id.asc())
            .all()
        )
        assert len(snapshots) == 2
        assert snapshots[-1].market_value == 12000.0
        assert snapshots[-1].cash == 500000.0
        assert snapshots[-1].portfolio_value == 512000.0
    finally:
        db.close()
