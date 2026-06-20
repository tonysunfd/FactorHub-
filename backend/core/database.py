"""
数据库连接管理模块
"""
from sqlalchemy import inspect, text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from backend.core.settings import settings


# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# 创建 Session 工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """数据库模型基类"""
    pass


def init_db() -> None:
    """初始化数据库，创建所有表"""
    from backend.models.factor import FactorModel, AnalysisCacheModel
    from backend.models.backtest import BacktestResultModel, TradeRecordModel
    from backend.models.paper import PaperStrategyModel, PaperSnapshotModel, PaperOrderModel
    from backend.models.cache_metadata import CacheMetadataModel
    from backend.models.factor_version import FactorVersionModel
    from backend.models.mining_history import MiningHistoryModel

    Base.metadata.create_all(bind=engine)
    _ensure_schema_compatibility()


def _ensure_schema_compatibility() -> None:
    """对已有 SQLite 库做最小兼容升级，避免开发期因缺字段直接失败。"""
    inspector = inspect(engine)

    try:
        columns = {column["name"] for column in inspector.get_columns("backtest_results")}
    except Exception:
        columns = set()

    if "backtest_results" in inspector.get_table_names() and "strategy_config" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE backtest_results ADD COLUMN strategy_config JSON"))

    try:
        factor_columns = {column["name"] for column in inspector.get_columns("factors")}
    except Exception:
        factor_columns = set()

    if "factors" in inspector.get_table_names():
        factor_alters = {
            "formula_type": "ALTER TABLE factors ADD COLUMN formula_type VARCHAR(20) DEFAULT 'expression'",
            "scope_type": "ALTER TABLE factors ADD COLUMN scope_type VARCHAR(20) DEFAULT 'stock'",
            "target_stock_code": "ALTER TABLE factors ADD COLUMN target_stock_code VARCHAR(32) DEFAULT ''",
            "target_universe": "ALTER TABLE factors ADD COLUMN target_universe VARCHAR(64) DEFAULT ''",
            "origin_type": "ALTER TABLE factors ADD COLUMN origin_type VARCHAR(32) DEFAULT 'manual'",
            "task_metadata": "ALTER TABLE factors ADD COLUMN task_metadata JSON DEFAULT '{}'",
        }
        with engine.begin() as connection:
            for column_name, alter_sql in factor_alters.items():
                if column_name not in factor_columns:
                    connection.execute(text(alter_sql))


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """获取数据库会话（非上下文管理器方式）"""
    return SessionLocal()
