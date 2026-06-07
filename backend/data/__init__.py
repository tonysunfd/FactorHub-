"""
标准数据模块入口。
"""

from backend.data.enrichment import (
    ALL_SUPPORTED_VARIABLES,
    DERIVED_VARIABLES,
    FUNDAMENTAL_VARIABLES,
    SPECIAL_VARIABLES,
    MarketDataEnrichmentService,
    market_data_enrichment_service,
)
from backend.data.health import SystemHealthService, system_health_service
from backend.data.providers import AkshareDataProvider, BaoStockDataProvider
from backend.data.registry import (
    DataSourceDefinition,
    DataSourceRegistry,
    build_default_data_source_registry,
)
from backend.data.service import DataService, data_service, get_data_service

__all__ = [
    "ALL_SUPPORTED_VARIABLES",
    "DERIVED_VARIABLES",
    "FUNDAMENTAL_VARIABLES",
    "SPECIAL_VARIABLES",
    "AkshareDataProvider",
    "BaoStockDataProvider",
    "DataService",
    "DataSourceDefinition",
    "DataSourceRegistry",
    "MarketDataEnrichmentService",
    "SystemHealthService",
    "build_default_data_source_registry",
    "data_service",
    "get_data_service",
    "market_data_enrichment_service",
    "system_health_service",
]
