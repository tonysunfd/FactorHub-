"""
标准数据源注册表。

把不同 provider 的能力、优先级和元信息收敛到统一入口，
方便后续继续接入新的数据源。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DataSourceDefinition:
    """单个数据源的标准定义。"""

    key: str
    label: str
    capabilities: tuple[str, ...]
    priority: int

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "capabilities": list(self.capabilities),
            "priority": self.priority,
        }


class DataSourceRegistry:
    """数据源注册表。"""

    def __init__(self, sources: Iterable[DataSourceDefinition]):
        self._sources = {source.key: source for source in sources}

    def list_sources(self) -> list[DataSourceDefinition]:
        return sorted(self._sources.values(), key=lambda item: item.priority)

    def get(self, key: str) -> DataSourceDefinition:
        return self._sources[key]

    def get_sources_for_capability(self, capability: str) -> list[DataSourceDefinition]:
        return [source for source in self.list_sources() if source.supports(capability)]

    def describe(self) -> dict:
        ordered_sources = self.list_sources()
        capability_order = (
            "stock_daily",
            "same_day_refresh",
            "benchmark",
            "universe",
            "fundamentals",
            "dividends",
            "industry",
        )
        capability_map = {
            capability: [source.key for source in self.get_sources_for_capability(capability)]
            for capability in capability_order
            if self.get_sources_for_capability(capability)
        }
        return {
            "primary": ordered_sources[0].key if ordered_sources else None,
            "fallback": [source.key for source in ordered_sources[1:]],
            "capability_priority": capability_map,
            "sources": [source.to_dict() for source in ordered_sources],
        }


def build_default_data_source_registry() -> DataSourceRegistry:
    """构建默认数据源注册表。"""
    return DataSourceRegistry(
        [
            DataSourceDefinition(
                key="akshare",
                label="AKShare",
                capabilities=("stock_daily", "same_day_refresh"),
                priority=10,
            ),
            DataSourceDefinition(
                key="baostock",
                label="BaoStock",
                capabilities=("stock_daily", "benchmark", "universe", "fundamentals", "dividends", "industry"),
                priority=20,
            ),
        ]
    )
