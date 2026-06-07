"""
后端依赖获取器
"""
from importlib import import_module
from typing import Any


def service_attr(module_name: str, attr_name: str) -> Any:
    """按需加载重量级服务，减少应用导入阶段的耦合和耗时。"""
    return getattr(import_module(module_name), attr_name)
