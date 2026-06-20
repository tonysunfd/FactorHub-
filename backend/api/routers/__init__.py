"""
API路由模块
"""
import importlib

_ROUTER_NAMES = {
    "factors",
    "analysis",
    "llm",
    "mining",
    "portfolio",
    "backtest",
    "data",
    "paper",
    "wqbrain",
    "research_tools",
    "kronos",
    "kronos_proxy",
}


def __getattr__(name: str):
    if name in _ROUTER_NAMES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)


__all__ = sorted(_ROUTER_NAMES)
