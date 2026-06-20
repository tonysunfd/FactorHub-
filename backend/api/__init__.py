"""
FastAPI Web服务
为FactorFlow前端提供REST API
"""

def __getattr__(name: str):
    if name == "app":
        from .main import app

        return app
    raise AttributeError(name)


__all__ = ["app"]
