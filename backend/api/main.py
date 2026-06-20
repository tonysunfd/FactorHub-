"""
FastAPI主应用
"""
import asyncio
import importlib
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.settings import settings
from backend.core.database import init_db


def _include_routers(app: FastAPI) -> None:
    """延迟导入路由，减少应用导入阶段的顶层耦合。"""
    factors = importlib.import_module("backend.api.routers.factors")
    analysis = importlib.import_module("backend.api.routers.analysis")
    llm = importlib.import_module("backend.api.routers.llm")
    mining = importlib.import_module("backend.api.routers.mining")
    portfolio = importlib.import_module("backend.api.routers.portfolio")
    backtest = importlib.import_module("backend.api.routers.backtest")
    data = importlib.import_module("backend.api.routers.data")
    paper = importlib.import_module("backend.api.routers.paper")
    paper_factors = importlib.import_module("backend.api.routers.paper_factors")
    research_tools = importlib.import_module("backend.api.routers.research_tools")
    wqbrain = importlib.import_module("backend.api.routers.wqbrain")
    kronos = importlib.import_module("backend.api.routers.kronos")
    kronos_proxy = importlib.import_module("backend.api.routers.kronos_proxy")

    app.include_router(factors.router, prefix="/api/factors", tags=["因子管理"])
    app.include_router(analysis.router, prefix="/api/analysis", tags=["因子分析"])
    app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
    app.include_router(mining.router, prefix="/api/mining", tags=["因子挖掘"])
    app.include_router(portfolio.router, prefix="/api/portfolio", tags=["组合分析"])
    app.include_router(backtest.router, prefix="/api/backtest", tags=["策略回测"])
    app.include_router(data.router, prefix="/api/data", tags=["数据管理"])
    app.include_router(paper.router, prefix="/api/paper", tags=["模拟盘"])
    app.include_router(paper_factors.router, prefix="/api/paper-factors", tags=["论文因子"])
    app.include_router(research_tools.router, prefix="/api/research-tools", tags=["研究工具"])
    app.include_router(wqbrain.router, prefix="/api/wqbrain", tags=["WQ BRAIN"])
    app.include_router(kronos.router, prefix="/api/kronos", tags=["Kronos 集成 API"])
    app.include_router(kronos_proxy.router, prefix="/kronos-ui", tags=["Kronos WebUI Proxy"])


def _load_preset_factors_sync() -> None:
    """延迟加载预置因子，避免阻塞 HTTP 服务监听。"""
    factor_service = importlib.import_module("backend.services.factor_service").factor_service
    factor_service.load_preset_factors()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("启动FastAPI服务...")
    init_db()
    app.state.preset_factors_ready = False
    app.state.preset_factors_error = None

    async def preload() -> None:
        try:
            await asyncio.to_thread(_load_preset_factors_sync)
            app.state.preset_factors_ready = True
            print("数据库和预置因子加载完成")
        except Exception as exc:  # pragma: no cover - 启动诊断
            app.state.preset_factors_error = str(exc)
            print(f"预置因子异步加载失败: {exc}")

    asyncio.create_task(preload())

    yield

    # 关闭时清理
    print("关闭FastAPI服务...")


# 自定义JSON编码器来处理numpy浮点数值
class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# 配置JSON编码器
def jsonable_encoder_with_numpy(obj, *args, **kwargs):
    """处理numpy类型的JSON编码器"""
    try:
        return jsonable_encoder(obj, *args, **kwargs, custom_serializer=lambda x: NumpyJSONEncoder().default(x))
    except:
        return jsonable_encoder(obj, *args, **kwargs)


# 创建FastAPI应用
app = FastAPI(
    title="FactorFlow API",
    description="股票因子分析系统 REST API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=JSONResponse
)

_include_routers(app)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],  # 允许的前端来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================
# Static File Serving (for production)
# ============================================
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "react-antd" / "dist"

if FRONTEND_DIST.exists():
    print(f"Serving static files from: {FRONTEND_DIST}")
    # Mount static assets directory (js, css, images, etc.)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.exception_handler(404)
async def spa_fallback(request, exc):
    """SPA fallback - return index.html for 404 errors (non-API routes)"""
    # Only handle non-API routes for HTML requests
    if FRONTEND_DIST.exists() and not request.url.path.startswith(("/api", "/docs", "/redoc", "/openapi.json")):
        return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found"}
    )


@app.get("/api")
async def api_root():
    """API 根路径"""
    return {
        "message": "FactorFlow API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    preset_ready = getattr(app.state, "preset_factors_ready", False)
    preset_error = getattr(app.state, "preset_factors_error", None)
    return {
        "status": "healthy",
        "preset_factors_ready": preset_ready,
        "preset_factors_error": preset_error,
        "backend_port": os.getenv("FACTORHUB_BACKEND_PORT", "8001"),
        "reload_enabled": os.getenv("FACTORHUB_RELOAD", "1") != "0",
    }


# 全局异常处理
# 覆盖FastAPI的默认JSON响应编码器
@app.on_event("startup")
async def startup_event():
    """应用启动事件 - 覆盖默认JSON编码器"""
    app.json_encoder = NumpyJSONEncoder

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    print(f"[ERROR] 请求错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": str(exc),
            "detail": "服务器内部错误"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
