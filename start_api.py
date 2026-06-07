"""
FactorFlow API 服务启动脚本
"""
import os
import uvicorn
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    port = int(os.getenv("FACTORHUB_BACKEND_PORT", "8001"))
    reload_enabled = os.getenv("FACTORHUB_RELOAD", "1") != "0"

    print("=" * 50)
    print("启动 FactorFlow API 服务...")
    print("=" * 50)
    print(f"API地址: http://localhost:{port}")
    print(f"API文档: http://localhost:{port}/docs")
    print(f"自动重载: {'开启' if reload_enabled else '关闭'}")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
        log_level="info"
    )
