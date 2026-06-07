"""
数据管理API路由
"""
import math
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.api.dependencies import service_attr

router = APIRouter()


def _sanitize_records(records: list[dict]) -> list[dict]:
    """把 NaN / inf 清洗成可 JSON 序列化的值。"""
    sanitized: list[dict] = []
    for record in records:
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                cleaned[key] = None
            else:
                cleaned[key] = value
        sanitized.append(cleaned)
    return sanitized


# ========== 数据模型 ==========

class StockDataRequest(BaseModel):
    """获取股票数据请求"""
    code: str
    start_date: str
    end_date: str


class BenchmarkRequest(BaseModel):
    """获取 benchmark 数据请求"""
    benchmark: str = "hs300"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ========== API端点 ==========

@router.get("/stock/{code}")
async def get_stock_data(
    code: str,
    start_date: str,
    end_date: str
):
    """
    获取股票数据

    参数:
    - code: 股票代码
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    """
    try:
        data_service = service_attr("backend.data.service", "data_service")
        data = data_service.get_stock_data(
            stock_code=code,
            start_date=start_date,
            end_date=end_date
        )

        if data is None or len(data) == 0:
            raise HTTPException(status_code=404, detail="未获取到数据")

        # 转换为JSON格式
        data_dict = {
            "index": data.index.astype(str).tolist(),
            "columns": data.columns.tolist(),
            "data": data.values.tolist()
        }

        return {
            "success": True,
            "data": data_dict
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def get_data_sources():
    """获取数据源能力与优先级。"""
    try:
        data_service = service_attr("backend.data.service", "data_service")
        return {
            "success": True,
            "data": data_service.get_supported_data_sources(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/universe/{name}")
async def get_stock_universe(name: str, date: Optional[str] = None):
    """获取股票池。"""
    try:
        data_service = service_attr("backend.data.service", "data_service")
        codes = data_service.get_stock_universe(name=name, date=date)
        return {
            "success": True,
            "data": {
                "name": name,
                "date": date,
                "count": len(codes),
                "codes": codes,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark")
async def get_benchmark_data(request: BenchmarkRequest):
    """获取 benchmark 收益率数据。"""
    try:
        data_service = service_attr("backend.data.service", "data_service")
        df = data_service.get_benchmark_returns(
            benchmark=request.benchmark,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return {
            "success": True,
            "data": {
                "benchmark": request.benchmark,
                "records": _sanitize_records(df.to_dict(orient="records")),
                "count": len(df),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计"""
    try:
        data_service = service_attr("backend.data.service", "data_service")
        stats = data_service.get_cache_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/cleanup")
async def cleanup_cache():
    """清理过期缓存"""
    try:
        data_service = service_attr("backend.data.service", "data_service")
        cleaned = data_service.cleanup_cache()
        return {
            "success": True,
            "data": {
                "cleaned_count": cleaned
            },
            "message": f"已清理 {cleaned} 个过期缓存"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """清空全部缓存"""
    try:
        data_service = service_attr("backend.data.service", "data_service")
        cleared = data_service.clear_cache()
        return {
            "success": True,
            "data": {
                "cleared_count": cleared
            },
            "message": f"已清空 {cleared} 个缓存"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
