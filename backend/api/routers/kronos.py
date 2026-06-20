"""
Kronos WebUI 与任务 API
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.kronos_task_service import kronos_task_service

router = APIRouter()


class LoadDataRequest(BaseModel):
    source_type: str = "factorhub_stock"
    stock_code: str | None = None
    universe: str | None = None
    as_of_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    file_path: str | None = None


class PredictTaskRequest(BaseModel):
    source_type: str = "factorhub_stock"
    stock_code: str | None = None
    start_date: str
    end_date: str
    model_name: str = "kronos-base"
    tokenizer_name: str | None = None
    device: str = "cpu"
    lookback: int = 400
    pred_len: int = 120
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1


class BatchPredictTaskRequest(BaseModel):
    source_type: str = "factorhub_universe"
    universe: str = "hs300"
    as_of_date: str | None = None
    start_date: str
    end_date: str
    model_name: str = "kronos-base"
    tokenizer_name: str | None = None
    device: str = "cpu"
    lookback: int = 240
    pred_len: int = 20
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 1
    max_stocks: int = Field(default=30, ge=1, le=500)
    link_backtest: bool = False


class BatchBacktestTaskRequest(BaseModel):
    prediction_task_id: str
    start_date: str
    end_date: str


@router.get("/api/data-files")
async def list_data_files() -> dict[str, Any]:
    return {"success": True, "data": kronos_task_service.list_data_files()}


@router.post("/api/load-data")
async def load_data(request: LoadDataRequest) -> dict[str, Any]:
    try:
        prepared = kronos_task_service.load_dataset(request.model_dump())
        return {
            "success": True,
            "data": {
                "source_type": prepared.source_type,
                "title": prepared.title,
                "preview": prepared.data_preview,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/tasks/predict")
async def create_predict_task(request: PredictTaskRequest) -> dict[str, Any]:
    try:
        task = kronos_task_service.enqueue_task("single_predict", request.model_dump())
        return {"success": True, "data": task}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/tasks/batch-predict")
async def create_batch_predict_task(request: BatchPredictTaskRequest) -> dict[str, Any]:
    try:
        task = kronos_task_service.enqueue_task("batch_predict", request.model_dump())
        return {"success": True, "data": task}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/tasks/batch-backtest")
async def create_batch_backtest_task(request: BatchBacktestTaskRequest) -> dict[str, Any]:
    try:
        task = kronos_task_service.enqueue_task("batch_backtest", request.model_dump())
        return {"success": True, "data": task}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/tasks")
async def list_tasks(limit: int = 20, task_type: str | None = None) -> dict[str, Any]:
    return {"success": True, "data": kronos_task_service.list_tasks(limit=limit, task_type=task_type)}


@router.get("/api/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = kronos_task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "data": task}


@router.get("/api/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> dict[str, Any]:
    task = kronos_task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "data": task.get("result_payload", {})}


@router.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    try:
        task = kronos_task_service.cancel_task(task_id)
        return {"success": True, "data": task}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/runtime-status")
async def get_runtime_status() -> dict[str, Any]:
    return {"success": True, "data": kronos_task_service.get_runtime_status()}
