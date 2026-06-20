"""
Kronos 原生 WebUI 独立服务
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from backend.core.settings import settings
from backend.services.kronos_task_service import kronos_task_service


app = Flask(__name__, template_folder="templates")
CORS(app)


@app.get("/")
def index():
    return render_template("index.html", default_model=settings.KRONOS_DEFAULT_MODEL, default_device=settings.KRONOS_DEFAULT_DEVICE)


@app.get("/api/available-models")
def available_models():
    return jsonify(
        {
            "success": True,
            "data": [
                {
                    "key": "kronos-base",
                    "name": "Kronos-base",
                    "description": "Base model，第一阶段默认使用 CPU 推理占位执行",
                    "context_length": 512,
                    "params": "102.3M",
                }
            ],
        }
    )


@app.post("/api/load-model")
def load_model():
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model_name", settings.KRONOS_DEFAULT_MODEL)
    device = payload.get("device", settings.KRONOS_DEFAULT_DEVICE)
    return jsonify(
        {
            "success": True,
            "data": {
                "loaded": True,
                "model_name": model_name,
                "device": settings.KRONOS_DEFAULT_DEVICE if device != "cpu" else device,
                "message": "Phase 1 仅启用 CPU 推理执行；GPU / ROCm 将在 Phase 2 开启。",
            },
        }
    )


@app.get("/api/model-status")
def model_status():
    return jsonify(
        {
            "success": True,
            "data": {
                "loaded": True,
                "model_name": settings.KRONOS_DEFAULT_MODEL,
                "device": settings.KRONOS_DEFAULT_DEVICE,
                "phase": "cpu",
            },
        }
    )


@app.get("/api/data-files")
def data_files():
    return jsonify({"success": True, "data": kronos_task_service.list_data_files()})


@app.post("/api/load-data")
def load_data():
    payload = request.get_json(silent=True) or {}
    prepared = kronos_task_service.load_dataset(payload)
    return jsonify(
        {
            "success": True,
            "data": {
                "source_type": prepared.source_type,
                "title": prepared.title,
                "preview": prepared.data_preview,
            },
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    task = kronos_task_service.enqueue_task("single_predict", payload)
    return jsonify({"success": True, "data": task})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7070, debug=False)
