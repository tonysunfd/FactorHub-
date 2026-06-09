"""
因子管理API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.api.dependencies import service_attr

router = APIRouter()


# ========== 数据模型 ==========

class FactorCreate(BaseModel):
    """创建因子请求"""
    name: str
    code: str
    category: str
    description: str = ""
    formula_type: str = "expression"  # expression 或 function
    scope_type: str = "stock"
    target_stock_code: str = ""
    target_universe: str = ""
    origin_type: str = "manual"
    task_metadata: dict = {}


class FactorUpdate(BaseModel):
    """更新因子请求"""
    name: Optional[str] = None
    code: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None


class BatchGenerateRequest(BaseModel):
    """批量生成因子请求"""
    base_factors: List[str]
    generate_methods: List[str]  # ["arithmetic", "statistics", "technical"]
    ic_threshold: float = 0.03
    ir_threshold: float = 0.5
    min_valid_ratio: float = 0.7


class PreselectRequest(BaseModel):
    """预筛选因子请求"""
    factors: List[str]
    ic_threshold: float = 0.03
    ir_threshold: float = 0.5
    min_valid_ratio: float = 0.7


# ========== API端点 ==========

@router.get("/")
async def get_factors(
    category: Optional[str] = None,
    source: Optional[str] = None
):
    """
    获取因子列表

    参数:
    - category: 分类筛选（可选）
    - source: 来源筛选 preset/user（可选）
    """
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        factors = factor_service.get_all_factors()

        # 筛选
        if category:
            factors = [f for f in factors if f.get("category") == category]
        if source:
            factors = [f for f in factors if f.get("source") == source]

        return {
            "success": True,
            "data": factors,
            "total": len(factors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_factor_stats():
    """获取因子统计信息"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        stats = factor_service.get_factor_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{factor_id}")
async def get_factor(factor_id: int):
    """获取因子详情"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        factor = factor_service.get_factor_detail(factor_id)
        return {
            "success": True,
            "data": factor
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{factor_id}/snapshots")
async def list_factor_snapshots(factor_id: int):
    """获取因子任务快照列表"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        snapshots = factor_service.list_factor_task_snapshots(factor_id)
        return {
            "success": True,
            "data": snapshots,
            "total": len(snapshots),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{factor_id}/snapshots/{snapshot_id}")
async def get_factor_snapshot(factor_id: int, snapshot_id: int):
    """获取单个因子任务快照"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        snapshot = factor_service.get_factor_task_snapshot(factor_id, snapshot_id)
        return {
            "success": True,
            "data": snapshot,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_factor(request: FactorCreate):
    """创建新因子"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        # 创建因子
        factor = factor_service.create_factor(
            name=request.name,
            code=request.code,
            category=request.category,
            description=request.description,
            formula_type=request.formula_type,
            scope_type=request.scope_type,
            target_stock_code=request.target_stock_code,
            target_universe=request.target_universe,
            origin_type=request.origin_type,
            task_metadata=request.task_metadata,
        )

        return {
            "success": True,
            "data": factor,
            "message": "因子创建成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{factor_id}")
async def update_factor(factor_id: int, request: FactorUpdate):
    """更新因子"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        # 更新因子
        factor_service.update_factor(
            factor_id=factor_id,
            name=request.name,
            code=request.code,
            category=request.category,
            description=request.description
        )

        return {
            "success": True,
            "message": "因子更新成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{factor_id}")
async def delete_factor(factor_id: int):
    """删除因子"""
    try:
        factor_service = service_attr("backend.services.factor_service", "factor_service")
        success = factor_service.delete_factor(factor_id)

        if not success:
            raise HTTPException(status_code=404, detail="因子不存在或删除失败")

        return {
            "success": True,
            "message": "因子删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate")
async def batch_generate_factors(request: BatchGenerateRequest):
    """批量生成因子"""
    try:
        factor_generator_service = service_attr("backend.services.factor_generator_service", "factor_generator_service")
        all_generated_factors = []

        # 根据选择的生成方法调用相应的函数
        for method in request.generate_methods:
            if method == "arithmetic":
                # 算术运算组合
                factors = factor_generator_service.generate_binary_combinations(
                    base_factors=request.base_factors,
                    max_depth=2,
                    max_combinations=50
                )
                all_generated_factors.extend(factors)

            elif method == "statistics":
                # 统计变换
                factors = factor_generator_service.generate_statistical_combinations(
                    base_factors=request.base_factors,
                    max_combinations=50
                )
                all_generated_factors.extend(factors)

            elif method == "technical":
                # 技术指标组合
                factors = factor_generator_service.generate_indicator_combinations(
                    base_factors=request.base_factors,
                    max_combinations=30
                )
                all_generated_factors.extend(factors)

        # 混合因子生成
        if len(request.generate_methods) > 1:
            hybrid_factors = factor_generator_service.generate_hybrid_factors(
                base_factors=request.base_factors,
                n_factors=20
            )
            all_generated_factors.extend(hybrid_factors)

        # 去重（处理混合了字符串和字典的情况）
        seen = set()
        unique_factors = []
        for factor in all_generated_factors:
            # 如果是字典，使用其expression字段作为唯一标识
            key = factor["expression"] if isinstance(factor, dict) else factor
            if key not in seen:
                seen.add(key)
                unique_factors.append(factor)

        all_generated_factors = unique_factors

        result = {
            "generated_count": len(all_generated_factors),
            "factors": all_generated_factors[:20],  # 只返回前20个示例
            "total_possible": len(all_generated_factors)
        }

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preselect")
async def preselect_factors(request: PreselectRequest):
    """预筛选因子"""
    try:
        # 这里需要实现预筛选逻辑
        # 暂时返回示例数据
        return {
            "success": True,
            "data": {
                "total": len(request.factors),
                "selected": len(request.factors),
                "factors": request.factors
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_factor(request: dict):
    """验证因子公式"""
    try:
        code = request.get("code", "")
        formula_type = request.get("formula_type", "expression")

        if not code:
            return {
                "success": False,
                "message": "代码不能为空"
            }

        # 字符检查：确保只包含合法字符
        import re
        # 使用更宽松的检查：只禁止控制字符，允许所有可打印字符（包括中文）
        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', code):
            return {
                "success": False,
                "message": "代码包含非法控制字符"
            }

        # 调用真正的验证逻辑：执行代码来测试
        is_valid, message = factor_service.validate_factor_code(code)

        if not is_valid:
            return {
                "success": False,
                "message": message
            }

        return {
            "success": True,
            "data": {
                "code": code,
                "formula_type": formula_type,
                "valid": True
            },
            "message": "验证通过"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@router.post("/{factor_id}/copy")
async def copy_factor(factor_id: int):
    """复制因子"""
    try:
        # 获取原因子信息
        factors = factor_service.get_all_factors()
        original_factor = next((f for f in factors if f.get("id") == factor_id), None)

        if not original_factor:
            raise HTTPException(status_code=404, detail="因子不存在")

        # 生成新的因子名称（名称_数字）
        base_name = original_factor.get("name", "")
        new_name = base_name

        # 查找已存在的同名副本数量
        existing_copies = [
            f for f in factors
            if f.get("source") == "user" and f.get("name", "").startswith(base_name + "_")
        ]

        # 提取已有的数字后缀
        suffix_numbers = []
        for f in existing_copies:
            name = f.get("name", "")
            if name.startswith(base_name + "_"):
                suffix = name[len(base_name) + 1:]
                if suffix.isdigit():
                    suffix_numbers.append(int(suffix))

        # 生成新的数字后缀
        if suffix_numbers:
            new_suffix = max(suffix_numbers) + 1
        else:
            new_suffix = 1

        new_name = f"{base_name}_{new_suffix}"

        # 创建新因子（作为用户自定义因子）
        new_factor = factor_service.create_factor(
            name=new_name,
            code=original_factor.get("code", ""),
            category=original_factor.get("category", ""),
            description=original_factor.get("description", ""),
            formula_type=original_factor.get("formula_type", "expression")
        )

        return {
            "success": True,
            "data": new_factor,
            "message": f"因子已复制为 {new_name}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
