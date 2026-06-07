"""
系统健康检查服务。
"""
from __future__ import annotations

import threading
import time
from typing import Dict


class SystemHealthService:
    """提供带缓存的数据源健康检查，避免高频探测拖慢首页和启动路径。"""

    def __init__(self, ttl_seconds: int = 60):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached_result: Dict[str, Dict[str, str | bool]] = {}

    def get_source_health(self, force_refresh: bool = False) -> Dict[str, Dict[str, str | bool]]:
        now = time.time()
        if not force_refresh and self._cached_result and (now - self._cached_at) < self.ttl_seconds:
            return self._cached_result

        with self._lock:
            now = time.time()
            if not force_refresh and self._cached_result and (now - self._cached_at) < self.ttl_seconds:
                return self._cached_result

            result = {
                "akshare": self._check_akshare(),
                "baostock": self._check_baostock(),
            }
            self._cached_result = result
            self._cached_at = now
            return result

    def _check_akshare(self) -> Dict[str, str | bool]:
        try:
            import akshare as ak

            ak.stock_zh_a_daily(
                symbol="sz000001",
                start_date="20230903",
                end_date="20230908",
                adjust="qfq",
            )
            return {
                "healthy": True,
                "label": "AKShare",
                "message": "数据接口可用",
            }
        except Exception as exc:
            return {
                "healthy": False,
                "label": "AKShare",
                "message": str(exc),
            }

    def _check_baostock(self) -> Dict[str, str | bool]:
        try:
            import baostock as bs

            login_result = bs.login()
            if getattr(login_result, "error_code", "0") != "0":
                return {
                    "healthy": False,
                    "label": "BaoStock",
                    "message": getattr(login_result, "error_msg", "登录失败"),
                }
            try:
                return {
                    "healthy": True,
                    "label": "BaoStock",
                    "message": "数据接口可用",
                }
            finally:
                bs.logout()
        except Exception as exc:
            return {
                "healthy": False,
                "label": "BaoStock",
                "message": str(exc),
            }


system_health_service = SystemHealthService()
