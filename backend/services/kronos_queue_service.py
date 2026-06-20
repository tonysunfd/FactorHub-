"""
Kronos 队列服务
"""
from __future__ import annotations

from dataclasses import dataclass

from redis import Redis
from rq import Queue

from backend.core.settings import settings


@dataclass
class QueueHandle:
    """队列句柄"""

    connection: Redis
    queue: Queue


class KronosQueueService:
    """Redis + RQ 队列封装"""

    def __init__(self) -> None:
        self._handle: QueueHandle | None = None

    def get_handle(self) -> QueueHandle:
        if self._handle is None:
            connection = Redis.from_url(settings.REDIS_URL)
            queue = Queue(settings.KRONOS_QUEUE_NAME, connection=connection, default_timeout=60 * 60)
            self._handle = QueueHandle(connection=connection, queue=queue)
        return self._handle

    def enqueue(self, func_path: str, *args, job_id: str | None = None, **kwargs):
        handle = self.get_handle()
        return handle.queue.enqueue(func_path, *args, job_id=job_id, kwargs=kwargs)

    def ping(self) -> bool:
        try:
            self.get_handle().connection.ping()
            return True
        except Exception:
            return False


kronos_queue_service = KronosQueueService()
