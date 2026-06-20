"""
Kronos RQ Worker 启动入口
"""
from rq import Worker

from backend.services.kronos_queue_service import kronos_queue_service


def main() -> None:
    handle = kronos_queue_service.get_handle()
    worker = Worker([handle.queue.name], connection=handle.connection)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
