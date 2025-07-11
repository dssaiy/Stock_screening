"""
Task Manager module for handling long-running tasks.
Implements a simple in-memory task queue with status tracking.
"""
import uuid
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
import threading
import traceback

try:
    from api.json_utils import sanitize_task_result
except ImportError:
    # 如果绝对导入失败，尝试相对导入（本地开发环境）
    from .json_utils import sanitize_task_result


class TaskStatus(Enum):
    """Enum representing the possible states of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """Represents a background task with status tracking."""

    def __init__(self, task_id: str, created_at: float,
                 status: str = TaskStatus.PENDING, progress: int = 0,
                 message: str = "Task created", result: Optional[List[Dict[str, Any]]] = None,
                 error: Optional[str] = None, updated_at: Optional[float] = None,
                 completed_at: Optional[float] = None,
                 cached: bool = False):
        self.task_id = task_id
        self.status = status
        self.progress = progress
        self.message = message
        self.result = result
        self.error = error
        self.created_at = created_at
        self.updated_at = updated_at if updated_at is not None else created_at
        self.completed_at = completed_at
        self.cached = cached

    def update(self, status: Optional[str] = None, progress: Optional[int] = None,
               message: Optional[str] = None, result: Optional[List[Dict[str, Any]]] = None,
               error: Optional[str] = None, completed_at: Optional[float] = None,
               cached: Optional[bool] = None):
        if status:
            self.status = TaskStatus(status) # Convert string to Enum
            if TaskStatus(status) in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                self.completed_at = time.time()
        if progress is not None:
            self.progress = progress
        if message:
            self.message = message
        if result is not None:
            self.result = result
        if error:
            self.error = error
        if completed_at is not None:
            self.completed_at = completed_at
        if cached is not None:
            self.cached = cached
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        # Sanitize result to handle NaN and Infinity values
        sanitized_result = sanitize_task_result(self.result)

        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": sanitized_result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "cached": self.cached
        }


class TaskManager:
    """Manages background tasks and their statuses."""
    _instance = None
    _tasks: Dict[str, Task] = {}
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one TaskManager exists."""
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
        return cls._instance

    def create_task(self) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = Task(task_id=task_id, created_at=time.time())
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> None:
        """Update a task's status and details."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.update(**kwargs)

    def run_task_in_background(self, task_id: str, func: Callable, *args, **kwargs) -> None:
        """Run a function in a background thread and track its status."""
        def wrapper():
            task = self.get_task(task_id)
            if not task:
                return

            self.update_task(task_id, status=TaskStatus.RUNNING,
                             message="Task started")

            try:
                result = func(*args, **kwargs)
                self.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    progress=100,
                    message="Task completed successfully"
                )
            except Exception as e:
                error_msg = f"Task failed: {str(e)}"
                error_traceback = traceback.format_exc()
                self.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=f"{error_msg}\n{error_traceback}",
                    message=error_msg
                )

        # Start the background thread
        thread = threading.Thread(target=wrapper)
        thread.daemon = True  # Allow the thread to be terminated when the main process exits
        thread.start()

    def clean_old_tasks(self, max_age_seconds: int = 3600) -> None:
        """Remove tasks older than the specified age."""
        current_time = time.time()
        with self._lock:
            task_ids_to_remove = []
            for task_id, task in self._tasks.items():
                # Remove completed or failed tasks that are older than max_age_seconds
                if (task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) and
                    task.completed_at and
                        current_time - task.completed_at > max_age_seconds):
                    task_ids_to_remove.append(task_id)

            for task_id in task_ids_to_remove:
                del self._tasks[task_id]

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks as dictionaries."""
        with self._lock:
            return [task.to_dict() for task in self._tasks.values()]


# Create a singleton instance
task_manager = TaskManager()
