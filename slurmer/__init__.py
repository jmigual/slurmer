"""Slurmer package.

Use this package to run tasks in a computing cluster. Supported task schedulers are:
    - `SLURM <https://slurm.schedmd.com/documentation.html>`_
"""
from .task_runner import Task, TaskFailedError, TaskParameters, TaskResult

__all__ = ["Task", "TaskFailedError", "TaskParameters", "TaskResult"]
