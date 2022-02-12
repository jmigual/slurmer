# noqa: D101
import multiprocessing as mp
import os
import signal
import subprocess  # noqa: S404
from dataclasses import dataclass
from time import sleep
from typing import Iterator, Optional

from slurmer import Task, TaskParameters, TaskResult


@dataclass(slots=True)
class MyTaskParameters(TaskParameters):
    tid: int

    def __hash__(self):
        return hash(self.tid)


class MyTask(Task):
    def generate_parameters(self) -> Iterator[TaskParameters]:
        for i in range(4):
            yield MyTaskParameters(tid=i)

    @staticmethod
    def processor_function(_: TaskParameters) -> Optional[TaskResult]:
        # We disable the checks of security as sleep is safe. Also it is a "global" command.
        subprocess.run(["sleep", "5"])  # noqa: S603,S607
        return TaskResult()


def run_tasks():
    try:
        # Create a task
        task = MyTask()
        task.execute_tasks()
    except KeyboardInterrupt:
        print("We got a proper interrupt!")


def test_sigint():
    p = mp.Process(target=run_tasks)
    p.start()
    sleep(1)
    assert p.is_alive()
    pid = p.pid
    assert pid is not None
    os.kill(pid, signal.SIGINT)
    p.join()
    assert not p.is_alive()


def test_sigterm():
    p = mp.Process(target=run_tasks)
    p.start()
    sleep(1)
    assert p.is_alive()
    pid = p.pid
    assert pid is not None
    os.kill(pid, signal.SIGTERM)
    p.join()
    assert not p.is_alive()


def test_run_tasks():
    run_tasks()
