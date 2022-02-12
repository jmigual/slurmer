from __future__ import annotations

import abc
import math
import multiprocessing as mp
import os
import random
import signal
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterator, Optional

from tqdm.auto import tqdm


class Error(Enum):
    """Type of termination error."""

    #: No error occurred during execution
    NONE = auto()

    #: A keyboard interrupt was received
    TERMINATE = auto()

    #: A task produced and undesired output
    SYSTEM = auto()


class _PoolDummy:
    def terminate(self):
        pass

    def join(self):
        pass

    def close(self):
        pass


errored = Error.NONE
pool = _PoolDummy()


def _sigterm_handler_soft(*_):
    global errored, pool
    print("SIGTERM SOFT")
    errored = Error.TERMINATE
    pool.terminate()
    raise KeyboardInterrupt


@dataclass(slots=True, order=True)
class TaskParameters:
    """Data class containing the parameters that should be used by a task.

    Inherit this class to add parameters to your task.
    """

    pass


@dataclass(slots=True)
class _PrivateTaskParameters:
    f: Callable[[TaskParameters], Optional[TaskResult]]
    p: TaskParameters


@dataclass(slots=True)
class TaskResult:
    """Data class containing the result of a task.

    Inherit this class to add results to your task.

    Example:
        You can inherit this class as follows:

        .. code-block:: python
            :linenos:

            from dataclass import dataclass
            from slurmer import TaskResult

            @dataclass
            class MyTaskResult(TaskResult):
                my_result: int

    """

    pass


class TaskFailedError(Exception):
    """Exception thrown if one of the tasks fails."""

    pass


class Task(abc.ABC):
    """Base class defining a set of Tasks to be done."""

    def __init__(
        self,
        cluster_id: Optional[int] = None,
        cluster_total: Optional[int] = None,
    ):
        """Initialize the class.

        Args:
            cluster_id: Id of the current node in the cluster, if None
                the value is read from SLURM environment variable SLURM_ARRAY_TASK_ID. Defaults to
                None.
            cluster_total (Optional[int], optional): Number of allocated nodes in the cluster, if
                None the value is read from the SLURM environment variable SLURM_ARRAY_TASK_MAX.
                Defaults to None.
        """
        self.cluster_id, self.cluster_total = Task.get_cluster_ids(cluster_id, cluster_total)

    @abc.abstractmethod
    def generate_parameters(self) -> Iterator[TaskParameters]:
        """Generate all the parameters for the different tasks to run.

        This function **must be implemented**. It should return an iterator over all the parameters
        that should be passed to the processor_function(). The number of tasks to run is determined
        by the number of parameters returned by this method.
        """
        pass

    def make_dirs(self):
        """Make directories before execution.

        If some directories need to be created before executing the tasks, inherit this method and
        create the directories here. *Override this function to add some behaviour*.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def processor_function(parameters: TaskParameters) -> Optional[TaskResult]:
        """Execute the task.

        This function **must be implemented**. Here you can run the task that you want to run.
        It should return a TaskResult if the execution worked as expected (even if it's just an
        empty object), None if the execution failed and execution should be terminated.
        """
        pass

    def process_output(self, result: TaskResult) -> bool:
        """Process the generated output.

        Process the output generated with processor_function() and evaluate whether execution
        should be terminated. Override this method if the default behaviour (return False) should
        be changed.

        Args:
            result (TaskResult): Result of the task as returned by the processor_function()

        Returns:
            bool: True if the execution should be terminated, False otherwise.
        """
        return False

    def after_run(self):
        """Handle results after running tasks.

        Override this method to handle results after everything has been run. Keep in mind that
        this is run after all the tasks of this fold have been run but that in another node they
        may still be running.
        """
        pass

    def key_interrupt(self):
        """Override this method to handle the status after a keyboard interrupt."""
        pass

    # Do not override anything past this point

    @staticmethod
    def get_cluster_ids(cluster_id: Optional[int], cluster_total: Optional[int]) -> tuple[int, int]:
        """Get the cluster id and the total number of nodes.

        If the cluster_id and cluster_total are not set, they are obtained from the environment
        variables passed by the SLURM task scheduler.

        Args:
            cluster_id (Optional[int]): ID of the current node.
            cluster_total (Optional[int]): Total number of nodes nodes that we can use in the
                cluster.

        Returns:
            tuple[int, int]: Tuple with current cluster ID and total number of nodes.
        """
        if cluster_id is None:
            cluster_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 1)) - 1
        if cluster_total is None:
            cluster_total = int(os.getenv("SLURM_ARRAY_TASK_MAX", 1))

        return cluster_id, cluster_total

    @staticmethod
    def _tasks_distribution(total_tasks: int, workers: int) -> list[tuple[int, int]]:
        length = int(math.ceil(total_tasks / workers))
        limit = total_tasks - (length - 1) * workers

        task_list = []
        prev_end = 0
        for i in range(workers):
            task_begin = prev_end
            task_end = task_begin + length - (0 if i < limit else 1)
            task_end = prev_end = min(task_end, total_tasks)
            task_list.append((task_begin, task_end))

        return task_list

    def _obtain_current_fold(self):
        params = sorted(set(self.generate_parameters()))
        task_list = Task._tasks_distribution(len(params), self.cluster_total)
        task_begin, task_end = task_list[self.cluster_id]

        # Shuffle in order to have big tasks matched with small ones in order to save memory
        random.seed(42)
        random.shuffle(params)

        if self.cluster_total > 1:
            print(f"Running fold {self.cluster_id + 1} out of {self.cluster_total}")
            print(
                f"{task_begin} to {task_end} tasks will be run instead of the whole {len(params)}"
            )

        return params[task_begin:task_end]

    @staticmethod
    def _process(p: _PrivateTaskParameters) -> Optional[TaskResult]:
        try:
            return p.f(p.p)
        except KeyboardInterrupt:
            return None

    def execute_tasks(
        self,
        make_dirs_only: bool = False,
        debug: bool = False,
        processes: int = None,
        no_bar: bool = False,
        description: str = "",
    ) -> Error:
        """Execute all the tasks.

        The tasks are executed in a random order so do not rely on the order and use IDs instead.
        Even if the order is random, it is consistent across nodes.

        Args:
            make_dirs_only (bool, optional): If True, only the directories will be created and
                execution will return. Defaults to False.
            debug (bool, optional): If True, the execution will be run in debug mode disabling all
                the parallelism. Defaults to False.
            processes (int, optional): Number of processes to use. Pass None to use as many as the
                system has. Defaults to None.
            no_bar (bool, optional): If True, the progress bar will not be displayed. Defaults to
                False.
            description (str, optional): Description of the task. Defaults to "".

        Raises:
            KeyboardInterrupt: If a keyboard interrupt is passed, all the tasks are stopped and a
                KeyboardInterrupt is raised.
            TaskFailedError: If one of the tasks returns a strange result then a TaskFailedError is
                raised.

        Returns:
            Error: The termination result of the tasks. If everything is fine, the result is
            Error.None.
        """
        global errored, pool
        params = self._obtain_current_fold()
        params2 = (_PrivateTaskParameters(self.processor_function, p) for p in params)

        self.make_dirs()
        if make_dirs_only:
            return Error.NONE

        if len(params) <= 0:
            print("WARNING: No tasks have to be done, are you sure you did everything right?")

        signal_list = [signal.SIGINT, signal.SIGTERM]
        signals = {s: signal.getsignal(s) for s in signal_list}

        if debug:
            pool = _PoolDummy()
            generator = map(self._process, params2)
        else:
            pool = mp.Pool(processes=processes)
            generator = pool.imap_unordered(self._process, params2)

        errored = Error.NONE
        for s in signal_list:
            signal.signal(s, _sigterm_handler_soft)

        try:
            for output in tqdm(
                generator,
                total=len(params),
                desc=description,
                disable=(len(params) <= 0) or no_bar,
                smoothing=0.02,
            ):
                if output is None:
                    # This is a keyboard interrupt
                    errored = Error.TERMINATE
                    pool.terminate()
                    break
                if self.process_output(output):
                    # This is caused by an actual error while running
                    errored = Error.SYSTEM
                    pool.terminate()
                    break

                if errored != Error.NONE:
                    pool.terminate()
                    break

        except KeyboardInterrupt:
            errored = Error.TERMINATE

        if errored != Error.NONE:
            pool.terminate()
            pool.join()
            if errored == Error.TERMINATE:
                self.key_interrupt()
                raise KeyboardInterrupt
            elif errored == Error.SYSTEM:
                raise TaskFailedError("System returned non 0 exit code")

        pool.close()
        self.after_run()
        for k, v in signals.items():
            if v is None:
                continue
            signal.signal(k, v)

        return errored
