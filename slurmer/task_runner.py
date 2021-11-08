from __future__ import annotations

import abc
import math
import multiprocessing as mp
import os
import random
import signal
from enum import Enum, auto
from typing import Iterator, NamedTuple, Optional

from tqdm.auto import tqdm


class Error(Enum):
    """Type of termination error."""

    NONE = auto()
    TERMINATE = auto()
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
    print("SIGTERM SOFT!!!")
    errored = Error.TERMINATE
    pool.terminate()
    raise KeyboardInterrupt


class TaskParameters(NamedTuple):
    """Named tuple containing the parameters that should be used by a task."""

    pass


class TaskResult(NamedTuple):
    pass


class TaskFailedError(Exception):
    pass


class Task(abc.ABC):
    """Base class defining a set of Tasks to be done."""

    def __init__(
        self,
        description: str = "",
        cluster_id: Optional[int] = None,
        cluster_total: Optional[int] = None,
    ):
        """Initialize the class.

        Args:
            description (str, optional): [description]. Defaults to "".
            cluster_id (Optional[int], optional): [description]. Defaults to None.
            cluster_total (Optional[int], optional): [description]. Defaults to None.
        """
        self.description = description

        self.cluster_id, self.cluster_total = Task._get_cluster_ids(cluster_id, cluster_total)

    @abc.abstractmethod
    def _generate_parameters(self) -> Iterator[TaskParameters]:
        pass

    def _make_dirs(self):
        """Make directories before execution.

        If some directories need to be created before executing the tasks, inherit this method and
        create the directories here.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _processor_function(parameters: TaskParameters) -> Optional[TaskResult]:
        pass

    def _process_output(self, result: TaskResult) -> bool:
        """Process the generated output.

        Process the output generated with _processor_function() and evaluate whether execution
        should be terminated. Override this method if the default behaviour (return False) should
        be changed.

        Args:
            result (TaskResult): Result of the task as returned by the _processor_function()

        Returns:
            bool: True if the execution should be terminated, False otherwise.
        """
        return False

    def _after_run(self):
        """Handle results after running tasks.

        Override this method to handle results after everything has been run. Keep in mind that
        this is run after all the tasks of this fold have been run but that in another node they
        may still be running.
        """
        pass

    def _key_interrupt(self):
        """Override this method to handle the status after a keyboard interrupt."""
        pass

    # Do not override anything past this point

    @staticmethod
    def _get_cluster_ids(
        cluster_id: Optional[int], cluster_total: Optional[int]
    ) -> tuple[int, int]:
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
        params = sorted(set(self._generate_parameters()))
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

    def execute_tasks(
        self,
        make_dirs_only: bool = False,
        debug: bool = False,
        processes: int = None,
        no_bar: bool = False,
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

        self._make_dirs()
        if make_dirs_only:
            return Error.NONE

        prev_signal = signal.getsignal(signal.SIGTERM)
        if prev_signal is None:
            prev_signal = signal.SIG_DFL

        if len(params) <= 0:
            print("WARNING: No tasks have to be done, are you sure you did everything right?")

        if debug:
            pool = _PoolDummy()
            generator = map(self._processor_function, params)
        else:
            pool = mp.Pool(processes=processes)
            generator = pool.imap_unordered(self._processor_function, params)

        signal.signal(signal.SIGTERM, _sigterm_handler_soft)

        try:
            for output in tqdm(
                generator,
                total=len(params),
                desc=self.description,
                disable=(len(params) <= 0) or no_bar,
                smoothing=0.02,
            ):
                if output is None:
                    # This is a keyboard interrupt
                    errored = Error.TERMINATE
                    pool.terminate()
                    break
                if self._process_output(output):
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
                self._key_interrupt()
                raise KeyboardInterrupt
            elif errored == Error.SYSTEM:
                raise TaskFailedError("System returned non 0 exit code")

        pool.close()
        self._after_run()
        signal.signal(signal.SIGTERM, prev_signal)
        return errored
