# Slurmer

This package is designed to run heavy loads in a computer cluster. Supported task schedulers are:

- Running on a single node
- [SLURM](https://slurm.schedmd.com/documentation.html)

PRs to add more schedulers are welcome.

## Features

- Run multiple tasks in parallel.
- Run tasks in randomised order but consistently in multiple computing nodes.
- Consistent pipeline to work with heavy tasks (directory creation, handling errors, postprocessing...)

## Example

You can find the documentation [here](https://jmigual.github.io/slurmer/). 

The package can be used as follows:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import slurmer


@dataclass
class MyTaskParameters(slurmer.TaskParameters):
    processing_id: int


@dataclass
class MyTaskResult(slurmer.TaskResult):
    good_result: bool


class MyTask(slurmer.Task):

    def __init__(self, min_job: int, max_job: int):
        super().__init__()
        self.min_job = min_job
        self.max_job = max_job

    def generate_parameters(self) -> Iterator[MyTaskParameters]:
        for i in range(self.min_job, self.max_job):
            yield MyTaskParameters(processing_id=i)

    def make_dirs(self):
        # Dirs are created before the task is run
        for i in range(self.min_job, self.max_job):
            Path(f"out/result_{i}").mkdir(parents=True, exist_ok=True)

    def process_function(self, parameters: MyTaskParameters) -> MyTaskResult:
        # Run your heavy code here
        for i in range(parameters.processing_id, parameters.processing_id + 10):
            print(f"Processing {i}")

        return MyTaskResult(good_result=True)

    def process_output(self, result: MyTaskResult) -> bool:
        # Postprocess output, this is NOT run in parallel and only handles the tasks run 
        # in the current node. If you want to postprocess everything wait for all the nodes
        # to finsish computing.
        #
        # This function should check whether the output is the "expected" one and return True 
        # if something went wrong.
        return not result.good_result
```

Then in your main script:
```python
>>> task = MyTask(min_job=1, max_job=10)
>>> task.execute_tasks()
```
