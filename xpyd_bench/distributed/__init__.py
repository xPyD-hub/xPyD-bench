"""Distributed benchmark coordination (M32).

Multi-machine load generation with a coordinator/worker architecture.
"""

from xpyd_bench.distributed.coordinator import DistributedResult, run_distributed
from xpyd_bench.distributed.worker import WorkerApp

__all__ = ["DistributedResult", "WorkerApp", "run_distributed"]
