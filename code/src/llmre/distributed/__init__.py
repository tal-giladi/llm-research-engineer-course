"""llmre.distributed — data/model parallelism, simulated on a single process.

Companion code for Module 9 (Distributed training). Everything here is a
single-process **simulation** of collectives and data parallelism: there is no
NCCL and no second device. See ``lessons/module-09/`` for the concepts and the
real-hardware notes.
"""

from .collectives import all_gather, ring_all_reduce
from .data_parallel import data_parallel_grads, split_batch

__all__ = [
    "ring_all_reduce",
    "all_gather",
    "data_parallel_grads",
    "split_batch",
]
