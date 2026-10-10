import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)


def train(device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Every rank refers to the same logical dataset. The sampler selects rows.
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(256, 16, generator=generator)
    y = x.sum(dim=1, keepdim=True)
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, drop_last=True)
    if len(loader) == 0:
        raise ValueError("Not enough examples for one full batch per rank.")

    # DDP synchronizes the starting model across ranks by default.
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 1).to(device)
    device_ids = [device.index] if device.type == "cuda" else None
    model = DistributedDataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    model.train()

    for epoch in range(3):
        sampler.set_epoch(epoch)                 # a new coordinated shuffle
        loss_sum = torch.zeros((), device=device)

        for local_x, local_y in loader:
            local_x = local_x.to(device)
            local_y = local_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(local_x), local_y)
            loss.backward()                     # DDP averages gradients here
            optimizer.step()                    # every rank takes the same step
            loss_sum += loss.detach()

        # Separate collective for a reporting metric; not gradient synchronization.
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        if rank == 0:
            mean_loss = (loss_sum / (len(loader) * world_size)).item()
            log.info("epoch=%d mean_loss=%.6f", epoch, mean_loss)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Launch with torchrun, not python train.py.")
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError("Each local process needs its own CUDA GPU.")
            torch.cuda.set_device(local_rank)
            device, backend = torch.device("cuda", local_rank), "nccl"
        else:
            # CPU fallback: same DDP code, Gloo instead of NCCL, no speedup.
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
        train(device)
    except Exception:
        log.exception("Training failed on rank %s", os.environ.get("RANK", "unknown"))
        raise                                   # let the launcher report the failure
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
