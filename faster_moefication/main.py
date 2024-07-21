import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from moefication import kmeans_balanced

dist.init_process_group(backend="nccl")
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)

if rank == 0:
    w_in_s = [torch.rand(768*8, 768).cuda() for _ in range(8)]
    w_out_s = [torch.rand(768, 768*8).cuda() for _ in range(8)]
else:
    w_in_s = [torch.zeros(768*8, 768).cuda() for _ in range(8)]

torch.distributed.barrier()
start_time = time.time()

for w_in in w_in_s:
    dist.broadcast(w_in, src=0)

centers, labels, obj = kmeans_balanced(w_in_s[rank].cpu().numpy(), 32, 192)
print(labels)

if rank == 0:
    gather_list = [torch.zeros_like(labels) for _ in range(8)]
else:
    gather_list = None

dist.gather(labels, gather_list=gather_list, dst=0)

torch.distributed.barrier()
end_time = time.time()

if rank == 0:
    print("Clustering time:", (end_time - start_time))

    for layer, labels in enumerate(gather_list):
        labels = torch.tensor(labels).cpu()
        w_in_ = w_in_s[layer].cpu()
        w_out = w_out_s[layer].cpu()
        tmp_in = []
        tmp_in_norm = []
        for i in range(32):
            tmp_in.append(w_in[labels == i, :])
            tmp_in_norm.append(w_in_[labels == i, :])
        tmp_in = torch.stack(tmp_in, dim=0)
        tmp_in_norm = np.stack(tmp_in_norm, axis=0)

        tmp_out = []
        for i in range(32):
            tmp_out.append(w_out[:, labels == i].transpose(0, 1))
        tmp_out = torch.cat(tmp_out, dim=0)

        # wg = tmp_in.mean(1)
        wg = tmp_in_norm.mean(1)
        wg = torch.tensor(wg, dtype=tmp_in.dtype, device=tmp_in.device)
        wg_norm = wg / torch.norm(wg, dim=-1, keepdim=True) * tmp_in[0, 0, :].norm()

    end_time = time.time()
    print("Total time:", (end_time - start_time))