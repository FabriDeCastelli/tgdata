"""Batched windows of a time-varying graph: tgdata's vectorised gather against a Python list.

Both build, on the GPU, the step-major batch `window_edges` returns: for a batch of windows,
the edges of window step k of every sample together, sample b's nodes shifted by b * N. The
list approach slices each (sample, step) snapshot in a Python loop, as per-sample loaders do,
and concatenates. Outputs are checked bit for bit before timing.

  python benchmarks/dynamic_edges.py --data DIR
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import tgdata
from tgdata.device import window_edges


def list_edges(data: dict, first: torch.Tensor, window: int, num_nodes: int) -> dict:
    ptr, ei, ew = data["edge_ptr"].tolist(), data["edge_index"], data["edge_weight"]
    starts = first.tolist()
    per_step, weights, counts = [], [], []
    for k in range(window):
        n = 0
        for b, s in enumerate(starts):
            lo, hi = ptr[s + k], ptr[s + k + 1]
            per_step.append(ei[:, lo:hi] + b * num_nodes)
            weights.append(ew[lo:hi])
            n += hi - lo
        counts.append(n)
    edge_ptr = torch.tensor([0, *np.cumsum(counts)], device=ei.device)
    return {"edge_index": torch.cat(per_step, 1), "edge_weight": torch.cat(weights),
            "edge_ptr": edge_ptr}


def timed(fn, *args, reps: int = 50) -> float:
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1e3)


def synthetic(steps: int, nodes: int, edges_per_step: int, seed: int = 0) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    counts = torch.poisson(torch.full((steps,), float(edges_per_step), device="cuda"),
                           generator=gen).long()
    total = int(counts.sum())
    return {"edge_index": torch.randint(nodes, (2, total), device="cuda", generator=gen),
            "edge_weight": torch.rand(total, device="cuda", generator=gen),
            "edge_ptr": torch.cat([counts.new_zeros(1), counts.cumsum(0)])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.manual_seed(0)
    g = tgdata.load_dir(f"{args.data}/twittertennis-rg17")
    task = g.task(split=None, device="cuda")
    graphs = {"twittertennis-rg17 (1k nodes, ~340 edges/step)":
              (task.data, g.num_steps, g.num_nodes),
              "synthetic (10k nodes, ~20k edges/step)": (synthetic(500, 10_000, 20_000), 500,
                                                         10_000)}
    print(f"GPU: {torch.cuda.get_device_name()} | batch {args.batch_size} windows | "
          "ms per batch, median of 50")
    for label, (data, steps, nodes) in graphs.items():
        print(f"\n{label}")
        for window in (1, 4, 12):
            first = torch.randint(0, steps - window + 1, (args.batch_size,), device="cuda")
            fast, slow = (window_edges(data, first, window, nodes),
                          list_edges(data, first, window, nodes))
            same = all(torch.equal(fast[k], slow[k]) for k in fast)
            t_fast = timed(window_edges, data, first, window, nodes)
            t_slow = timed(list_edges, data, first, window, nodes)
            print(f"  window {window:2d}: {fast['edge_index'].shape[1]:9,d} edges | "
                  f"list {t_slow:7.2f} ms  vectorised {t_fast:6.3f} ms  "
                  f"{t_slow / t_fast:6.1f}x faster | bit-identical: {same}")


if __name__ == "__main__":
    main()
