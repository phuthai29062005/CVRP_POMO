"""
5 lần chạy: neural full_node -> LS lặp cho đến khi không cải thiện nữa.
Không dùng GA, không có time budget — chỉ neural + LS until convergence.
"""
import os
import sys
import time
import random

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from read_data import read_data
from GA import solve_remaining_with_neural
from caculate import get_fitness
from Local_search.local_search import local_search, reset_ls_cache

# =========================================================
INSTANCE_PATH = os.path.join(
    BASE_DIR, "ML4VRP2026", "Instances", "cvrp", "vrp", "X-n101-k25.vrp"
)

NEURAL_DECODE_TYPE = "sampling"

NEURAL_CKPT_CANDIDATES = [
    os.path.join(BASE_DIR, "checkpoints_neural_fill_v3", "model_best_sampling.pt"),
    os.path.join(BASE_DIR, "Train_Neural", "checkpoints_neural_fill_v2", "model_best_sampling.pt"),
    os.path.join(BASE_DIR, "checkpoints_neural_fill_v2", "model_best_sampling.pt"),
]

NUM_RUNS = 5
# =========================================================


def resolve_ckpt():
    for p in NEURAL_CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Không tìm thấy checkpoint neural. Đã kiểm:\n" +
                            "\n".join(NEURAL_CKPT_CANDIDATES))


def fmt(s):
    if s < 60:
        return f"{s:.2f}s"
    m = int(s // 60)
    return f"{m}m {s % 60:.2f}s"


def main():
    dimension, capacity, nodes = read_data(INSTANCE_PATH)
    instance_name = os.path.basename(INSTANCE_PATH).split(".")[0]
    ckpt_path = resolve_ckpt()

    print("=" * 65)
    print(f"Instance  : {instance_name}")
    print(f"Dimension : {dimension}  |  Capacity : {capacity}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Decode    : {NEURAL_DECODE_TYPE}")
    print("=" * 65)

    all_customers = list(range(2, dimension + 1))
    results = []

    for run in range(1, NUM_RUNS + 1):
        reset_ls_cache()
        print(f"\n{'='*65}")
        print(f"  RUN {run}/{NUM_RUNS}")
        print(f"{'='*65}")

        t0 = time.perf_counter()

        # ---- 1. Tạo nghiệm ban đầu bằng neural (toàn bộ node) ----
        perm, markers = solve_remaining_with_neural(
            remaining_vertices=all_customers,
            nodes_data=nodes,
            capacity_value=capacity,
            ckpt_path=ckpt_path,
            decode_type=NEURAL_DECODE_TYPE,
        )

        initial_fit = get_fitness(perm, markers, nodes)
        n_routes_init = sum(markers)
        print(f"  [Neural] fit={initial_fit:.0f}  routes={n_routes_init}")

        # ---- 2. LS lặp cho đến khi không cải thiện nữa ----
        parent  = [list(perm)]
        route   = [list(markers)]
        fitness = [(initial_fit, 0)]
        best_fit = initial_fit
        ls_iter  = 0

        while True:
            prev_fit = fitness[0][0]

            parent, route, fitness = local_search(
                parent, capacity, nodes, route, fitness,
                elite_ratio=1.0,
                global_best_fit=best_fit,
                gen=ls_iter,
                deadline=None,
            )
            fitness.sort()
            cur_fit = fitness[0][0] if fitness else prev_fit
            ls_iter += 1

            improved = cur_fit < prev_fit - 1e-9
            print(f"  [LS iter {ls_iter}] {prev_fit:.0f} -> {cur_fit:.0f}"
                  + ("  (+)" if improved else "  (no improve → stop)"))

            if not improved:
                break
            best_fit = cur_fit

        elapsed = time.perf_counter() - t0
        n_routes_final = sum(route[0])
        print(f"\n  => Run {run}: neural={initial_fit:.0f}  "
              f"final={best_fit:.0f}  routes={n_routes_final}  "
              f"LS_iters={ls_iter}  time={fmt(elapsed)}")

        results.append(dict(run=run, neural=initial_fit, final=best_fit,
                            routes=n_routes_final, iters=ls_iter, time=elapsed))

    # ---- Tổng kết ----
    print(f"\n{'='*65}")
    print("  TỔNG KẾT 5 LẦN CHẠY")
    print(f"{'='*65}")
    print(f"  {'Run':>4}  {'Neural':>8}  {'Final':>8}  {'Routes':>7}  "
          f"{'LS iters':>9}  {'Time':>10}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}")
    for r in results:
        print(f"  {r['run']:>4}  {r['neural']:>8.0f}  {r['final']:>8.0f}  "
              f"{r['routes']:>7}  {r['iters']:>9}  {fmt(r['time']):>10}")

    finals = [r["final"] for r in results]
    print(f"  {'-'*55}")
    print(f"  Best  : {min(finals):.0f}")
    print(f"  Worst : {max(finals):.0f}")
    print(f"  Avg   : {sum(finals)/len(finals):.1f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
