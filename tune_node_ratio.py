"""
Tune keep_node_ratio cho instance n < 200.

Instances : X-n101-k25, X-n120-k6, X-n148-k46, X-n172-k51, X-n195-k51
Ratios     : 0.20, 0.50, 0.75
Trials     : 5 lần mỗi combo
Generations: 5 gen GA, sau đó 1 lần local search
Output     : tune_node_ratio_results.txt

Metric ghi ra:
  - best_fit : fitness tốt nhất sau local search
  - best_rank: rank (1-based) của cá thể đó trong population trước LS
"""

import os
import random
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# CONFIG
# =========================================================

INSTANCE_DIR = os.path.join(BASE_DIR, "ML4VRP2026", "Instances", "cvrp", "vrp")

INSTANCES = [
    "X-n303-k21.vrp",
    "X-n336-k84.vrp",
    "X-n359-k29.vrp",
    "X-n376-k94.vrp",
    "X-n393-k38.vrp",
]

KEEP_NODE_RATIOS = [0.20, 0.50, 0.75]

NUM_TRIALS     = 5
NUM_GENS       = 5
POPULATION     = 100
ELITE_RATIO    = 0.10
MIN_KEPT_NODES = 10

USE_NEURAL_FILL    = True
NEURAL_DECODE_TYPE = "sampling"
NEURAL_MAX_NODES   = 120

NEURAL_CKPT_CANDIDATES = [
    os.path.join(BASE_DIR, "checkpoints_neural_fill_v3", "model_best_sampling.pt"),
    os.path.join(BASE_DIR, "Train_Neural", "checkpoints_neural_fill_v2", "model_best_sampling.pt"),
    os.path.join(BASE_DIR, "checkpoints_neural_fill_v2", "model_best_sampling.pt"),
]

OUTPUT_FILE = os.path.join(BASE_DIR, "tune_node_ratio_results_399.txt")

# =========================================================
# IMPORTS
# =========================================================

from read_data import read_data
from GA import GA
from caculate import get_route, get_fitness
from Local_search.local_search import local_search, reset_ls_cache


# =========================================================
# HELPERS
# =========================================================

def resolve_neural_checkpoint():
    if not USE_NEURAL_FILL:
        return None
    for p in NEURAL_CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No neural checkpoint found.")


def get_pop(population, dimension):
    customers = list(range(2, dimension + 1))
    parent = []
    for _ in range(population):
        p = customers.copy()
        random.shuffle(p)
        parent.append(p)
    return parent


def evaluate_population(parent, route, nodes):
    fitness = [(get_fitness(parent[i], route[i], nodes), i) for i in range(len(parent))]
    fitness.sort()
    return fitness


def _select_parent(fitness, pool_ratio=0.20):
    pool = max(2, int(len(fitness) * pool_ratio))
    return fitness[random.randint(0, pool - 1)][1]


# =========================================================
# TRIAL
# =========================================================

def run_trial(instance_path, keep_node_ratio, neural_ckpt, seed=None):
    """
    Chạy một trial: 5 gen GA + 1 lần local search.

    Returns:
        best_fit  : fitness tốt nhất sau LS
        best_rank : rank (1-based, theo fitness trước LS) của cá thể đó
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dimension, capacity, nodes = read_data(instance_path)
    reset_ls_cache()

    parent = get_pop(POPULATION, dimension)
    route  = get_route(parent, dimension, POPULATION, capacity, nodes)
    fitness = evaluate_population(parent, route, nodes)

    elite_count = max(1, int(POPULATION * ELITE_RATIO))

    # ---------- 5 gen GA ----------
    for _gen in range(NUM_GENS):
        fitness.sort(key=lambda x: x[0])
        non_elite = [fitness[j][1] for j in range(elite_count, len(fitness))]

        new_parent, new_route, new_fitness = [], [], []

        # Elitism
        for j in range(elite_count):
            idx = fitness[j][1]
            new_idx = len(new_parent)
            new_parent.append(list(parent[idx]))
            new_route.append(list(route[idx]))
            new_fitness.append((fitness[j][0], new_idx))

        # Crossover
        for _ in range(POPULATION - elite_count):
            par1 = _select_parent(fitness)
            par2 = _select_parent(fitness)
            while par2 == par1:
                par2 = _select_parent(fitness)
            par3 = random.choice(non_elite)

            try:
                child, child_route, child_fit = GA(
                    parent, route, par1, par2, par3,
                    use_neural_fill=USE_NEURAL_FILL,
                    neural_ckpt_path=neural_ckpt,
                    neural_decode_type=NEURAL_DECODE_TYPE,
                    num_good_routes_per_parent=8,
                    keep_node_ratio=keep_node_ratio,
                    min_kept_nodes=MIN_KEPT_NODES,
                    max_kept_nodes_cap=None,
                    selection_trials=5,
                    route_select_temperature=0.8,
                    max_neural_nodes=NEURAL_MAX_NODES,
                    target_neural_nodes=100,
                    min_neural_nodes=30,
                    dimension_value=dimension,
                    capacity_value=capacity,
                    nodes_data=nodes,
                )
            except Exception as e:
                print(f"      [WARN] GA error: {e}")
                idx_fb = fitness[0][1]
                child       = list(parent[idx_fb])
                child_route = list(route[idx_fb])
                child_fit   = fitness[0][0]

            new_idx = len(new_parent)
            new_parent.append(child)
            new_route.append(child_route)
            new_fitness.append((child_fit, new_idx))

        parent, route, fitness = new_parent, new_route, new_fitness

    # ---------- Local search (1 lần) sau gen 5 ----------
    fitness.sort(key=lambda x: x[0])

    # Lưu rank trước LS: idx → rank (1-based)
    pre_ls_rank = {idx: rank + 1 for rank, (_, idx) in enumerate(fitness)}
    pre_ls_best_fit = fitness[0][0]

    parent, route, fitness = local_search(
        parent, capacity, nodes, route, fitness,
        elite_ratio=1.0,
        global_best_fit=pre_ls_best_fit,
        gen=NUM_GENS,
        deadline=float("inf"),
    )

    # Tìm cá thể tốt nhất sau LS
    fitness.sort(key=lambda x: x[0])
    best_fit_after_ls = fitness[0][0]
    best_idx_after_ls = fitness[0][1]

    # Rank trước LS của cá thể đó
    best_rank = pre_ls_rank.get(best_idx_after_ls, -1)

    return best_fit_after_ls, best_rank


# =========================================================
# MAIN TUNING LOOP
# =========================================================

def main():
    neural_ckpt = resolve_neural_checkpoint()
    print(f"Neural checkpoint: {neural_ckpt}\n")

    # results[instance][ratio] = list of (best_fit, best_rank) per trial
    results = {}

    total_combos = len(INSTANCES) * len(KEEP_NODE_RATIOS) * NUM_TRIALS
    done = 0

    for inst_file in INSTANCES:
        inst_path = os.path.join(INSTANCE_DIR, inst_file)
        inst_name = inst_file.replace(".vrp", "")
        results[inst_name] = {}

        for ratio in KEEP_NODE_RATIOS:
            trial_results = []

            for trial in range(NUM_TRIALS):
                done += 1
                seed = 42 + trial * 100
                t0 = time.perf_counter()

                best_fit, best_rank = run_trial(inst_path, ratio, neural_ckpt, seed=seed)
                elapsed = time.perf_counter() - t0

                trial_results.append((best_fit, best_rank))
                print(
                    f"[{done:3d}/{total_combos}] {inst_name} | ratio={ratio:.2f} | "
                    f"trial={trial+1}/{NUM_TRIALS} | fit={best_fit:.0f} | "
                    f"rank={best_rank} | t={elapsed:.1f}s"
                )

            results[inst_name][ratio] = trial_results

    # =========================================================
    # WRITE RESULTS
    # =========================================================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TUNING RESULTS: keep_node_ratio  (5 gen GA + 1 LS)\n")
        f.write(f"Instances  : {', '.join(INSTANCES)}\n")
        f.write(f"Ratios     : {KEEP_NODE_RATIOS}\n")
        f.write(f"Population : {POPULATION} | Gens: {NUM_GENS} | Trials: {NUM_TRIALS}\n")
        f.write("best_rank  : rank (1-based) của cá thể cho best fitness TRƯỚC khi LS\n")
        f.write("=" * 80 + "\n\n")

        for inst_name, ratio_dict in results.items():
            f.write(f"--- {inst_name} ---\n")
            header = f"{'ratio':>6} | {'min_fit':>10} | {'mean_fit':>10} | {'max_fit':>10} | {'mean_rank':>9} | trials (fit, rank)\n"
            f.write(header)
            f.write("-" * 78 + "\n")

            for ratio in KEEP_NODE_RATIOS:
                data = ratio_dict[ratio]
                fits  = [d[0] for d in data]
                ranks = [d[1] for d in data]
                trial_str = "  ".join(f"({f:.0f},{r})" for f, r in data)
                f.write(
                    f"  {ratio:.2f} | {min(fits):>10.0f} | {np.mean(fits):>10.1f} | "
                    f"{max(fits):>10.0f} | {np.mean(ranks):>9.1f} | {trial_str}\n"
                )
            f.write("\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY: ratio cho min(mean_fit) mỗi instance\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Instance':<20} | {'Best ratio':>10} | {'Mean fit':>12} | {'Mean rank':>10}\n")
        f.write("-" * 80 + "\n")

        best_ratios = []
        for inst_name, ratio_dict in results.items():
            best_ratio = min(
                KEEP_NODE_RATIOS,
                key=lambda r: np.mean([d[0] for d in ratio_dict[r]])
            )
            best_mean_fit  = np.mean([d[0] for d in ratio_dict[best_ratio]])
            best_mean_rank = np.mean([d[1] for d in ratio_dict[best_ratio]])
            best_ratios.append(best_ratio)
            f.write(
                f"  {inst_name:<18} | {best_ratio:>10.2f} | "
                f"{best_mean_fit:>12.1f} | {best_mean_rank:>10.1f}\n"
            )

        overall = float(np.median(best_ratios))
        f.write("-" * 80 + "\n")
        f.write(f"  Overall suggested ratio (median of bests): {overall:.2f}\n")
        f.write("=" * 80 + "\n")

    print(f"\nResults saved to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
