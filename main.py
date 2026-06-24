import os
import sys
import random
import numpy as np
import time

from read_data import read_data
from GA import GA
from caculate import get_route, get_fitness
from Local_search.local_search import local_search, reset_ls_cache
from Local_search.local_search_utils import (
    route_demand,
    rebuild_solution,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCE_PATH = os.path.join(
    BASE_DIR, "ML4VRP2026", "Instances", "cvrp", "vrp", "X-n401-k29.vrp",
)

OUTPUT_DIR = os.path.join(BASE_DIR, "New_Solutions_final_check")

# Reproducibility (None = không seed).
RANDOM_SEED = None

POPULATION   = 100
ELITE_RATIO  = 0.10

# Loop chạy theo THỜI GIAN; MAX_GENERATIONS chỉ là trần an toàn.
MAX_GENERATIONS = 100_000

# Paper: "local search is applied every five generations".
LOCAL_SEARCH_EVERY        = 5
LOCAL_SEARCH_START_GEN    = 5
LOCAL_SEARCH_ELITE_RATIO  = 0.15
LS_MAX_REPEAT             = 2   # lặp theo improvement (thay cho cứng 3 lần)
LS_MAX_REPEAT_AFTER_RENEW = 3

RENEW_PATIENCE        = 6
RENEW_AFTER_GEN       = 10
MAX_RENEWS            = 4       # 5 phases total (phase 1 + 4 renews)

# ----- Diversity preservation -----
PARENT_POOL_RATIO        = 0.20   # par1/par2 lấy từ top 20% (trước là 10%)
ACCEPT_PERCENTILE_START  = 55.0   # đầu run: nhận con tới ~p55 (nới)
ACCEPT_PERCENTILE_END    = 15.0   # cuối run: chỉ nhận con tới ~p15 (siết)
DEDUP_MUTATION_ATTEMPTS  = 3      # số lần mutate để thoát trùng signature

# ----- Neural fill -----
USE_NEURAL_FILL    = True
NEURAL_DECODE_TYPE = "sampling"
NEURAL_MAX_NODES   = 120          # nâng lên sau khi tích hợp model mới (POMO+ICAM)

NEURAL_CKPT_CANDIDATES = [
    os.path.join(BASE_DIR, "checkpoints_neural_fill_v3", "model_best_sampling.pt"),
    # fallback model cũ (chỉ chạy được nếu CHƯA đổi sang kiến trúc mới)
    os.path.join(BASE_DIR, "Train_Neural", "checkpoints_neural_fill_v2", "model_best_sampling.pt"),
    os.path.join(BASE_DIR, "checkpoints_neural_fill_v2", "model_best_sampling.pt"),
]


# =========================================================
# TIME LIMIT
# =========================================================

def _time_limit_for_dimension(n: int) -> float:
    if n <= 200:
        return 1_800.0
    if n <= 400:
        return 3_600.0
    return 7_200.0


TIME_BUFFER = 90.0


# =========================================================
# ADAPTIVE PARAMS THEO n
# =========================================================

def _adaptive_params(n: int) -> dict:
    # keep_node_ratio = % tổng customer sẽ được kế thừa từ parent
    # min_kept_nodes  = tối thiểu bao nhiêu node phải giữ
    # max_kept_nodes_cap = None nghĩa là không giới hạn cứng (chỉ dùng ratio)
    if n <= 150:
        return dict(selection_trials=15, keep_node_ratio=0.45, min_kept_nodes=20,
                    max_kept_nodes_cap=None, route_select_temperature=0.8)
    if n <= 250:
        return dict(selection_trials=8, keep_node_ratio=0.40, min_kept_nodes=25,
                    max_kept_nodes_cap=None, route_select_temperature=0.8)
    if n <= 400:
        return dict(selection_trials=5, keep_node_ratio=0.50, min_kept_nodes=40,
                    max_kept_nodes_cap=None, route_select_temperature=0.9)
    return dict(selection_trials=3, keep_node_ratio=0.50, min_kept_nodes=60,
                max_kept_nodes_cap=None, route_select_temperature=1.0)


# =========================================================
# HELPERS
# =========================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    sec = seconds % 60
    if minutes < 60:
        return f"{minutes}m {sec:.2f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {sec:.2f}s"


def resolve_neural_checkpoint():
    if not USE_NEURAL_FILL:
        return None
    for ckpt_path in NEURAL_CKPT_CANDIDATES:
        if os.path.exists(ckpt_path):
            return ckpt_path
    raise FileNotFoundError(
        "No neural checkpoint found. Checked:\n" + "\n".join(NEURAL_CKPT_CANDIDATES)
    )


def clone_list(x):
    return x.copy() if hasattr(x, "copy") else list(x)


def _fit_value(x):
    return x[0] if isinstance(x, tuple) else x


def _best_from_fitness(fitness_list):
    if len(fitness_list) == 0:
        return float("inf"), None
    if isinstance(fitness_list[0], tuple):
        fitness_list.sort(key=lambda x: x[0])
        return fitness_list[0][0], fitness_list[0][1]
    best_idx = min(range(len(fitness_list)), key=lambda i: fitness_list[i])
    return fitness_list[best_idx], best_idx


# =========================================================
# DIVERSITY HELPERS (self-contained, không phụ thuộc caculate)
# =========================================================

def _split_routes(parent_i, route_i):
    """Tách parent + route-marker (1 = đầu route mới) thành list các route."""
    routes = []
    cur = []
    for cust, marker in zip(parent_i, route_i):
        if marker == 1 and cur:
            routes.append(cur)
            cur = [cust]
        else:
            cur.append(cust)
    if cur:
        routes.append(cur)
    return routes


def _signature(parent_i, route_i):
    """Signature theo THÀNH VIÊN route (bất biến với thứ tự trong route)."""
    return frozenset(frozenset(r) for r in _split_routes(parent_i, route_i))


def population_diversity(parent, route):
    """Tỉ lệ signature unique / kích thước quần thể (1.0 = tất cả khác nhau)."""
    if not parent:
        return 0.0
    sigs = {_signature(parent[i], route[i]) for i in range(len(parent))}
    return len(sigs) / len(parent)


def feasible_mutation(parent_i, route_i, nodes, capacity, n_moves=1):
    """
    Mutation nhẹ GIỮ FEASIBILITY và ĐỔI SIGNATURE:
    dời n_moves khách sang một route khác còn đủ chỗ (đổi thành viên route).
    Đảo thứ tự nội route KHÔNG đổi signature nên không dùng ở đây.
    """
    routes = [list(r) for r in _split_routes(parent_i, route_i)]
    loads = [route_demand(r, nodes) for r in routes]

    for _ in range(n_moves):
        src = [i for i, r in enumerate(routes) if r]
        if len(routes) < 2 or not src:
            break
        r1 = random.choice(src)
        pos = random.randrange(len(routes[r1]))
        cust = routes[r1][pos]
        d = nodes[cust]["demand"]

        dst = [i for i in range(len(routes)) if i != r1 and loads[i] + d <= capacity]
        if not dst:
            continue
        r2 = random.choice(dst)

        routes[r1].pop(pos)
        loads[r1] -= d
        ins = random.randrange(len(routes[r2]) + 1)
        routes[r2].insert(ins, cust)
        loads[r2] += d

        if not routes[r1]:
            routes.pop(r1)
            loads.pop(r1)

    return rebuild_solution(routes)


def acceptance_threshold(fitness, progress):
    """
    Ngưỡng nhận con = percentile của fitness quần thể, SIẾT dần theo thời gian:
    đầu run p≈55 (nới, giữ đa dạng), cuối run p≈15 (siết, khai thác).
    fitness: list[(fit, idx)]; progress ∈ [0,1].
    """
    progress = min(max(progress, 0.0), 1.0)
    p = ACCEPT_PERCENTILE_START + (ACCEPT_PERCENTILE_END - ACCEPT_PERCENTILE_START) * progress
    vals = [_fit_value(f) for f in fitness]
    return float(np.percentile(vals, p))


# =========================================================
# POPULATION
# =========================================================

def get_pop(population, dimension, nodes=None):
    parent = []
    customers = list(range(2, dimension + 1))
    for _ in range(population):
        temp = customers.copy()
        random.shuffle(temp)
        parent.append(temp)
    return parent


def evaluate_population(parent, route, nodes):
    fitness = [(get_fitness(parent[i], route[i], nodes), i) for i in range(len(parent))]
    fitness.sort()
    return fitness


def update_global_best(fitness, parent, route, best_fit, best_parent, best_route):
    fitness.sort(key=lambda x: x[0] if isinstance(x, tuple) else x)
    current_best_fit, current_best_idx = fitness[0]
    if current_best_fit < best_fit:
        return current_best_fit, clone_list(parent[current_best_idx]), clone_list(route[current_best_idx]), True
    return best_fit, best_parent, best_route, False


def build_index_mapping(fitness):
    return {old_idx: pos for pos, (_, old_idx) in enumerate(fitness)}


def should_renew(stale_count, gen):
    return stale_count >= RENEW_PATIENCE and gen > RENEW_AFTER_GEN


def renew_population(population, dimension, capacity, nodes):
    parent = get_pop(population, dimension, nodes=nodes)
    route = get_route(parent, dimension, population, capacity, nodes)
    fitness = evaluate_population(parent, route, nodes)
    return parent, route, fitness


def _select_parent_from_pool(fitness, pool_ratio):
    pool_size = max(2, int(len(fitness) * pool_ratio))
    return fitness[random.randint(0, pool_size - 1)][1]


def add_individual(new_parent, new_route, new_fitness, individual, individual_route, individual_fit):
    new_idx = len(new_parent)
    new_parent.append(clone_list(individual))
    new_route.append(clone_list(individual_route))
    new_fitness.append((individual_fit, new_idx))


# =========================================================
# SAVE / VALIDATE
# =========================================================

def extract_route_list(best_parent, best_route):
    return _split_routes(best_parent, best_route)


def save_fitness_history(output_dir, instance_name, fitness_history):
    os.makedirs(output_dir, exist_ok=True)
    fitness_file = os.path.join(output_dir, f"{instance_name}_fitness.txt")
    with open(fitness_file, "w") as f:
        for gen_num, fit_val in enumerate(fitness_history, start=1):
            f.write(f"{gen_num}\t{fit_val}\n")
    print(f"Saved fitness history to {fitness_file}")


def _euc_2d(a, b):
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    return int((dx * dx + dy * dy) ** 0.5 + 0.5)


def route_distance_euc_2d(route_customers, nodes):
    if not route_customers:
        return 0
    total = 0
    prev = 1
    for c in route_customers:
        total += _euc_2d(nodes[prev], nodes[c])
        prev = c
    total += _euc_2d(nodes[prev], nodes[1])
    return total


def solution_objective(route_list, nodes, route_penalty=1000):
    total_distance = sum(route_distance_euc_2d(r, nodes) for r in route_list)
    return route_penalty * len(route_list) + total_distance


def validate_solution(route_list, dimension, capacity, nodes):
    """
    Kiểm: mỗi khách 2..dimension xuất hiện ĐÚNG một lần và mọi route <= capacity.
    Trả (ok, message).
    """
    seen = {}
    for r in route_list:
        load = 0
        for c in r:
            seen[c] = seen.get(c, 0) + 1
            load += nodes[c]["demand"]
        if load > capacity + 1e-9:
            return False, f"route vượt capacity ({load} > {capacity})"

    expected = set(range(2, dimension + 1))
    got = set(seen.keys())
    missing = expected - got
    extra = got - expected
    dup = {c: n for c, n in seen.items() if n > 1}
    if missing or extra or dup:
        return False, f"missing={len(missing)} extra={len(extra)} dup={len(dup)}"
    return True, "ok"


def save_best_routes(output_dir, instance_name, route_list, nodes, quiet=False):
    os.makedirs(output_dir, exist_ok=True)
    internal_file = os.path.join(output_dir, f"{instance_name}_routes_internal.txt")
    evaluator_dir = os.path.join(output_dir, "cvrp")
    os.makedirs(evaluator_dir, exist_ok=True)
    evaluator_file = os.path.join(evaluator_dir, f"{instance_name}.txt")

    obj = solution_objective(route_list, nodes, route_penalty=1000)

    with open(internal_file, "w") as f:
        for route_num, rc in enumerate(route_list, start=1):
            f.write(f"Route #{route_num}: {' '.join(map(str, rc))}\n")
        f.write(f"Cost {int(round(obj))}\n")

    with open(evaluator_file, "w") as f:
        for route_num, rc in enumerate(route_list, start=1):
            converted = [c - 1 for c in rc]
            f.write(f"Route #{route_num}: {' '.join(map(str, converted))}\n")

    if not quiet:
        print(f"Saved internal routes to {internal_file}")
        print(f"Saved ML4VRP evaluator solution to {evaluator_file}")


def _save_current_best(best_parent, best_route, instance_name, nodes, quiet=True):
    """Lưu TĂNG DẦN: luôn có một file hợp lệ trên đĩa kể cả khi bị kill."""
    if best_parent is None or best_route is None:
        return
    route_list = extract_route_list(best_parent, best_route)
    save_best_routes(OUTPUT_DIR, instance_name, route_list, nodes, quiet=quiet)


# =========================================================
# MAIN
# =========================================================

def main(instance_path=INSTANCE_PATH):
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    reset_ls_cache()

    algorithm_start_time = time.perf_counter()

    dimension, capacity, nodes = read_data(instance_path)
    instance_name = os.path.basename(instance_path).split(".")[0]

    time_limit = _time_limit_for_dimension(dimension)
    hard_deadline = float('inf')

    ga_params = _adaptive_params(dimension)

    print("=" * 70)
    print(f"Instance       : {instance_name}")
    print(f"Dimension      : {dimension}")
    print(f"Capacity       : {capacity}")
    print(f"Time limit     : {time_limit:.0f}s (buffer {TIME_BUFFER:.0f}s)")
    print(f"Population     : {POPULATION} | Elite ratio: {ELITE_RATIO}")
    print(f"Parent pool    : top {PARENT_POOL_RATIO*100:.0f}%")
    print(f"Accept pctile  : {ACCEPT_PERCENTILE_START:.0f} -> {ACCEPT_PERCENTILE_END:.0f}")
    print(f"LS every       : {LOCAL_SEARCH_EVERY} gens from gen {LOCAL_SEARCH_START_GEN}")
    print(f"GA params      : {ga_params}")
    print(f"Seed           : {RANDOM_SEED}")
    print("=" * 70)

    neural_ckpt_path = resolve_neural_checkpoint()
    if USE_NEURAL_FILL:
        print(f"Neural checkpoint : {neural_ckpt_path}")
        print(f"Neural decode     : {NEURAL_DECODE_TYPE}")

    # Initial population
    parent = get_pop(POPULATION, dimension, nodes=nodes)
    route = get_route(parent, dimension, POPULATION, capacity, nodes)
    fitness = evaluate_population(parent, route, nodes)

    best_fit = np.inf
    best_parent = None
    best_route = None
    best_fitness_history = []

    phase_bests = []
    phase_start_time = algorithm_start_time
    phase_best_fit = np.inf
    renew_count = 0
    stale_count = 0
    ls_no_improve_count = 0
    ls_full_repeat = True
    ls_first_time  = True   # lần LS đầu tiên -> elite_ratio=1.0
    last_gen_time = 0.0

    try:
        for gen in range(1, MAX_GENERATIONS + 1):
            gen_start = time.perf_counter()

            # Global best đầu generation.
            best_fit, best_parent, best_route, improved0 = update_global_best(
                fitness, parent, route, best_fit, best_parent, best_route)
            if improved0:
                _save_current_best(best_parent, best_route, instance_name, nodes)

            # Stale tracking.
            current_pop_best = fitness[0][0]
            if current_pop_best < phase_best_fit - 1e-9:
                phase_best_fit = current_pop_best
                stale_count = 0
            else:
                stale_count += 1

            # Renewal / phase end.
            if should_renew(stale_count, gen):
                phase_elapsed = time.perf_counter() - phase_start_time
                phase_bests.append(_fit_value(best_fit))
                over = "  [!!!OVER TIME LIMIT!!!]" if phase_elapsed > time_limit else ""
                print(f"[PHASE {len(phase_bests)} END] time={format_time(phase_elapsed)}{over} | best={_fit_value(best_fit):.0f}")

                if renew_count >= MAX_RENEWS:
                    print(f"[DONE] Hoàn thành {MAX_RENEWS + 1} phases ({MAX_RENEWS} lần renew). Dừng.")
                    break

                renew_count += 1
                stale_count = 0
                ls_no_improve_count = 0
                phase_start_time = time.perf_counter()
                print(f"[RENEW #{renew_count}] Gen={gen} | global_best={_fit_value(best_fit):.0f}")
                parent, route, fitness = renew_population(POPULATION, dimension, capacity, nodes)
                fitness.sort(key=lambda x: x[0])
                phase_best_fit = fitness[0][0]
                ls_full_repeat = True
                ls_first_time  = True

            fitness.sort(key=lambda x: x[0])
            index_mapping = build_index_mapping(fitness)

            # Ngưỡng nhận con cho generation này.
            progress = min(1.0, renew_count / max(1, MAX_RENEWS))
            tau = acceptance_threshold(fitness, progress)

            new_parent, new_route, new_fitness = [], [], []
            new_sigs = set()

            # ---------- Elitism ----------
            elite_count = int(POPULATION * ELITE_RATIO)
            for j in range(elite_count):
                old_idx = fitness[j][1]
                add_individual(new_parent, new_route, new_fitness,
                               parent[old_idx], route[old_idx], fitness[j][0])
                new_sigs.add(_signature(parent[old_idx], route[old_idx]))

            non_elite_pool = [fitness[j][1] for j in range(elite_count, len(fitness))]

            # ---------- Crossover / neural fill ----------
            for _ in range(POPULATION - elite_count):
                par1 = _select_parent_from_pool(fitness, PARENT_POOL_RATIO)
                par2 = _select_parent_from_pool(fitness, PARENT_POOL_RATIO)
                while par2 == par1:
                    par2 = _select_parent_from_pool(fitness, PARENT_POOL_RATIO)
                par3 = random.choice(non_elite_pool)

                child, child_route, child_fitness = GA(
                    parent, route, par1, par2, par3,
                    use_neural_fill=USE_NEURAL_FILL,
                    neural_ckpt_path=neural_ckpt_path,
                    neural_decode_type=NEURAL_DECODE_TYPE,
                    num_good_routes_per_parent=16,
                    keep_node_ratio=ga_params["keep_node_ratio"],
                    min_kept_nodes=ga_params["min_kept_nodes"],
                    max_kept_nodes_cap=ga_params["max_kept_nodes_cap"],
                    selection_trials=ga_params["selection_trials"],
                    route_select_temperature=ga_params["route_select_temperature"],
                    max_neural_nodes=NEURAL_MAX_NODES,
                    target_neural_nodes=100,
                    min_neural_nodes=30,
                    verbose_neural_chunk=False,
                    dimension_value=dimension,
                    capacity_value=capacity,
                    nodes_data=nodes,
                )

                par_fits = [fitness[index_mapping[p]][0] for p in (par1, par2, par3)]
                best_parent_fit = min(par_fits)

                # (c) Nhận con nếu vượt parent tốt nhất HOẶC không tệ hơn ngưỡng tau.
                if child_fitness < best_parent_fit or child_fitness <= tau:
                    cand_p, cand_r, cand_f = child, child_route, child_fitness
                else:
                    # (a) Fallback: par3 + mutation feasibility-preserving.
                    mp, mr = feasible_mutation(parent[par3], route[par3], nodes, capacity)
                    cand_p, cand_r = mp, mr
                    cand_f = get_fitness(mp, mr, nodes)

                # (b) De-dup theo signature: mutate cho tới khi khác (cap attempts).
                sig = _signature(cand_p, cand_r)
                attempts = 0
                while sig in new_sigs and attempts < DEDUP_MUTATION_ATTEMPTS:
                    cand_p, cand_r = feasible_mutation(cand_p, cand_r, nodes, capacity)
                    cand_f = get_fitness(cand_p, cand_r, nodes)
                    sig = _signature(cand_p, cand_r)
                    attempts += 1

                new_sigs.add(sig)
                add_individual(new_parent, new_route, new_fitness, cand_p, cand_r, cand_f)

            parent, route, fitness = new_parent, new_route, new_fitness

            # ---------- Local search (improvement-bounded) ----------
            ls_time = 0.0
            run_ls = (gen >= LOCAL_SEARCH_START_GEN
                      and (gen - LOCAL_SEARCH_START_GEN) % LOCAL_SEARCH_EVERY == 0)

            if run_ls and time.perf_counter() < hard_deadline:
                ls_start = time.perf_counter()
                ls_cap = LS_MAX_REPEAT_AFTER_RENEW if ls_full_repeat else LS_MAX_REPEAT
                ls_full_repeat = False
                ls_improved_global = False

                ls_elite = 1.0 if ls_first_time else LOCAL_SEARCH_ELITE_RATIO
                ls_first_time = False

                for _ in range(ls_cap):
                    if time.perf_counter() >= hard_deadline:
                        break
                    prev_pop_best = fitness[0][0] if fitness else np.inf

                    parent, route, fitness = local_search(
                        parent, capacity, nodes, route, fitness,
                        elite_ratio=ls_elite,
                        global_best_fit=_fit_value(best_fit),
                        gen=gen,
                        deadline=hard_deadline,
                    )
                    ls_elite = LOCAL_SEARCH_ELITE_RATIO  # vòng lặp sau về bình thường

                    cur_best, cur_idx = _best_from_fitness(fitness)
                    if cur_idx is not None and cur_best < _fit_value(best_fit):
                        best_fit = cur_best
                        best_parent = clone_list(parent[cur_idx])
                        best_route = clone_list(route[cur_idx])
                        ls_improved_global = True
                        _save_current_best(best_parent, best_route, instance_name, nodes)

                    # Dừng lặp LS nếu không cải thiện population best nữa.
                    if not (fitness and fitness[0][0] < prev_pop_best - 1e-9):
                        break

                ls_no_improve_count = 0 if ls_improved_global else ls_no_improve_count + 1
                ls_time = time.perf_counter() - ls_start
                print(f"[LS] Gen={gen} | time={format_time(ls_time)} | ls_no_imp={ls_no_improve_count}")

            # ---------- Update global best after GA + LS ----------
            fitness.sort(key=lambda x: x[0] if isinstance(x, tuple) else x)
            best_fit, best_parent, best_route, improved_any = update_global_best(
                fitness, parent, route, best_fit, best_parent, best_route)
            if improved_any:
                ls_no_improve_count = 0
                _save_current_best(best_parent, best_route, instance_name, nodes)

            end_pop_best = fitness[0][0]
            if end_pop_best < phase_best_fit - 1e-9:
                phase_best_fit = end_pop_best
                stale_count = 0

            best_fitness_history.append(best_fit)

            # ---------- Log ----------
            cur_best, cur_idx = _best_from_fitness(fitness)
            cur_routes = sum(route[cur_idx]) if cur_idx is not None else 0
            glob_routes = sum(best_route) if best_route is not None else 0
            div = population_diversity(parent, route)
            elapsed = time.perf_counter() - algorithm_start_time
            last_gen_time = time.perf_counter() - gen_start

            print(
                f"Gen {gen:04d} | Cur={cur_best:.0f} ({cur_routes}r) | "
                f"Global={_fit_value(best_fit):.0f} ({glob_routes}r) | "
                f"Div={div:.2f} | Stale={stale_count} | Renew={renew_count} | "
                f"gen_t={format_time(last_gen_time)} | "
                f"Elapsed={format_time(elapsed)}/{format_time(time_limit)}"
            )

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Lưu best hiện tại trước khi thoát.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e} — lưu best hiện tại trước khi raise.")
        _save_current_best(best_parent, best_route, instance_name, nodes, quiet=False)
        raise

    # ---------- Final save ----------
    if best_parent is None or best_route is None:
        raise RuntimeError("No feasible best solution found.")

    route_list = extract_route_list(best_parent, best_route)
    ok, msg = validate_solution(route_list, dimension, capacity, nodes)
    if not ok:
        print(f"[WARN] Nghiệm cuối KHÔNG hợp lệ: {msg}")

    save_fitness_history(OUTPUT_DIR, instance_name, best_fitness_history)
    save_best_routes(OUTPUT_DIR, instance_name, route_list, nodes)

    total_time = time.perf_counter() - algorithm_start_time
    # Record last phase if loop ended without triggering should_renew one last time
    if len(phase_bests) <= renew_count:
        phase_elapsed = time.perf_counter() - phase_start_time
        phase_bests.append(_fit_value(best_fit))
        over = "  [!!!OVER TIME LIMIT!!!]" if phase_elapsed > time_limit else ""
        print(f"[PHASE {len(phase_bests)} END] time={format_time(phase_elapsed)}{over} | best={_fit_value(best_fit):.0f}")

    print("=" * 70)
    print(f"Total time   : {format_time(total_time)}")
    print(f"Best fitness : {_fit_value(best_fit):.0f}")
    print(f"Routes       : {len(route_list)}")
    print(f"Valid        : {ok} ({msg})")
    print("-" * 70)
    print(f"{'Phase':<8} {'Best':>10}")
    for i, pb in enumerate(phase_bests, 1):
        marker = " <-- BEST" if pb == min(phase_bests) else ""
        print(f"  {i:<6} {pb:>10.0f}{marker}")
    print("=" * 70)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else INSTANCE_PATH
    main(path)
