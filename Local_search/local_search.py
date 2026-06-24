from caculate import get_fitness, separate_routes
from Local_search.opt_2 import k_opt
from Local_search.relocation import relocation

import time
from collections import defaultdict


def _try_import(name, attr):
    try:
        mod = __import__(f"Local_search.{name}", fromlist=[attr])
        return getattr(mod, attr), True
    except Exception as e:  # ImportError hoặc thiếu file
        print(f"[local_search] Cannot import {attr}: {e}")
        return None, False


route_elimination, HAS_ROUTE_ELIMINATION = _try_import("route_elimination", "route_elimination")
inter_route_swap, HAS_INTER_ROUTE_SWAP = _try_import("inter_route_swap", "inter_route_swap")
two_customer_relocation, HAS_TWO_CUSTOMER_RELOCATION = _try_import("two_customer_relocation", "two_customer_relocation")
multi_customer_swap, HAS_MULTI_CUSTOMER_SWAP = _try_import("multi_customer_swap", "multi_customer_swap")
route_reduction, HAS_ROUTE_REDUCTION = _try_import("route_reduction", "route_reduction")
two_opt_star, HAS_TWO_OPT_STAR = _try_import("opt_2_star", "two_opt_star")


EPS = 1e-12

# route_reduction (penalized, đắt) chỉ chạy trên TOP elite mỗi pha LS.
RR_ELITES = 2
# Số vòng VND tối đa cho mỗi nghiệm (descent + giảm K + dọn lại).
MAX_VND_ROUNDS = 3

# =========================================================
# CACHE CỰC-TIỂU-CỤC-BỘ (order-aware)
# =========================================================
_LOCAL_OPTIMA_CACHE = set()
_LS_CACHE_MAX = 20000


def reset_ls_cache():
    """Gọi 1 lần ở đầu mỗi instance trong main (cache không nên chéo instance)."""
    _LOCAL_OPTIMA_CACHE.clear()


def _exact_key(parent_i, route_i):
    return hash((tuple(parent_i), tuple(route_i)))


def _cache_add(parent_i, route_i):
    if len(_LOCAL_OPTIMA_CACHE) >= _LS_CACHE_MAX:
        _LOCAL_OPTIMA_CACHE.clear()
    _LOCAL_OPTIMA_CACHE.add(_exact_key(parent_i, route_i))


def _solution_signature(parent_i, route_i):
    """Membership-signature (bất biến thứ tự) — dùng chọn elite UNIQUE."""
    routes = separate_routes(parent_i, route_i)
    return frozenset(frozenset(r) for r in routes)


def _count_routes(route_marker):
    return sum(route_marker)


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.4f}s"
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}m {sec:.2f}s"


def _init_operator_stats():
    return defaultdict(lambda: {
        "time": 0.0, "calls": 0, "improved": 0, "route_reduced": 0,
        "best_delta": 0.0, "best_after": float("inf"), "new_global_best": 0,
    })


def _apply_operator_timed(op_name, op_func, parent_i, route_i, fit_i, nodes,
                          capacity, stats, global_best_fit=None, gen=None,
                          idx=None, **kwargs):
    before_fit = fit_i
    before_routes = _count_routes(route_i)
    start_time = time.perf_counter()

    if capacity is None:
        new_parent_i, new_route_i, new_fit_i = op_func(parent_i, route_i, fit_i, nodes, **kwargs)
    else:
        new_parent_i, new_route_i, new_fit_i = op_func(parent_i, route_i, fit_i, nodes, capacity, **kwargs)

    elapsed = time.perf_counter() - start_time
    after_routes = _count_routes(new_route_i)
    delta = new_fit_i - before_fit

    stats[op_name]["time"] += elapsed
    stats[op_name]["calls"] += 1

    if new_fit_i < before_fit - EPS:
        stats[op_name]["improved"] += 1
        if delta < stats[op_name]["best_delta"]:
            stats[op_name]["best_delta"] = delta
            stats[op_name]["best_after"] = new_fit_i
        if after_routes < before_routes:
            stats[op_name]["route_reduced"] += 1
            print(f"[LS ROUTE--] Gen={gen} | op={op_name} | idx={idx} | "
                  f"{before_routes}->{after_routes} | fit {before_fit:.0f}->{new_fit_i:.0f}")

    if global_best_fit is not None and new_fit_i < global_best_fit - EPS:
        stats[op_name]["new_global_best"] += 1
        print(f"[LS BEST] Gen={gen} | op={op_name} | idx={idx} | "
              f"{global_best_fit:.0f}->{new_fit_i:.0f} | routes={after_routes}")
    return new_parent_i, new_route_i, new_fit_i


def _print_local_search_report(stats, total_time, before_best, after_best,
                               processed, skipped, gen=None):
    print("-" * 78)
    print(f"[LS SUMMARY] Gen={gen} | {before_best:.0f} -> {after_best:.0f} | "
          f"gain={before_best - after_best:.0f} | "
          f"LS'd={processed} skip(cache)={skipped} | time={_format_time(total_time)}")
    print(f"{'Operator':<30} {'Time':>10} {'Imp':>5} {'R--':>5} {'Best':>5} {'BestDelta':>10}")
    for op_name, s in sorted(stats.items(), key=lambda kv: kv[1]["time"], reverse=True):
        print(f"{op_name:<30} {_format_time(s['time']):>10} {s['improved']:>5} "
              f"{s['route_reduced']:>5} {s['new_global_best']:>5} {s['best_delta']:>10.0f}")
    print("-" * 78)


def _build_fitness_by_idx(fitness, pop_size, parent, route, nodes):
    fitness_by_idx = {}
    for fit, idx in fitness:
        if 0 <= idx < pop_size:
            fitness_by_idx[idx] = fit
    for idx in range(pop_size):
        if idx not in fitness_by_idx:
            fitness_by_idx[idx] = get_fitness(parent[idx], route[idx], nodes)
    return fitness_by_idx


def _make_sorted_fitness(fitness_by_idx):
    new_fitness = [(fit, idx) for idx, fit in fitness_by_idx.items()]
    new_fitness.sort(key=lambda x: x[0])
    return new_fitness


# =========================================================
# RVND-ish CHAIN, có VÒNG NGOÀI (outer VND loop)
# =========================================================
def _run_operator_chain(parent, route, fitness_by_idx, idx, fit_val,
                        nodes, capacity, stats, global_best_fit, gen, K_OPT_VALUE,
                        with_route_reduction=False, deadline=None,
                        max_vnd_rounds=MAX_VND_ROUNDS):
    """
    Thứ tự (rẻ -> đắt, intra -> inter, distance -> giảm K):
      1. 2-opt (intra cleanup)
      2. relocation        (inter, 1 khách)
      3. two_opt_star      (inter, TRÁO ĐUÔI — neighborhood distance mạnh, MỚI)
      4. inter_route_swap  (inter, 1-1)
      5. two_customer_relocation
      6. multi_customer_swap
      7. 2-opt (dọn lại sau move inter-route)
      8. route_reduction   (giảm K, top elite, penalized)
      9. route_elimination (giảm K)
    Lặp cả chuỗi tới khi fitness không giảm nữa (tối đa max_vnd_rounds).
    -> Nghiệm sau khi GIẢM K được DỌN LẠI distance ở vòng kế (vá đúng nhược
       điểm cũ: nghiệm K-1 thô bị cache mà không được polish).
    """
    def _upd(gb, f):
        return f if (gb is not None and f < gb) else gb

    def _past_deadline():
        return deadline is not None and time.perf_counter() >= deadline

    try_rr = with_route_reduction

    for _ in range(max_vnd_rounds):
        round_start_fit = fit_val

        parent[idx], route[idx], fit_val = _apply_operator_timed(
            f"k_opt_{K_OPT_VALUE}_early", k_opt, parent[idx], route[idx], fit_val,
            nodes, None, stats, global_best_fit, gen, idx, k=K_OPT_VALUE)
        global_best_fit = _upd(global_best_fit, fit_val)
        if _past_deadline():
            break

        parent[idx], route[idx], fit_val = _apply_operator_timed(
            "relocation", relocation, parent[idx], route[idx], fit_val,
            nodes, capacity, stats, global_best_fit, gen, idx)
        global_best_fit = _upd(global_best_fit, fit_val)
        if _past_deadline():
            break

        if HAS_TWO_OPT_STAR:
            parent[idx], route[idx], fit_val = _apply_operator_timed(
                "two_opt_star", two_opt_star, parent[idx], route[idx], fit_val,
                nodes, capacity, stats, global_best_fit, gen, idx)
            global_best_fit = _upd(global_best_fit, fit_val)
            if _past_deadline():
                break

        if HAS_INTER_ROUTE_SWAP:
            parent[idx], route[idx], fit_val = _apply_operator_timed(
                "inter_route_swap", inter_route_swap, parent[idx], route[idx], fit_val,
                nodes, capacity, stats, global_best_fit, gen, idx)
            global_best_fit = _upd(global_best_fit, fit_val)

        if HAS_TWO_CUSTOMER_RELOCATION:
            parent[idx], route[idx], fit_val = _apply_operator_timed(
                "two_customer_relocation", two_customer_relocation, parent[idx], route[idx], fit_val,
                nodes, capacity, stats, global_best_fit, gen, idx)
            global_best_fit = _upd(global_best_fit, fit_val)

        if HAS_MULTI_CUSTOMER_SWAP:
            parent[idx], route[idx], fit_val = _apply_operator_timed(
                "multi_customer_swap_1_2", multi_customer_swap, parent[idx], route[idx], fit_val,
                nodes, capacity, stats, global_best_fit, gen, idx,
                enable_swap_1_2=True, enable_swap_2_2=True, max_rounds=1)
            global_best_fit = _upd(global_best_fit, fit_val)

        parent[idx], route[idx], fit_val = _apply_operator_timed(
            f"k_opt_{K_OPT_VALUE}_late", k_opt, parent[idx], route[idx], fit_val,
            nodes, None, stats, global_best_fit, gen, idx, k=K_OPT_VALUE)
        global_best_fit = _upd(global_best_fit, fit_val)
        if _past_deadline():
            break

        # --- giảm K ---
        k_before_rr = _count_routes(route[idx])
        if try_rr and HAS_ROUTE_REDUCTION:
            parent[idx], route[idx], fit_val = _apply_operator_timed(
                "route_reduction", route_reduction, parent[idx], route[idx], fit_val,
                nodes, capacity, stats, global_best_fit, gen, idx)
            global_best_fit = _upd(global_best_fit, fit_val)
            # chỉ tiếp tục thử RR ở vòng sau nếu nó CÒN giảm được K (tránh phí khi bế tắc)
            try_rr = _count_routes(route[idx]) < k_before_rr
        if _past_deadline():
            break

        if HAS_ROUTE_ELIMINATION:
            parent[idx], route[idx], fit_val = _apply_operator_timed(
                "route_elimination", route_elimination, parent[idx], route[idx], fit_val,
                nodes, capacity, stats, global_best_fit, gen, idx)
            global_best_fit = _upd(global_best_fit, fit_val)

        # Vòng ngoài: dừng khi cả vòng không cải thiện.
        if fit_val >= round_start_fit - EPS:
            break

    fitness_by_idx[idx] = fit_val
    return global_best_fit


def local_search(parent, capacity, nodes, route, fitness,
                 elite_ratio=0.15, global_best_fit=None, gen=None, deadline=None):
    ls_start_time = time.perf_counter()

    pop_size = len(parent)
    if pop_size == 0:
        return parent, route, []

    elite_count = max(1, int(pop_size * elite_ratio))
    fitness_by_idx = _build_fitness_by_idx(fitness, pop_size, parent, route, nodes)
    fitness_sorted = _make_sorted_fitness(fitness_by_idx)
    before_ls_best = fitness_sorted[0][0]

    seen_sigs = set()
    elite_items = []
    for fit_val, idx in fitness_sorted:
        sig = _solution_signature(parent[idx], route[idx])
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            elite_items.append((fit_val, idx))
        if len(elite_items) >= elite_count:
            break

    K_OPT_VALUE = 2
    stats = _init_operator_stats()
    processed = 0
    skipped = 0
    ls_ref = before_ls_best  # [LS BEST] so với best đầu lần LS này, không phải global all-time

    print("=" * 90)
    print(f"[LOCAL SEARCH START] Gen={gen} | "
          f"unique_elites={len(elite_items)}/{elite_count} target | "
          f"before_best={before_ls_best:.4f}")
    print("=" * 90)

    for rank, (fit_val, idx) in enumerate(elite_items):
        if deadline is not None and time.perf_counter() >= deadline:
            print(f"[LS] Gen={gen} | dừng sớm theo deadline sau {processed} nghiệm.")
            break

        if _exact_key(parent[idx], route[idx]) in _LOCAL_OPTIMA_CACHE:
            skipped += 1
            continue

        ls_ref = _run_operator_chain(
            parent, route, fitness_by_idx, idx, fit_val,
            nodes, capacity, stats, ls_ref, gen, K_OPT_VALUE,
            with_route_reduction=(rank < RR_ELITES), deadline=deadline)

        _cache_add(parent[idx], route[idx])
        processed += 1

    new_fitness = _make_sorted_fitness(fitness_by_idx)
    after_ls_best = new_fitness[0][0]
    total_ls_time = time.perf_counter() - ls_start_time

    _print_local_search_report(stats, total_ls_time, before_ls_best, after_ls_best,
                               processed, skipped, gen)
    return parent, route, new_fitness
