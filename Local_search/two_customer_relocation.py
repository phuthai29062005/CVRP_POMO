import math
from itertools import combinations

from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    euclid,
    DEPOT_ID,
    ROUTE_PENALTY,
)


EPS = 1e-12
MAX_PAIRS = 50


def _adaptive_max_iter(n: int) -> int:
    if n <= 150:
        return 10
    if n <= 250:
        return 5
    if n <= 300:
        return 3
    return 4


def _adaptive_k_nearest(n: int) -> int:
    """Số target route tối đa để thử per pair (k-nearest filtering)."""
    if n <= 250:
        return 999  # không filter
    if n <= 300:
        return 20
    return 15


def _limited_pairs(n, max_pairs=MAX_PAIRS):
    count = 0
    for pair in combinations(range(n), 2):
        yield pair
        count += 1
        if count >= max_pairs:
            break


def _remove_two_indices(route, i, j):
    return [c for idx, c in enumerate(route) if idx != i and idx != j]


def _route_centroid(route, nodes):
    if not route:
        return 0.0, 0.0
    cx = sum(nodes[c]["x"] for c in route) / len(route)
    cy = sum(nodes[c]["y"] for c in route) / len(route)
    return cx, cy


def _get_k_nearest_route_indices(routes, nodes, cx, cy, exclude_idx, k):
    """K route gần nhất với điểm (cx, cy), bỏ qua exclude_idx."""
    if k >= len(routes) - 1:
        return [i for i in range(len(routes)) if i != exclude_idx]

    dists = []
    for i, r in enumerate(routes):
        if i == exclude_idx:
            continue
        rcx, rcy = _route_centroid(r, nodes)
        d = math.sqrt((cx - rcx) ** 2 + (cy - rcy) ** 2)
        dists.append((d, i))

    dists.sort()
    return [i for _, i in dists[:k]]


def _prev_next_for_second(target_route, first, pos1, pos2, n):
    """
    Tính prev/next khi chèn customer thứ 2 vào vị trí pos2 của temp1,
    trong đó temp1 = target_route[:pos1] + [first] + target_route[pos1:] (length n+1).
    Không cần tạo list temp1 thực sự → O(1).
    """
    if pos2 == 0:
        prev2 = DEPOT_ID
    elif pos2 <= pos1:
        prev2 = target_route[pos2 - 1]
    elif pos2 == pos1 + 1:
        prev2 = first
    else:
        prev2 = target_route[pos2 - 2]

    if pos2 >= n + 1:
        next2 = DEPOT_ID
    elif pos2 < pos1:
        next2 = target_route[pos2]
    elif pos2 == pos1:
        next2 = first
    else:
        next2 = target_route[pos2 - 1]

    return prev2, next2


def _best_insert_two(target_route, customers, nodes, base_cost=None):
    """
    Tìm cách chèn 2 customer vào target_route với delta nhỏ nhất.
    Dùng O(1) delta thay vì gọi route_distance đầy đủ cho mỗi combo vị trí.
    Chạy nhanh hơn ~10x so với brute-force route_distance.
    """
    c1, c2 = customers
    n = len(target_route)

    if base_cost is None:
        base_cost = route_distance(target_route, nodes)

    best_delta = float("inf")
    best_pos = None  # (first, second, pos1, pos2)

    for first, second in ((c1, c2), (c2, c1)):
        for pos1 in range(n + 1):
            prev1 = DEPOT_ID if pos1 == 0 else target_route[pos1 - 1]
            next1 = DEPOT_ID if pos1 == n else target_route[pos1]

            d1 = (
                euclid(nodes, prev1, first)
                + euclid(nodes, first, next1)
                - euclid(nodes, prev1, next1)
            )

            for pos2 in range(n + 2):
                prev2, next2 = _prev_next_for_second(target_route, first, pos1, pos2, n)

                d2 = (
                    euclid(nodes, prev2, second)
                    + euclid(nodes, second, next2)
                    - euclid(nodes, prev2, next2)
                )

                total_delta = d1 + d2
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_pos = (first, second, pos1, pos2)

    if best_pos is None:
        return list(target_route), base_cost

    first, second, pos1, pos2 = best_pos
    temp1 = target_route[:pos1] + [first] + target_route[pos1:]
    best_route = temp1[:pos2] + [second] + temp1[pos2:]
    return best_route, base_cost + best_delta


def two_customer_relocation(parent, route, fitness, nodes, capacity):
    """
    Di chuyển 2 customer cùng lúc từ route r1 sang route r2.
    Tối ưu:
    - O(1) delta thay vì O(n) route_distance per insertion combination (~10x faster)
    - K-nearest route filtering: chỉ thử K route gần nhất địa lý (~4-5x faster cho n lớn)
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    n = len(nodes)
    max_iter = _adaptive_max_iter(n)
    k_nearest = _adaptive_k_nearest(n)

    improved = True
    iter_count = 0

    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1

        current_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

        for r1 in range(len(routes)):
            len_r1 = len(routes[r1])
            if len_r1 < 2:
                continue

            pairs_r1 = list(_limited_pairs(len_r1))

            for i, j in pairs_r1:
                customers_to_move = (routes[r1][i], routes[r1][j])
                move_demand = (
                    nodes[customers_to_move[0]]["demand"]
                    + nodes[customers_to_move[1]]["demand"]
                )

                new_r1 = _remove_two_indices(routes[r1], i, j)
                remove_empty_source = len(new_r1) == 0
                new_r1_cost = 0.0 if remove_empty_source else route_distance(new_r1, nodes)

                source_delta = new_r1_cost - route_costs[r1]
                penalty_delta = -ROUTE_PENALTY if remove_empty_source else 0.0

                # K-nearest filtering: chỉ thử route gần với midpoint của 2 customer
                cx = (nodes[customers_to_move[0]]["x"] + nodes[customers_to_move[1]]["x"]) / 2
                cy = (nodes[customers_to_move[0]]["y"] + nodes[customers_to_move[1]]["y"]) / 2
                target_indices = _get_k_nearest_route_indices(
                    routes, nodes, cx, cy, r1, k_nearest
                )

                for r2 in target_indices:
                    if route_loads[r2] + move_demand > capacity:
                        continue

                    new_r2, new_r2_cost = _best_insert_two(
                        routes[r2],
                        customers_to_move,
                        nodes,
                        base_cost=route_costs[r2],
                    )

                    target_delta = new_r2_cost - route_costs[r2]
                    new_fitness = current_fitness + source_delta + target_delta + penalty_delta

                    if new_fitness < current_fitness - EPS:
                        routes[r2] = new_r2
                        route_costs[r2] = new_r2_cost
                        route_loads[r2] += move_demand

                        if remove_empty_source:
                            del routes[r1]
                            del route_costs[r1]
                            del route_loads[r1]
                        else:
                            routes[r1] = new_r1
                            route_costs[r1] = new_r1_cost
                            route_loads[r1] -= move_demand

                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

    return new_parent, new_route, new_fitness
