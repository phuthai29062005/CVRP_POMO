from itertools import permutations

from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    euclid,
    DEPOT_ID,
    ROUTE_PENALTY,
)


MAX_SOURCE_ROUTE_SIZE = 3
TOP_INSERTIONS_PER_CUSTOMER = 8


def _solution_fitness(routes, nodes):
    """
    F = 1000 * number_of_routes + total_distance
    """
    return (
        len(routes) * ROUTE_PENALTY
        + sum(route_distance(r, nodes) for r in routes)
    )


def _insert_delta(route, customer, pos, nodes):
    """
    Delta distance khi chèn customer vào route tại vị trí pos.
    """
    prev_node = DEPOT_ID if pos == 0 else route[pos - 1]
    next_node = DEPOT_ID if pos == len(route) else route[pos]

    return (
        -euclid(nodes, prev_node, next_node)
        + euclid(nodes, prev_node, customer)
        + euclid(nodes, customer, next_node)
    )


def _get_top_insertions(routes, route_loads, customer, nodes, capacity):
    """
    Tìm các vị trí chèn tốt nhất cho một customer.
    Chỉ giữ TOP_INSERTIONS_PER_CUSTOMER vị trí tốt nhất để code không quá chậm.
    """
    demand = nodes[customer]["demand"]
    candidates = []

    for r_idx, r in enumerate(routes):
        if route_loads[r_idx] + demand > capacity:
            continue

        for pos in range(len(r) + 1):
            delta = _insert_delta(r, customer, pos, nodes)
            candidates.append((delta, r_idx, pos))

    candidates.sort(key=lambda x: x[0])
    return candidates[:TOP_INSERTIONS_PER_CUSTOMER]


def _try_eliminate_small_route(routes, source_idx, nodes, capacity):
    """
    Thử xóa một route nhỏ bằng cách chuyển toàn bộ customer sang các route khác.

    Chỉ xét route có 1 đến 3 customers.
    Trả về:
        success, best_routes, best_fitness
    """
    source_route = list(routes[source_idx])

    if not (1 <= len(source_route) <= MAX_SOURCE_ROUTE_SIZE):
        return False, routes, float("inf")

    # Bỏ route nguồn ra khỏi nghiệm tạm thời
    base_routes = [
        list(r)
        for idx, r in enumerate(routes)
        if idx != source_idx
    ]
    base_loads = [route_demand(r, nodes) for r in base_routes]

    best_routes = None
    best_fitness = float("inf")

    # Với route nhỏ 1–3 khách, thử mọi thứ tự chuyển khách
    for order in set(permutations(source_route)):
        order = list(order)

        def dfs(k, cur_routes, cur_loads):
            nonlocal best_routes, best_fitness

            if k == len(order):
                fit = _solution_fitness(cur_routes, nodes)

                if fit < best_fitness:
                    best_fitness = fit
                    best_routes = [list(r) for r in cur_routes]

                return

            customer = order[k]
            demand = nodes[customer]["demand"]

            insertions = _get_top_insertions(
                cur_routes,
                cur_loads,
                customer,
                nodes,
                capacity,
            )

            for _, r_idx, pos in insertions:
                new_routes = [list(r) for r in cur_routes]
                new_loads = list(cur_loads)

                new_routes[r_idx].insert(pos, customer)
                new_loads[r_idx] += demand

                dfs(k + 1, new_routes, new_loads)

        dfs(0, base_routes, base_loads)

    if best_routes is None:
        return False, routes, float("inf")

    return True, best_routes, best_fitness


def route_elimination(parent, route, fitness, nodes, capacity):
    """
    Best-improvement route elimination.

    Chỉ tập trung vào route nhỏ 1–3 customers.
    Mỗi vòng:
        - thử xóa tất cả route nhỏ
        - tính objective thật: 1000 * routes + distance
        - chọn cách giảm fitness nhiều nhất
        - lặp lại cho đến khi không còn giảm được
    """
    routes = separate_routes(parent, route)

    improved = True

    while improved:
        improved = False

        current_fitness = _solution_fitness(routes, nodes)

        best_candidate_routes = None
        best_candidate_fitness = current_fitness

        # Ưu tiên route nhỏ trước, route ít khách trước
        candidate_indices = sorted(
            [
                idx for idx, r in enumerate(routes)
                if 1 <= len(r) <= MAX_SOURCE_ROUTE_SIZE
            ],
            key=lambda idx: (
                route_demand(routes[idx], nodes),
                len(routes[idx])
            )
        )

        for source_idx in candidate_indices:
            success, new_routes, new_fitness = _try_eliminate_small_route(
                routes=routes,
                source_idx=source_idx,
                nodes=nodes,
                capacity=capacity,
            )

            if not success:
                continue

            if new_fitness < best_candidate_fitness - 1e-12:
                best_candidate_fitness = new_fitness
                best_candidate_routes = new_routes

        if best_candidate_routes is not None:
            routes = best_candidate_routes
            improved = True

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = _solution_fitness(routes, nodes)

    return new_parent, new_route, new_fitness
