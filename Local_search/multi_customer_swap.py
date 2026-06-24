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
MAX_PAIRS           = 25
MAX_PAIRS_2_2       = 10
MAX_ROUTE_NEIGHBORS = 8   # mỗi route chỉ swap với K route gần nhất về địa lý


def _build_neighbor_pairs(routes, nodes, k=MAX_ROUTE_NEIGHBORS):
    """
    Với mỗi route, tìm K route gần nhất dựa trên centroid.
    Trả về list các (r1, r2) unique với r1 < r2.
    Thay thế vòng lặp C(K,2) = K*(K-1)/2 bằng K*neighbors/2 pairs.
    """
    centroids = []
    for r in routes:
        if not r:
            centroids.append((0.0, 0.0))
            continue
        cx = sum(nodes[c]["x"] for c in r) / len(r)
        cy = sum(nodes[c]["y"] for c in r) / len(r)
        centroids.append((cx, cy))

    n_routes = len(routes)
    pairs = set()
    for i in range(n_routes):
        cx, cy = centroids[i]
        dists = []
        for j in range(n_routes):
            if j == i:
                continue
            dx = centroids[j][0] - cx
            dy = centroids[j][1] - cy
            dists.append((dx * dx + dy * dy, j))
        dists.sort()
        for _, j in dists[:k]:
            pairs.add((min(i, j), max(i, j)))

    return list(pairs)


def _limited_pairs(n, max_pairs=MAX_PAIRS):
    count = 0
    for pair in combinations(range(n), 2):
        yield pair
        count += 1
        if count >= max_pairs:
            break


def _remove_at(route, idx):
    return route[:idx] + route[idx + 1:]


def _remove_pair(route, i, j):
    """Xóa 2 phần tử tại i < j (guaranteed)."""
    return route[:i] + route[i + 1:j] + route[j + 1:]


def _sequence_demand(seq, nodes):
    return sum(nodes[c]["demand"] for c in seq)


# ============================================================
# Precompute removal deltas — O(n_total) một lần mỗi round.
# removal_delta[r][k] = delta distance khi xóa customer tại pos k.
# (âm = tiết kiệm distance)
# ============================================================

def _precompute_removal_deltas(routes, nodes):
    result = []
    for r in routes:
        n = len(r)
        deltas = []
        for k in range(n):
            prev = DEPOT_ID if k == 0     else r[k - 1]
            nxt  = DEPOT_ID if k == n - 1 else r[k + 1]
            delta = (
                euclid(nodes, prev, nxt)
                - euclid(nodes, prev, r[k])
                - euclid(nodes, r[k],  nxt)
            )
            deltas.append(delta)
        result.append(deltas)
    return result


def _pair_removal_delta(route, rd, j, k, nodes):
    """
    Delta khi xóa pair (j < k).
    Non-adjacent (k > j+1): sum của 2 single deltas — chính xác vì các edge độc lập.
    Adjacent (k == j+1): tính trực tiếp vì 2 customer chung một edge.
    """
    if k > j + 1:
        return rd[j] + rd[k]
    n = len(route)
    prev = DEPOT_ID if j == 0     else route[j - 1]
    nxt  = DEPOT_ID if k == n - 1 else route[k + 1]
    return (
        euclid(nodes, prev, nxt)
        - euclid(nodes, prev,     route[j])
        - euclid(nodes, route[j], route[k])
        - euclid(nodes, route[k], nxt)
    )


# ============================================================
# _prev_next_for_second — dùng trong _best_insert_sequence
# ============================================================

def _prev_next_for_second(base_route, first, pos1, pos2, n):
    if pos2 == 0:
        prev2 = DEPOT_ID
    elif pos2 <= pos1:
        prev2 = base_route[pos2 - 1]
    elif pos2 == pos1 + 1:
        prev2 = first
    else:
        prev2 = base_route[pos2 - 2]

    if pos2 >= n + 1:
        next2 = DEPOT_ID
    elif pos2 < pos1:
        next2 = base_route[pos2]
    elif pos2 == pos1:
        next2 = first
    else:
        next2 = base_route[pos2 - 1]

    return prev2, next2


def _best_insert_sequence(base_route, seq, nodes, base_cost=None):
    """
    Chèn 1 hoặc 2 customer vào base_route, tìm vị trí delta nhỏ nhất.
    base_cost: truyền vào để tránh gọi route_distance thừa.
    """
    if len(seq) == 0:
        cost = base_cost if base_cost is not None else route_distance(base_route, nodes)
        return list(base_route), cost

    if base_cost is None:
        base_cost = route_distance(base_route, nodes)
    n = len(base_route)

    if len(seq) == 1:
        c = seq[0]
        best_delta = float("inf")
        best_pos = 0
        for pos in range(n + 1):
            prev = DEPOT_ID if pos == 0 else base_route[pos - 1]
            nxt  = DEPOT_ID if pos == n else base_route[pos]
            delta = (
                euclid(nodes, prev, c)
                + euclid(nodes, c,    nxt)
                - euclid(nodes, prev, nxt)
            )
            if delta < best_delta:
                best_delta = delta
                best_pos   = pos
        return base_route[:best_pos] + [c] + base_route[best_pos:], base_cost + best_delta

    # len(seq) == 2
    c1, c2 = seq
    best_delta = float("inf")
    best_pos   = None
    for first, second in ((c1, c2), (c2, c1)):
        for pos1 in range(n + 1):
            prev1 = DEPOT_ID if pos1 == 0 else base_route[pos1 - 1]
            next1 = DEPOT_ID if pos1 == n else base_route[pos1]
            d1 = (
                euclid(nodes, prev1, first)
                + euclid(nodes, first, next1)
                - euclid(nodes, prev1, next1)
            )
            for pos2 in range(n + 2):
                prev2, next2 = _prev_next_for_second(base_route, first, pos1, pos2, n)
                d2 = (
                    euclid(nodes, prev2, second)
                    + euclid(nodes, second, next2)
                    - euclid(nodes, prev2, next2)
                )
                if d1 + d2 < best_delta:
                    best_delta = d1 + d2
                    best_pos   = (first, second, pos1, pos2)

    if best_pos is None:
        return list(base_route), base_cost

    first, second, pos1, pos2 = best_pos
    temp = base_route[:pos1] + [first] + base_route[pos1:]
    return temp[:pos2] + [second] + temp[pos2:], base_cost + best_delta


# ============================================================
# Swap 1-2
# ============================================================

def _apply_swap_1_2_once(routes, route_costs, route_loads, nodes, capacity, removal_deltas, neighbor_pairs):
    for r1, r2 in neighbor_pairs:
        len_r1  = len(routes[r1])
        rd_r1   = removal_deltas[r1]
        pairs_r1 = list(_limited_pairs(len_r1)) if len_r1 >= 2 else []

        len_r2  = len(routes[r2])
        rd_r2   = removal_deltas[r2]
        pairs_r2 = list(_limited_pairs(len_r2)) if len_r2 >= 2 else []
        old_cost = route_costs[r1] + route_costs[r2]

        # --------------------------------------------------
        # Case 1: 1 khách từ r1 ↔ 2 khách từ r2
        # --------------------------------------------------
        if pairs_r2:
            for i in range(len_r1):
                single        = routes[r1][i]
                single_demand = nodes[single]["demand"]
                delta_single  = rd_r1[i]
                base_r1       = None   # lazy: chỉ tạo khi filter pass

                for j, k in pairs_r2:
                    pair_demand = (
                        nodes[routes[r2][j]]["demand"]
                        + nodes[routes[r2][k]]["demand"]
                    )
                    new_load_r1 = route_loads[r1] - single_demand + pair_demand
                    new_load_r2 = route_loads[r2] - pair_demand  + single_demand
                    if new_load_r1 > capacity or new_load_r2 > capacity:
                        continue

                    # O(1) lower-bound filter:
                    # ngay cả nếu insertion không tốn gì, có cải thiện không?
                    delta_pair = _pair_removal_delta(routes[r2], rd_r2, j, k, nodes)
                    if delta_single + delta_pair >= -EPS:
                        continue

                    # Tính đầy đủ chỉ khi filter pass
                    if base_r1 is None:
                        base_r1 = _remove_at(routes[r1], i)

                    base_r2 = _remove_pair(routes[r2], j, k)
                    pair    = [routes[r2][j], routes[r2][k]]

                    new_r1, new_cost_r1 = _best_insert_sequence(
                        base_r1, pair, nodes,
                        base_cost=route_costs[r1] + delta_single,
                    )
                    new_r2, new_cost_r2 = _best_insert_sequence(
                        base_r2, [single], nodes,
                        base_cost=route_costs[r2] + delta_pair,
                    )

                    if new_cost_r1 + new_cost_r2 < old_cost - EPS:
                        routes[r1]      = new_r1
                        routes[r2]      = new_r2
                        route_costs[r1] = new_cost_r1
                        route_costs[r2] = new_cost_r2
                        route_loads[r1] = new_load_r1
                        route_loads[r2] = new_load_r2
                        return True

        # --------------------------------------------------
        # Case 2: 2 khách từ r1 ↔ 1 khách từ r2
        # --------------------------------------------------
        if pairs_r1:
            for i, j in pairs_r1:
                pair_demand   = (
                    nodes[routes[r1][i]]["demand"]
                    + nodes[routes[r1][j]]["demand"]
                )
                delta_pair_r1 = _pair_removal_delta(routes[r1], rd_r1, i, j, nodes)
                base_r1       = None   # lazy

                for k in range(len_r2):
                    single        = routes[r2][k]
                    single_demand = nodes[single]["demand"]

                    new_load_r1 = route_loads[r1] - pair_demand  + single_demand
                    new_load_r2 = route_loads[r2] - single_demand + pair_demand
                    if new_load_r1 > capacity or new_load_r2 > capacity:
                        continue

                    delta_single_r2 = rd_r2[k]
                    if delta_pair_r1 + delta_single_r2 >= -EPS:
                        continue

                    if base_r1 is None:
                        base_r1 = _remove_pair(routes[r1], i, j)

                    base_r2 = _remove_at(routes[r2], k)
                    pair    = [routes[r1][i], routes[r1][j]]

                    new_r1, new_cost_r1 = _best_insert_sequence(
                        base_r1, [single], nodes,
                        base_cost=route_costs[r1] + delta_pair_r1,
                    )
                    new_r2, new_cost_r2 = _best_insert_sequence(
                        base_r2, pair, nodes,
                        base_cost=route_costs[r2] + delta_single_r2,
                    )

                    if new_cost_r1 + new_cost_r2 < old_cost - EPS:
                        routes[r1]      = new_r1
                        routes[r2]      = new_r2
                        route_costs[r1] = new_cost_r1
                        route_costs[r2] = new_cost_r2
                        route_loads[r1] = new_load_r1
                        route_loads[r2] = new_load_r2
                        return True

    return False


# ============================================================
# Swap 2-2
# ============================================================

def _apply_swap_2_2_once(routes, route_costs, route_loads, nodes, capacity, removal_deltas, neighbor_pairs):
    for r1, r2 in neighbor_pairs:
        len_r1 = len(routes[r1])
        if len_r1 < 2:
            continue
        rd_r1    = removal_deltas[r1]
        pairs_r1 = list(_limited_pairs(len_r1))

        len_r2 = len(routes[r2])
        if len_r2 < 2:
            continue
        rd_r2    = removal_deltas[r2]
        pairs_r2 = list(_limited_pairs(len_r2, max_pairs=MAX_PAIRS_2_2))
        old_cost = route_costs[r1] + route_costs[r2]

        for i, j in pairs_r1:
            demand_1      = (
                nodes[routes[r1][i]]["demand"]
                + nodes[routes[r1][j]]["demand"]
            )
            delta_pair_r1 = _pair_removal_delta(routes[r1], rd_r1, i, j, nodes)
            base_r1       = None   # lazy

            for k, l in pairs_r2:
                demand_2 = (
                    nodes[routes[r2][k]]["demand"]
                    + nodes[routes[r2][l]]["demand"]
                )
                new_load_r1 = route_loads[r1] - demand_1 + demand_2
                new_load_r2 = route_loads[r2] - demand_2 + demand_1
                if new_load_r1 > capacity or new_load_r2 > capacity:
                    continue

                delta_pair_r2 = _pair_removal_delta(routes[r2], rd_r2, k, l, nodes)
                if delta_pair_r1 + delta_pair_r2 >= -EPS:
                    continue

                if base_r1 is None:
                    base_r1 = _remove_pair(routes[r1], i, j)

                base_r2 = _remove_pair(routes[r2], k, l)
                pair_1  = [routes[r1][i], routes[r1][j]]
                pair_2  = [routes[r2][k], routes[r2][l]]

                new_r1, new_cost_r1 = _best_insert_sequence(
                    base_r1, pair_2, nodes,
                    base_cost=route_costs[r1] + delta_pair_r1,
                )
                new_r2, new_cost_r2 = _best_insert_sequence(
                    base_r2, pair_1, nodes,
                    base_cost=route_costs[r2] + delta_pair_r2,
                )

                if new_cost_r1 + new_cost_r2 < old_cost - EPS:
                    routes[r1]      = new_r1
                    routes[r2]      = new_r2
                    route_costs[r1] = new_cost_r1
                    route_costs[r2] = new_cost_r2
                    route_loads[r1] = new_load_r1
                    route_loads[r2] = new_load_r2
                    return True

    return False


# ============================================================
# Public entry point
# ============================================================

def multi_customer_swap(
    parent,
    route,
    fitness,
    nodes,
    capacity,
    enable_swap_1_2=True,
    enable_swap_2_2=True,
    max_rounds=1,
):
    routes      = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes)   for r in routes]

    # Precompute neighbor pairs một lần — O(K² log K), dùng lại mọi round
    neighbor_pairs = _build_neighbor_pairs(routes, nodes)

    for _ in range(max_rounds):
        # Precompute removal deltas một lần mỗi round — O(n_total)
        removal_deltas = _precompute_removal_deltas(routes, nodes)

        improved = False

        if enable_swap_1_2:
            improved = _apply_swap_1_2_once(
                routes, route_costs, route_loads, nodes, capacity, removal_deltas, neighbor_pairs
            )

        if not improved and enable_swap_2_2:
            improved = _apply_swap_2_2_once(
                routes, route_costs, route_loads, nodes, capacity, removal_deltas, neighbor_pairs
            )

        if not improved:
            break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)
    return new_parent, new_route, new_fitness
