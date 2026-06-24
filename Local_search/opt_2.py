from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    rebuild_solution,
    euclid,
    DEPOT_ID,
    ROUTE_PENALTY,
)


EPS = 1e-12


# =========================================================
# 2-OPT FOR ONE ROUTE
# =========================================================

def _two_opt_single_route(customers, nodes):
    """
    Intra-route 2-opt cho một route.
    First improvement, dùng delta O(1).
    """
    customers = list(customers)
    n = len(customers)

    if n < 3:
        return customers, route_distance(customers, nodes), False

    cost = route_distance(customers, nodes)
    changed_any = False

    improved = True
    while improved:
        improved = False
        n = len(customers)

        for i in range(n - 1):
            a = DEPOT_ID if i == 0 else customers[i - 1]
            b = customers[i]

            for j in range(i + 1, n):
                c = customers[j]
                d = DEPOT_ID if j == n - 1 else customers[j + 1]

                delta = (
                    -euclid(nodes, a, b)
                    -euclid(nodes, c, d)
                    + euclid(nodes, a, c)
                    + euclid(nodes, b, d)
                )

                if delta < -EPS:
                    customers[i:j + 1] = reversed(customers[i:j + 1])
                    cost += delta

                    improved = True
                    changed_any = True
                    break

            if improved:
                break

    return customers, cost, changed_any


# =========================================================
# 3-OPT FOR ONE ROUTE
# =========================================================

def _generate_3opt_candidates(route, i, j, k):
    """
    Sinh các cấu hình 3-opt cơ bản.

    route không chứa depot.
    Cắt thành:
        A | B | C | D

    Sau đó thử đảo / đổi vị trí B, C.
    """
    A = route[:i]
    B = route[i:j]
    C = route[j:k]
    D = route[k:]

    candidates = [
        A + B[::-1] + C + D,
        A + B + C[::-1] + D,
        A + B[::-1] + C[::-1] + D,

        A + C + B + D,
        A + C[::-1] + B + D,
        A + C + B[::-1] + D,
        A + C[::-1] + B[::-1] + D,
    ]

    return candidates


def _three_opt_single_route(
    customers,
    nodes,
    max_route_len_for_3opt=40,
    max_rounds=1,
):
    """
    Intra-route 3-opt cho một route.

    Lưu ý:
    - 3-opt nặng hơn 2-opt khá nhiều.
    - Vì route trong CVRP thường không quá dài nên vẫn chạy được.
    - Có giới hạn max_route_len_for_3opt để tránh route quá dài làm chậm.
    """
    customers = list(customers)
    n = len(customers)

    if n < 4:
        return customers, route_distance(customers, nodes), False

    if n > max_route_len_for_3opt:
        return customers, route_distance(customers, nodes), False

    cost = route_distance(customers, nodes)
    changed_any = False

    for _ in range(max_rounds):
        improved = False
        n = len(customers)

        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n + 1):
                    candidates = _generate_3opt_candidates(customers, i, j, k)

                    for cand in candidates:
                        if cand == customers:
                            continue

                        new_cost = route_distance(cand, nodes)

                        if new_cost < cost - EPS:
                            customers = cand
                            cost = new_cost

                            improved = True
                            changed_any = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return customers, cost, changed_any


# =========================================================
# K-OPT MAIN
# =========================================================

def k_opt(
    parent,
    route,
    fitness,
    nodes,
    k=2,
    max_route_len_for_3opt=40,
    three_opt_rounds=1,
):
    """
    Intra-route k-opt.

    k = 2:
        Chạy 2-opt.

    k = 3:
        Chạy 2-opt trước để làm sạch route,
        sau đó chạy thêm 3-opt để tìm cải thiện sâu hơn.

    Hàm này không đổi số route.
    Nó chỉ tối ưu thứ tự customer bên trong từng route.
    """
    if k not in (2, 3):
        raise ValueError("k_opt hiện chỉ hỗ trợ k=2 hoặc k=3")

    routes = separate_routes(parent, route)
    route_costs = []

    for r_idx, customers in enumerate(routes):
        # Bước 1: luôn chạy 2-opt trước
        new_customers, new_cost, _ = _two_opt_single_route(customers, nodes)

        # Bước 2: nếu k=3 thì chạy thêm 3-opt
        if k == 3:
            new_customers, new_cost, _ = _three_opt_single_route(
                new_customers,
                nodes,
                max_route_len_for_3opt=max_route_len_for_3opt,
                max_rounds=three_opt_rounds,
            )

            # Sau 3-opt, chạy lại 2-opt nhẹ để làm mượt
            new_customers, new_cost, _ = _two_opt_single_route(
                new_customers,
                nodes,
            )

        routes[r_idx] = new_customers
        route_costs.append(new_cost)

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

    return new_parent, new_route, new_fitness


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def opt_2(parent, route, fitness, nodes):
    """
    Giữ lại tên cũ để local_search.py không bị lỗi.
    """
    return k_opt(
        parent,
        route,
        fitness,
        nodes,
        k=2,
    )


def opt_3(parent, route, fitness, nodes):
    """
    Gọi 3-opt.
    """
    return k_opt(
        parent,
        route,
        fitness,
        nodes,
        k=3,
        max_route_len_for_3opt=40,
        three_opt_rounds=1,
    )