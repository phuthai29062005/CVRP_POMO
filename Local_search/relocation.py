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


def _same_route_reloc_delta(route, i, j, nodes):
    """
    Delta cho relocation trong cùng route O(1).

    Quy ước j giống code cũ:
        - remove route[i] trước
        - insert node đó vào vị trí j của route sau khi remove

    Không dựng route mới và không gọi route_distance trong lúc thử move.
    """
    if i == j:
        return 0.0

    n = len(route)
    b = route[i]

    # Xóa b khỏi route: a - b - c  ->  a - c
    a = DEPOT_ID if i == 0 else route[i - 1]
    c = DEPOT_ID if i == n - 1 else route[i + 1]

    delta_remove = (
        -euclid(nodes, a, b)
        -euclid(nodes, b, c)
        + euclid(nodes, a, c)
    )

    # Chèn b vào vị trí j của route sau khi đã remove b.
    # temp = route[:i] + route[i + 1:]
    # new_route = temp[:j] + [b] + temp[j:]
    if j < i:
        # Chèn trước route[j], giữa route[j - 1] và route[j]
        u = DEPOT_ID if j == 0 else route[j - 1]
        v = route[j]
    else:
        # j > i vì j == i đã return ở trên.
        # Sau khi remove, vị trí j nằm giữa route[j] và route[j + 1].
        u = route[j]
        v = DEPOT_ID if j == n - 1 else route[j + 1]

    delta_insert = (
        -euclid(nodes, u, v)
        + euclid(nodes, u, b)
        + euclid(nodes, b, v)
    )

    return delta_remove + delta_insert


def _apply_same_route_reloc(route, i, j):
    """
    Dựng route mới chỉ khi same-route move đã được accept.
    """
    node = route[i]
    temp = route[:i] + route[i + 1:]
    return temp[:j] + [node] + temp[j:]


def _inter_route_reloc_delta(route1, i, route2, j, nodes):
    """
    Di chuyển route1[i] sang route2 tại vị trí j.

    Trả về riêng:
        delta_remove: thay đổi distance của route nguồn route1
        delta_insert: thay đổi distance của route đích route2

    Nhờ tách riêng 2 phần này, khi accept move có thể cập nhật:
        route_costs[r1] += delta_remove
        route_costs[r2] += delta_insert

    Không cần gọi lại route_distance(new_r1, nodes) hoặc route_distance(new_r2, nodes).
    """
    b = route1[i]

    # Xóa b khỏi route1: a - b - c  ->  a - c
    a = DEPOT_ID if i == 0 else route1[i - 1]
    c = DEPOT_ID if i == len(route1) - 1 else route1[i + 1]

    delta_remove = (
        -euclid(nodes, a, b)
        -euclid(nodes, b, c)
        + euclid(nodes, a, c)
    )

    # Chèn b vào route2 tại vị trí j: u - v  ->  u - b - v
    u = DEPOT_ID if j == 0 else route2[j - 1]
    v = DEPOT_ID if j == len(route2) else route2[j]

    delta_insert = (
        -euclid(nodes, u, v)
        + euclid(nodes, u, b)
        + euclid(nodes, b, v)
    )

    return delta_remove, delta_insert


def relocation(parent, route, fitness, nodes, capacity):
    """
    Relocation 1-customer, first improvement.

    Có 2 loại move:
        1. Same-route relocation:
            - di chuyển 1 customer trong cùng route
            - chỉ làm giảm distance
            - route_load không đổi

        2. Inter-route relocation:
            - di chuyển 1 customer từ route r1 sang route r2
            - check capacity
            - nếu route nguồn rỗng thì xóa route và trừ ROUTE_PENALTY

    Tối ưu quan trọng:
        - same-route delta tính O(1)
        - inter-route delta trả riêng delta_remove và delta_insert
        - sau khi accept inter-route move, cập nhật route_costs bằng delta
          thay vì gọi lại route_distance()
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True

    while improved:
        improved = False

        for r1 in range(len(routes)):
            for i in range(len(routes[r1])):
                node_to_move = routes[r1][i]
                demand = nodes[node_to_move]["demand"]

                for r2 in range(len(routes)):
                    # =====================================================
                    # Case 1: relocation trong cùng route
                    # =====================================================
                    if r1 == r2:
                        route_len = len(routes[r1])

                        for j in range(route_len):
                            if j == i:
                                continue

                            delta = _same_route_reloc_delta(
                                routes[r1],
                                i,
                                j,
                                nodes,
                            )

                            if delta < -EPS:
                                routes[r1] = _apply_same_route_reloc(
                                    routes[r1],
                                    i,
                                    j,
                                )
                                route_costs[r1] += delta
                                improved = True
                                break

                        if improved:
                            break

                    # =====================================================
                    # Case 2: relocation giữa 2 route khác nhau
                    # =====================================================
                    else:
                        if route_loads[r2] + demand > capacity:
                            continue

                        # delta_remove và penalty_delta chỉ phụ thuộc vào (r1, i),
                        # không đổi theo j → tính 1 lần trước j-loop.
                        b = routes[r1][i]
                        a = DEPOT_ID if i == 0 else routes[r1][i - 1]
                        c = DEPOT_ID if i == len(routes[r1]) - 1 else routes[r1][i + 1]
                        delta_remove = (
                            -euclid(nodes, a, b)
                            - euclid(nodes, b, c)
                            + euclid(nodes, a, c)
                        )
                        remove_empty_source = len(routes[r1]) == 1
                        penalty_delta = -ROUTE_PENALTY if remove_empty_source else 0.0

                        for j in range(len(routes[r2]) + 1):
                            u = DEPOT_ID if j == 0 else routes[r2][j - 1]
                            v = DEPOT_ID if j == len(routes[r2]) else routes[r2][j]
                            delta_insert = (
                                -euclid(nodes, u, v)
                                + euclid(nodes, u, b)
                                + euclid(nodes, b, v)
                            )

                            total_delta = delta_remove + delta_insert + penalty_delta

                            if total_delta < -EPS:
                                new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                                new_r2 = routes[r2][:j] + [node_to_move] + routes[r2][j:]

                                # Cập nhật route đích bằng delta_insert, không gọi route_distance.
                                routes[r2] = new_r2
                                route_costs[r2] += delta_insert
                                route_loads[r2] += demand

                                if remove_empty_source:
                                    # Route nguồn chỉ có 1 customer nên sau khi move sẽ rỗng.
                                    # Xóa luôn route nguồn. Không cần route_costs[r1] += delta_remove
                                    # vì xóa route_costs[r1] tương đương loại toàn bộ cost của route đó.
                                    del routes[r1]
                                    del route_costs[r1]
                                    del route_loads[r1]
                                else:
                                    # Cập nhật route nguồn bằng delta_remove, không gọi route_distance.
                                    routes[r1] = new_r1
                                    route_costs[r1] += delta_remove
                                    route_loads[r1] -= demand

                                improved = True
                                break

                        if improved:
                            break

                if improved:
                    break

            if improved:
                break

    new_parent, new_route = rebuild_solution(routes)
    new_fitness = len(routes) * ROUTE_PENALTY + sum(route_costs)

    return new_parent, new_route, new_fitness