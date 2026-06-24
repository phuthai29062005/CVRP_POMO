"""
2-opt* (inter-route 2-opt / tail exchange).

Đây là neighborhood DISTANCE inter-route mạnh mà bộ operator cũ KHÔNG có:
các operator cũ chỉ di chuyển/tráo *từng khách*, không ai tráo cả ĐOẠN ĐUÔI
giữa hai route. 2-opt* chính là chỗ classical solver (LKH/HGS) kiếm được
phần distance mà LS chỉ-move-khách bỏ sót (đúng triệu chứng X-n280: route
đã đúng, distance còn lệch ~1.7%).

Move:
    cắt route A sau s1, route B sau s2, nối chéo đuôi:
        A_new = A[:s1] + B[s2:]
        B_new = B[:s2] + A[s1:]
Chỉ 2 cạnh cắt đổi -> delta O(1). Tải mới tính bằng prefix-load O(1).
Nếu một route thành rỗng (merge khả thi) -> xóa route, giảm K (bonus, giúp
gián tiếp cả instance K-bound).
"""

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


def _prefix_loads(route, nodes):
    """pl[s] = tổng demand của route[:s]; pl[len] = tải cả route."""
    pl = [0.0] * (len(route) + 1)
    for s in range(len(route)):
        pl[s + 1] = pl[s] + nodes[route[s]]["demand"]
    return pl


def two_opt_star(parent, route, fitness, nodes, capacity):
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True
    while improved:
        improved = False

        for r1 in range(len(routes)):
            R1 = routes[r1]
            m = len(R1)
            pl1 = _prefix_loads(R1, nodes)
            load1 = route_loads[r1]

            for r2 in range(r1 + 1, len(routes)):
                R2 = routes[r2]
                p = len(R2)
                pl2 = _prefix_loads(R2, nodes)
                load2 = route_loads[r2]

                for s1 in range(m + 1):
                    last_head1 = DEPOT_ID if s1 == 0 else R1[s1 - 1]
                    first_tail1 = DEPOT_ID if s1 == m else R1[s1]
                    head_load1 = pl1[s1]
                    tail_load1 = load1 - head_load1

                    for s2 in range(p + 1):
                        # bỏ no-op / full-swap (không đổi tập cạnh)
                        if (s1 == 0 and s2 == 0) or (s1 == m and s2 == p):
                            continue

                        last_head2 = DEPOT_ID if s2 == 0 else R2[s2 - 1]
                        first_tail2 = DEPOT_ID if s2 == p else R2[s2]
                        head_load2 = pl2[s2]
                        tail_load2 = load2 - head_load2

                        new_load1 = head_load1 + tail_load2
                        new_load2 = head_load2 + tail_load1
                        if new_load1 > capacity or new_load2 > capacity:
                            continue

                        removed = (euclid(nodes, last_head1, first_tail1)
                                   + euclid(nodes, last_head2, first_tail2))
                        added = (euclid(nodes, last_head1, first_tail2)
                                 + euclid(nodes, last_head2, first_tail1))
                        delta = added - removed

                        if delta < -EPS:
                            new_r1 = R1[:s1] + R2[s2:]
                            new_r2 = R2[:s2] + R1[s1:]

                            routes[r1] = new_r1
                            routes[r2] = new_r2
                            route_costs[r1] = route_distance(new_r1, nodes)
                            route_costs[r2] = route_distance(new_r2, nodes)
                            route_loads[r1] = new_load1
                            route_loads[r2] = new_load2

                            # Xóa route rỗng do merge (giảm K). Tối đa 1 route rỗng/move.
                            for rr in (r2, r1):  # xóa index cao trước
                                if not routes[rr]:
                                    del routes[rr]
                                    del route_costs[rr]
                                    del route_loads[rr]

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
