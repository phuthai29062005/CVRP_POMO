from caculate import separate_routes
from Local_search.local_search_utils import (
    route_distance,
    route_demand,
    rebuild_solution,
    ROUTE_PENALTY,
)


def inter_route_swap(parent, route, fitness, nodes, capacity):
    """
    Swap 1-1 giữa hai route khác nhau.
    Chấp nhận nếu objective giảm.
    Không đổi số route, nhưng có thể giảm distance và cân bằng tải,
    giúp route_elimination sau đó dễ xóa route hơn.
    """
    routes = separate_routes(parent, route)
    route_costs = [route_distance(r, nodes) for r in routes]
    route_loads = [route_demand(r, nodes) for r in routes]

    improved = True

    while improved:
        improved = False

        for r1 in range(len(routes)):
            for r2 in range(r1 + 1, len(routes)):
                for i in range(len(routes[r1])):
                    a = routes[r1][i]
                    demand_a = nodes[a]["demand"]

                    for j in range(len(routes[r2])):
                        b = routes[r2][j]
                        demand_b = nodes[b]["demand"]

                        new_load_r1 = route_loads[r1] - demand_a + demand_b
                        new_load_r2 = route_loads[r2] - demand_b + demand_a

                        if new_load_r1 > capacity or new_load_r2 > capacity:
                            continue

                        new_r1 = list(routes[r1])
                        new_r2 = list(routes[r2])

                        new_r1[i] = b
                        new_r2[j] = a

                        old_cost = route_costs[r1] + route_costs[r2]

                        # Tính mỗi route_distance đúng 1 lần cho mỗi route mới.
                        # Sau khi swap được chấp nhận, dùng lại 2 giá trị này
                        # để cập nhật route_costs, tránh gọi thừa route_distance.
                        new_cost_r1 = route_distance(new_r1, nodes)
                        new_cost_r2 = route_distance(new_r2, nodes)
                        new_cost = new_cost_r1 + new_cost_r2

                        delta = new_cost - old_cost

                        if delta < -1e-12:
                            routes[r1] = new_r1
                            routes[r2] = new_r2

                            route_costs[r1] = new_cost_r1
                            route_costs[r2] = new_cost_r2

                            route_loads[r1] = new_load_r1
                            route_loads[r2] = new_load_r2

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