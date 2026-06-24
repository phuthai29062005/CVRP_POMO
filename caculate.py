import numpy as np
import random

def get_dist(nodes, a, b, dist_matrix=None):
    if dist_matrix is not None:
        return int(dist_matrix[a, b])

    dx = nodes[a]["x"] - nodes[b]["x"]
    dy = nodes[a]["y"] - nodes[b]["y"]

    return int(np.sqrt(dx * dx + dy * dy) + 0.5)
    
def get_route(parent, dimension, population, capacity, nodes):
    """
    Tạo route markers cho mỗi parent
    
    Args:
        parent: Danh sách parents (mỗi parent là hoán vị khách hàng từ 2-101)
        dimension: 101 (bao gồm depot)
        population: Số lượng parents
        capacity: Dung tích xe
        nodes: Dictionary chứa info các nút
    
    Returns:
        Danh sách route markers [1, 0, 0, 1, 0, ...]
    """
    route = []
    for i in range(population):
        permutation = parent[i].copy()  # [2, 3, 4, ..., 101]
        first_route = []
        current_load = nodes[permutation[0]]['demand']
        first_route.append(1)  # Bắt đầu route mới với khách hàng đầu tiên
        
        for j in range(1, len(permutation)):
            if current_load + nodes[permutation[j]]['demand'] <= capacity:
                first_route.append(0)  # Tiếp tục route hiện tại
                current_load += nodes[permutation[j]]['demand']
            else:
                first_route.append(1)  # Bắt đầu route mới
                current_load = nodes[permutation[j]]['demand']
        
        route.append(first_route)
    return route

def get_fitness(parent, route, nodes, dist_matrix=None):
    routes = separate_routes(parent, route)

    total_distance = 0
    for route_customers in routes:
        total_distance += calculate_route_distance(
            route_customers,
            nodes,
            dist_matrix,
        )

    num_routes = len(routes)
    fitness = 1000 * num_routes + total_distance
    return fitness


def separate_routes(parent, route):
    """
    Tách các route riêng lẻ từ parent và route marker
    
    Args:
        parent: Danh sách khách hàng [c1, c2, c3, ...]
        route: Danh sách marker [1, 0, 0, 1, 0, 1, ...] (1 = bắt đầu route mới)
    
    Returns:
        Danh sách các route: [[c1, c2], [c3], [c4, c5, ...]]
    """
    routes = []
    current_route = []
    
    for i in range(len(parent)):
        if route[i] == 1:  # Bắt đầu route mới
            if current_route:  # Nếu có route trước đó, lưu lại
                routes.append(current_route)
            current_route = [parent[i]]
        else:  # Tiếp tục route hiện tại
            current_route.append(parent[i])
    
    # Lưu route cuối cùng
    if current_route:
        routes.append(current_route)
    
    return routes


def calculate_route_distance(route_customers, nodes, dist_matrix=None):
    distance = 0

    distance += get_dist(nodes, 1, route_customers[0], dist_matrix)

    for i in range(1, len(route_customers)):
        distance += get_dist(
            nodes,
            route_customers[i - 1],
            route_customers[i],
            dist_matrix,
        )

    distance += get_dist(nodes, route_customers[-1], 1, dist_matrix)

    return distance


def get_good_routes(parent, route, nodes, num_good_routes=5, dist_matrix=None):
    routes = separate_routes(parent, route)

    route_scores = []
    for r in routes:
        distance = calculate_route_distance(r, nodes, dist_matrix)
        num_customers = len(r)
        score = (1000 + distance) / num_customers
        route_scores.append((score, r))

    route_scores.sort(key=lambda x: x[0])
    good_routes_with_scores = route_scores[:num_good_routes]

    good_routes = [r[1] for r in good_routes_with_scores]
    good_scores = [r[0] for r in good_routes_with_scores]

    return good_routes, good_scores


def get_all_good_routes(parents, routes, nodes, num_good_routes=5):
    """
    Lấy các route tốt nhất từ tất cả parents
    
    Args:
        parents: Danh sách tất cả parent (hoán vị)
        routes: Danh sách route marker tương ứng
        nodes: Dictionary chứa tọa độ các nút
        num_good_routes: Số route tốt nhất từ mỗi parent
    
    Returns:
        Danh sách tất cả good routes: [[route1], [route2], ...]
    """
    all_good_routes = []
    
    for i in range(len(parents)):
        good_routes, good_scores = get_good_routes(parents[i], routes[i], nodes, num_good_routes)
        all_good_routes.extend(good_routes)
    
    return all_good_routes