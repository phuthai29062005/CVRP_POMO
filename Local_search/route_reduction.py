"""
Route reduction qua PENALIZED INFEASIBILITY (kiểu HGS).

Ý tưởng (chìa khóa để vượt rào K -> K-1 trên instance chật):
  1. Bỏ một route, rải khách của nó vào các route khác — CHO PHÉP quá tải tạm thời.
  2. Chạy local search "chịu quá tải": chấm move bằng
         distance + lambda * (tổng phần vượt capacity),
     với lambda TĂNG DẦN khi còn kẹt quá tải -> ép về khả thi.
  3. Chỉ NHẬN kết quả nếu nó FEASIBLE và fitness thật (1000*K + distance) tốt hơn.

LS feasible-only thông thường không qua được rào này vì bước "tạo chỗ" đầu
tiên luôn tạm thời quá tải hoặc xấu hơn -> bị reject.
"""

from caculate import get_fitness
from Local_search.local_search_utils import (
    euc_2d,
    route_demand,
    rebuild_solution,
    DEPOT_ID,
)

EPS = 1e-9


def _split_routes(parent_i, route_i):
    routes, cur = [], []
    for c, m in zip(parent_i, route_i):
        if m == 1 and cur:
            routes.append(cur)
            cur = [c]
        else:
            cur.append(c)
    if cur:
        routes.append(cur)
    return routes


def _neighbors(route, idx):
    prev = route[idx - 1] if idx > 0 else DEPOT_ID
    nxt = route[idx + 1] if idx + 1 < len(route) else DEPOT_ID
    return prev, nxt


def _insertion_delta(nodes, route, pos, c):
    prev = route[pos - 1] if pos > 0 else DEPOT_ID
    nxt = route[pos] if pos < len(route) else DEPOT_ID
    return (euc_2d(nodes, prev, c) + euc_2d(nodes, c, nxt)
            - euc_2d(nodes, prev, nxt))


def _best_insertion(nodes, route, c):
    best_pos, best_delta = 0, float("inf")
    for pos in range(len(route) + 1):
        d = _insertion_delta(nodes, route, pos, c)
        if d < best_delta:
            best_delta, best_pos = d, pos
    return best_pos, best_delta


def _removal_gain(nodes, route, idx):
    c = route[idx]
    prev, nxt = _neighbors(route, idx)
    return (euc_2d(nodes, prev, c) + euc_2d(nodes, c, nxt)
            - euc_2d(nodes, prev, nxt))


def _ov(load, cap):
    return load - cap if load > cap else 0.0


def _total_overload(L, cap):
    return sum(_ov(x, cap) for x in L)


def _drop_empty(R, L):
    keep = [i for i in range(len(R)) if R[i]]
    if len(keep) != len(R):
        R[:] = [R[i] for i in keep]
        L[:] = [L[i] for i in keep]


def _penalized_relocate(nodes, R, L, cap, lam):
    """Dời 1 khách từ route quá tải sang route khác nếu giảm (distance + lam*overload)."""
    over = [i for i in range(len(R)) if L[i] > cap + EPS]
    for r1 in over:
        for idx in range(len(R[r1])):
            c = R[r1][idx]
            dc = nodes[c]["demand"]
            gain = _removal_gain(nodes, R[r1], idx)
            ov1_old = _ov(L[r1], cap)
            ov1_new = _ov(L[r1] - dc, cap)
            for r2 in range(len(R)):
                if r2 == r1:
                    continue
                pos, ins = _best_insertion(nodes, R[r2], c)
                ov2_old = _ov(L[r2], cap)
                ov2_new = _ov(L[r2] + dc, cap)
                d_dist = ins - gain
                d_over = (ov1_new - ov1_old) + (ov2_new - ov2_old)
                if d_dist + lam * d_over < -EPS:
                    R[r1].pop(idx)
                    L[r1] -= dc
                    R[r2].insert(pos, c)
                    L[r2] += dc
                    return True
    return False


def _penalized_swap(nodes, R, L, cap, lam):
    """Tráo (in-place) một khách lớn ở route quá tải lấy một khách nhỏ hơn ở route khác."""
    over = [i for i in range(len(R)) if L[i] > cap + EPS]
    for r1 in over:
        for i1 in range(len(R[r1])):
            c1 = R[r1][i1]
            d1 = nodes[c1]["demand"]
            p1, n1 = _neighbors(R[r1], i1)
            for r2 in range(len(R)):
                if r2 == r1:
                    continue
                for i2 in range(len(R[r2])):
                    c2 = R[r2][i2]
                    d2 = nodes[c2]["demand"]
                    if d2 >= d1:
                        continue  # chỉ tráo để GIẢM tải r1
                    p2, n2 = _neighbors(R[r2], i2)
                    dr1 = (euc_2d(nodes, p1, c2) + euc_2d(nodes, c2, n1)
                           - euc_2d(nodes, p1, c1) - euc_2d(nodes, c1, n1))
                    dr2 = (euc_2d(nodes, p2, c1) + euc_2d(nodes, c1, n2)
                           - euc_2d(nodes, p2, c2) - euc_2d(nodes, c2, n2))
                    nl1 = L[r1] - d1 + d2
                    nl2 = L[r2] - d2 + d1
                    d_over = ((_ov(nl1, cap) - _ov(L[r1], cap))
                              + (_ov(nl2, cap) - _ov(L[r2], cap)))
                    if (dr1 + dr2) + lam * d_over < -EPS:
                        R[r1][i1] = c2
                        R[r2][i2] = c1
                        L[r1] = nl1
                        L[r2] = nl2
                        return True
    return False


def _try_remove_route(nodes, routes, t, cap, max_iters, lam0, lam_max):
    """Bỏ route t, rải khách (cho quá tải), rồi repair penalized về feasible."""
    R = [list(r) for i, r in enumerate(routes) if i != t]
    L = [route_demand(r, nodes) for r in R]
    removed = list(routes[t])

    # Greedy best-insertion (cho phép quá tải).
    for c in removed:
        best_r, best_pos, best_delta = 0, 0, float("inf")
        for ri in range(len(R)):
            pos, delta = _best_insertion(nodes, R[ri], c)
            if delta < best_delta:
                best_delta, best_r, best_pos = delta, ri, pos
        R[best_r].insert(best_pos, c)
        L[best_r] += nodes[c]["demand"]

    lam = lam0
    for _ in range(max_iters):
        _drop_empty(R, L)
        if _total_overload(L, cap) <= EPS:
            return R  # đã feasible với ít route hơn
        moved = (_penalized_relocate(nodes, R, L, cap, lam)
                 or _penalized_swap(nodes, R, L, cap, lam))
        if not moved:
            lam = min(lam * 2.0, lam_max)
            if lam >= lam_max:
                break

    _drop_empty(R, L)
    return R if _total_overload(L, cap) <= EPS else None


def route_reduction(
    parent_i,
    route_i,
    fit_i,
    nodes,
    capacity,
    max_targets=3,
    max_iters=600,
    lam0=10.0,
    lam_max=1e7,
):
    """
    Operator giảm route (chữ ký giống các operator LS khác).
    Trả về nghiệm có K nhỏ hơn nếu tìm được nghiệm FEASIBLE fitness tốt hơn,
    ngược lại trả về nguyên trạng.
    """
    routes = [list(r) for r in _split_routes(parent_i, route_i)]
    K = len(routes)
    if K <= 1:
        return parent_i, route_i, fit_i

    loads = [route_demand(r, nodes) for r in routes]
    total_demand = sum(loads)

    # Cận dưới bin-packing: nếu (K-1) thùng không chứa nổi tổng demand thì bỏ.
    if (K - 1) * capacity < total_demand - EPS:
        return parent_i, route_i, fit_i

    best_parent, best_route, best_fit = parent_i, route_i, fit_i

    # Ưu tiên bỏ route demand nhỏ nhất (dễ hấp thụ nhất).
    for t in sorted(range(K), key=lambda i: loads[i])[:max_targets]:
        R = _try_remove_route(nodes, routes, t, capacity, max_iters, lam0, lam_max)
        if R is None:
            continue
        new_parent, new_route = rebuild_solution(R)
        new_fit = get_fitness(new_parent, new_route, nodes)
        if new_fit < best_fit - EPS:
            best_parent, best_route, best_fit = new_parent, new_route, new_fit

    return best_parent, best_route, best_fit
