"""
Bộ sinh dữ liệu CVRP "rộng", bám sơ đồ sinh của Set-X (Uchoa 2017) làm SÀN
độ đa dạng (không phải để overfit Set-X). Mỗi instance được đặc trưng bởi 5 trục:
    - số khách N
    - vị trí depot: central / eccentric (góc) / random
    - vị trí khách: random / cluster / random-cluster (số seed biến thiên)
    - phân bố demand: 7 kiểu (unitary, nhỏ, U[5,10], U[1,100], U[50,100],
      phụ-thuộc-góc-phần-tư, bimodal nhiều-nhỏ-vài-lớn)
    - route-size r => capacity = round(r * mean_demand)  (couple để trải đủ
      các CHẾ ĐỘ MẬT ĐỘ ROUTE — quan trọng cho mục tiêu giảm K)

Tọa độ ở thang [0,1] (model scale-invariant nhờ chuẩn hóa). Demand trả về đã
chia cho capacity (chuẩn hóa, ≤ 1) — đúng interface mà CVRPenv mong đợi.

Một batch dùng MỘT cấu hình (giống code cũ); đa dạng đến từ việc bốc cấu hình
khác nhau qua nhiều batch.
"""

import random
import torch


DEPOT_TYPES = ["central", "eccentric", "random"]
CUSTOMER_TYPES = ["random", "cluster", "random_cluster"]
DEMAND_TYPES = ["unitary", "small", "u5_10", "u1_100", "u50_100", "quadrant", "bimodal"]

DEFAULT_ROUTE_SIZE_RANGE = (3.0, 25.0)   # ~5 khoảng route-size của Set-X
CLUSTER_SEED_RANGE = (2, 8)              # số seed cụm biến thiên
CLUSTER_SPREAD = 0.05                    # độ tản quanh seed


def _sample_depot(B, depot_type, device):
    if depot_type == "central":
        return torch.full((B, 1, 2), 0.5, device=device)
    if depot_type == "eccentric":
        return torch.zeros((B, 1, 2), device=device)   # góc dưới-trái
    return torch.rand(B, 1, 2, device=device)           # random


def _sample_customers(B, N, customer_type, device):
    if customer_type == "random":
        return torch.rand(B, N, 2, device=device)

    num_seeds = random.randint(*CLUSTER_SEED_RANGE)
    seeds = torch.rand(B, num_seeds, 2, device=device)
    assign = torch.randint(0, num_seeds, (B, N), device=device)
    bidx = torch.arange(B, device=device).unsqueeze(1)
    clustered = seeds[bidx, assign] + CLUSTER_SPREAD * torch.randn(B, N, 2, device=device)
    clustered = clustered.clamp(0.0, 1.0)

    if customer_type == "cluster":
        return clustered

    # random_cluster: nửa rải đều, nửa theo cụm
    n_rand = N // 2
    rand_part = torch.rand(B, n_rand, 2, device=device)
    return torch.cat([rand_part, clustered[:, n_rand:, :]], dim=1).clamp(0.0, 1.0)


def _sample_raw_demands(B, N, demand_type, customer_locs, device):
    if demand_type == "unitary":
        return torch.ones(B, N, device=device)
    if demand_type == "small":
        return torch.randint(1, 11, (B, N), device=device).float()
    if demand_type == "u5_10":
        return torch.randint(5, 11, (B, N), device=device).float()
    if demand_type == "u1_100":
        return torch.randint(1, 101, (B, N), device=device).float()
    if demand_type == "u50_100":
        return torch.randint(50, 101, (B, N), device=device).float()
    if demand_type == "quadrant":
        x = customer_locs[:, :, 0]
        y = customer_locs[:, :, 1]
        big = ((x > 0.5) & (y > 0.5)) | ((x <= 0.5) & (y <= 0.5))
        small_q = torch.randint(1, 51, (B, N), device=device).float()
        big_q = torch.randint(51, 101, (B, N), device=device).float()
        return torch.where(big, big_q, small_q)
    if demand_type == "bimodal":
        small_q = torch.randint(1, 11, (B, N), device=device).float()
        big_q = torch.randint(50, 101, (B, N), device=device).float()
        is_big = torch.rand(B, N, device=device) < 0.2
        return torch.where(is_big, big_q, small_q)
    raise ValueError(f"Unknown demand_type: {demand_type}")


def make_batch(
    B,
    N,
    device,
    depot_type=None,
    customer_type=None,
    demand_type=None,
    route_size=None,
    route_size_range=DEFAULT_ROUTE_SIZE_RANGE,
):
    """
    Trả về:
        locs:     [B, N+1, 2]  tọa độ [0,1], index 0 là depot
        demands:  [B, N]       đã chuẩn hóa (raw/capacity_i, ≤ 1)
        capacity: [B]          capacity mỗi instance (để log/tham khảo)
        meta:     dict cấu hình của batch

    Các tham số *_type / route_size: None = bốc ngẫu nhiên (train);
    truyền cụ thể = cố định (dùng cho validation).
    """
    depot_type = depot_type or random.choice(DEPOT_TYPES)
    customer_type = customer_type or random.choice(CUSTOMER_TYPES)
    demand_type = demand_type or random.choice(DEMAND_TYPES)
    r = route_size if route_size is not None else random.uniform(*route_size_range)

    depot = _sample_depot(B, depot_type, device)
    customers = _sample_customers(B, N, customer_type, device)
    locs = torch.cat([depot, customers], dim=1)

    raw = _sample_raw_demands(B, N, demand_type, customers, device)  # [B, N]

    mean_d = raw.mean(dim=1)
    max_d = raw.max(dim=1).values
    # capacity = round(r * mean_demand), nhưng không nhỏ hơn demand lớn nhất
    # (đảm bảo mọi khách vừa một xe rỗng => demand chuẩn hóa ≤ 1).
    capacity = torch.maximum(torch.round(r * mean_d), max_d).clamp(min=1.0)

    demands = raw / capacity.unsqueeze(1)

    meta = {
        "depot": depot_type,
        "customer": customer_type,
        "demand": demand_type,
        "route_size": round(float(r), 2),
    }
    return locs, demands, capacity, meta
