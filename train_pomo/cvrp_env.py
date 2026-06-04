"""
CVRP environment (bản nâng cấp).

Thay đổi so với bản cũ:
1. reset() tính sẵn ma trận khoảng cách cặp-cặp `self.dist` [B, M, M]
   để cấp cho distance-bias kiểu ICAM trong model.
2. Thêm hook phạt số route (use_vehicle_penalty / vehicle_penalty) —
   phục vụ mục tiêu F = C*K + distance. Mặc định TẮT để giữ tương thích.
   Lưu ý: vehicle_penalty phải tune theo THANG TỌA ĐỘ CHUẨN HÓA [0,1],
   không phải theo C=1000 của objective thật.
3. Hỗ trợ POMO multi-start: training loop chỉ cần ép action ở bước đầu
   (xem CVRPModel._run_decode), env không cần đổi gì.
4. Tiện ích augment_xy_data_by_8_fold cho instance augmentation lúc infer.

Interface công khai (reset / get_state / get_mask / step) giữ nguyên để
GA.py chạy được; state nay có thêm khóa "dist".
"""

import torch


def augment_xy_data_by_8_fold(locs):
    """
    POMO-style ×8 dihedral augmentation cho tọa độ trong [0,1].

    locs: [B, M, 2]  (M = num_nodes + 1, gồm depot ở index 0)
    return: [8B, M, 2]  — 8 phép biến đổi đối xứng của hình vuông đơn vị.

    Dùng lúc inference: chạy model trên 8 bản, lấy nghiệm tốt nhất.
    Vì khoảng cách Euclid bất biến qua các phép này nên nghiệm hợp lệ
    của bản augment cũng hợp lệ cho bản gốc.
    """
    x = locs[:, :, [0]]
    y = locs[:, :, [1]]

    augmented = torch.cat(
        [
            torch.cat([x, y], dim=2),
            torch.cat([y, x], dim=2),
            torch.cat([x, 1 - y], dim=2),
            torch.cat([y, 1 - x], dim=2),
            torch.cat([1 - x, y], dim=2),
            torch.cat([1 - y, x], dim=2),
            torch.cat([1 - x, 1 - y], dim=2),
            torch.cat([1 - y, 1 - x], dim=2),
        ],
        dim=0,
    )
    return augmented


class CVRPenv:
    def __init__(
        self,
        num_nodes=20,
        capacity=None,
        device=None,
        use_vehicle_penalty=False,
        vehicle_penalty=0.0,
    ):
        self.num_nodes = num_nodes
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        default_capacity = {20: 30.0, 50: 40.0, 100: 50.0}
        self.capacity = float(
            capacity if capacity is not None
            else default_capacity.get(num_nodes, 50.0)
        )

        # Hook phạt mỗi route. Phạt được trừ vào reward mỗi lần MỞ một route
        # mới (đi từ depot ra một customer). Tổng số lần đó = K (số route).
        self.use_vehicle_penalty = use_vehicle_penalty
        self.vehicle_penalty = float(vehicle_penalty)

    def reset(self, batch_size, locs=None, demands=None):
        self.batch_size = batch_size

        if locs is None or demands is None:
            depot_locs = torch.rand(batch_size, 1, 2, device=self.device)
            customers_locs = torch.rand(batch_size, self.num_nodes, 2, device=self.device)
            self.locs = torch.cat((depot_locs, customers_locs), dim=1)

            raw_demands = torch.randint(1, 10, (batch_size, self.num_nodes), device=self.device)
            self.demands = raw_demands.float() / self.capacity
        else:
            self.locs = locs.clone()
            self.demands = demands.clone()

        # Ma trận khoảng cách cặp-cặp [B, M, M], M = num_nodes + 1.
        # Tọa độ đã ở thang [0,1] nên distance ∈ [0, ~1.414]. Dùng cho
        # distance-bias kiểu ICAM (node xa bị giảm attention).
        self.dist = torch.cdist(self.locs, self.locs, p=2)

        self.current_node = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.remaining_capacity = torch.ones(batch_size, device=self.device)
        self.remaining_demands = self.demands.clone()

        self.route_count = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.total_distance = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        return self.get_state()

    def get_state(self):
        depot_demand = torch.zeros(self.batch_size, 1, device=self.device)
        node_demands = torch.cat([depot_demand, self.remaining_demands], dim=1)

        return {
            "locs": self.locs,
            "demands": node_demands,
            "current_node": self.current_node,
            "remaining_capacity": self.remaining_capacity,
            "dist": self.dist,
        }

    def get_mask(self):
        mask = torch.zeros(
            self.batch_size, self.num_nodes + 1, dtype=torch.bool, device=self.device
        )
        done_customers = (self.remaining_demands <= 1e-9).all(dim=1)

        # Đang ở depot và chưa xong => không cho đứng yên ở depot.
        mask[:, 0] = (self.current_node == 0) & (~done_customers)

        served = self.remaining_demands <= 1e-9
        over_capacity = self.remaining_demands > self.remaining_capacity[:, None]
        mask[:, 1:] = served | over_capacity

        # Hàng đã xong: chỉ cho ở depot.
        mask[done_customers, 0] = False
        mask[done_customers, 1:] = True

        return mask

    def step(self, next_node):
        next_node = next_node.long()
        B = self.batch_size
        batch_idx = torch.arange(B, device=self.device)

        prev_node = self.current_node.clone()

        prev_loc = self.locs[batch_idx, prev_node]
        next_loc = self.locs[batch_idx, next_node]
        step_dist = torch.norm(prev_loc - next_loc, dim=1)

        self.total_distance += step_dist

        is_customer = next_node > 0
        customer_idx = next_node - 1

        reward = -step_dist

        # Mở route mới = đi từ depot ra customer. Luôn ĐẾM (stat hữu ích),
        # chỉ phần PHẠT reward mới phụ thuộc cờ use_vehicle_penalty.
        opens_route = (prev_node == 0) & is_customer
        self.route_count += opens_route.long()

        if self.use_vehicle_penalty and self.vehicle_penalty > 0.0:
            reward = reward - opens_route.float() * self.vehicle_penalty

        chosen_demand = torch.zeros(B, device=self.device)
        valid_rows = torch.where(is_customer)[0]
        if len(valid_rows) > 0:
            chosen_demand[valid_rows] = self.remaining_demands[
                valid_rows, customer_idx[valid_rows]
            ]
            self.remaining_capacity[valid_rows] -= chosen_demand[valid_rows]
            self.remaining_demands[valid_rows, customer_idx[valid_rows]] = 0.0

        depot_rows = torch.where(~is_customer)[0]
        if len(depot_rows) > 0:
            self.remaining_capacity[depot_rows] = 1.0

        self.current_node = next_node

        all_served = (self.remaining_demands <= 1e-9).all(dim=1)
        done = all_served & (self.current_node == 0)

        return self.get_state(), reward, done
