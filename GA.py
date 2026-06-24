import os
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from read_data import read_data
from caculate import get_good_routes, get_fitness
from train_pomo.cvrp_model import CVRPModel
from train_pomo.cvrp_env import CVRPenv


# =========================================================
# INSTANCE CONTEXT
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# These are set by main_check_competition.py when GA(...) is called.
# Do not hard-code an instance here; otherwise GA can silently use the wrong
# dimension/capacity/nodes when main.py switches to another CVRP instance.
dimension = None
capacity = None
nodes = None


def _set_instance_context(dimension_value=None, capacity_value=None, nodes_data=None):
    """Synchronize GA module globals with the instance loaded in main."""
    global dimension, capacity, nodes

    if dimension_value is not None:
        dimension = int(dimension_value)
    if capacity_value is not None:
        capacity = capacity_value
    if nodes_data is not None:
        nodes = nodes_data

    if dimension is None or capacity is None or nodes is None:
        raise ValueError(
            "GA instance context is missing. Pass dimension_value, "
            "capacity_value and nodes_data from main when calling GA(...)."
        )

# =========================================================
# NEURAL MODEL CACHE
# =========================================================

_NEURAL_MODEL = None
_NEURAL_CKPT_LOADED = None
_NEURAL_LOAD_FAILED = False
_NEURAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_ckpt_path(ckpt_path: Optional[str]) -> str:
    """
    Resolve checkpoint path.
    - absolute path: giữ nguyên
    - relative path: nối từ thư mục chứa GA.py
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            BASE_DIR,
            "checkpoints_neural_fill_v3",
            "model_best_sampling.pt",
        )

    if os.path.isabs(ckpt_path):
        return ckpt_path

    return os.path.join(BASE_DIR, ckpt_path)


def _load_neural_model(
    ckpt_path: Optional[str] = None,
    embedding_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
):
    """
    Load neural model một lần, cache lại để GA không phải load checkpoint nhiều lần.
    """
    global _NEURAL_MODEL, _NEURAL_CKPT_LOADED, _NEURAL_LOAD_FAILED

    resolved_ckpt_path = _resolve_ckpt_path(ckpt_path)

    if _NEURAL_MODEL is not None and _NEURAL_CKPT_LOADED == resolved_ckpt_path:
        return _NEURAL_MODEL

    if _NEURAL_LOAD_FAILED:
        raise FileNotFoundError("Neural model was already attempted and failed to load.")

    if not os.path.exists(resolved_ckpt_path):
        _NEURAL_LOAD_FAILED = True
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt_path}")

    # weights_only=False vì checkpoint chứa cfg/optimizer (không chỉ tensor).
    ckpt = torch.load(resolved_ckpt_path, map_location=_NEURAL_DEVICE, weights_only=False)

    # Ưu tiên lấy kiến trúc TỪ checkpoint để khỏi lệch shape khi load
    # (model mới: POMO + ICAM distance-bias). Fallback về tham số mặc định.
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    emb = int(cfg.get("embedding_dim", embedding_dim))
    heads = int(cfg.get("num_heads", num_heads))
    layers = int(cfg.get("num_layers", num_layers))

    model = CVRPModel(
        embedding_dim=emb,
        num_heads=heads,
        num_layers=layers,
    ).to(_NEURAL_DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    _NEURAL_MODEL = model
    _NEURAL_CKPT_LOADED = resolved_ckpt_path

    print(f"[INFO] Loaded neural model from: {resolved_ckpt_path}")
    return _NEURAL_MODEL


def _make_env(num_nodes: int, capacity_value: float, device: torch.device):
    """
    Tạo CVRPenv. Có fallback để tránh lỗi nếu cvrp_env.py của bạn
    không còn tham số vehicle_penalty/use_vehicle_penalty.
    """
    try:
        return CVRPenv(
            num_nodes=num_nodes,
            capacity=capacity_value,
            device=device,
            use_vehicle_penalty=False,
            vehicle_penalty=0.0,
        )
    except TypeError:
        return CVRPenv(
            num_nodes=num_nodes,
            capacity=capacity_value,
            device=device,
        )


# =========================================================
# BASIC ROUTE UTILS
# =========================================================

def get_vertices_from_routes(routes: Sequence[Sequence[int]]) -> Set[int]:
    vertices = set()

    for route_seg in routes:
        vertices.update(route_seg)

    return vertices


def _euc_2d_distance(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    return int(np.sqrt(dx * dx + dy * dy) + 0.5)


def _euclidean_distance(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    # Giữ tên cũ để không phải sửa các chỗ gọi bên dưới
    return _euc_2d_distance(a, b)


def _route_distance(route_seg: Sequence[int]) -> float:
    """
    Tính distance của một route: depot -> customers -> depot.
    Depot là node 1.
    """
    if len(route_seg) == 0:
        return float("inf")

    total = 0.0
    prev = 1

    for v in route_seg:
        total += _euclidean_distance(nodes[prev], nodes[v])
        prev = v

    total += _euclidean_distance(nodes[prev], nodes[1])
    return total


def _route_demand(route_seg: Sequence[int]) -> float:
    return float(sum(nodes[v]["demand"] for v in route_seg))


def _fallback_route_score(route_seg: Sequence[int]) -> float:
    """
    Score càng nhỏ càng tốt.

    Dùng distance/customer để tránh việc route ngắn chỉ có 1 customer
    luôn được ưu tiên quá mạnh. Thêm penalty nhẹ nếu route dùng tải kém.
    """
    if len(route_seg) == 0:
        return float("inf")

    dist = _route_distance(route_seg)
    demand = _route_demand(route_seg)

    distance_per_customer = dist / max(len(route_seg), 1)

    load_ratio = demand / max(float(capacity), 1e-9)
    unused_capacity_penalty = max(0.0, 1.0 - load_ratio)

    over_capacity_penalty = 0.0
    if demand > capacity:
        over_capacity_penalty = 1000.0 * ((demand - capacity) / capacity)

    return distance_per_customer + 0.2 * unused_capacity_penalty + over_capacity_penalty


def _safe_score_from_get_good_routes(
    returned_scores: Any,
    route_rank: int,
    route_seg: Sequence[int],
) -> float:
    """
    Nếu get_good_routes trả về score thì dùng score đó.
    Nếu không đọc được score thì tự tính bằng _fallback_route_score().
    """
    if isinstance(returned_scores, (list, tuple)) and route_rank < len(returned_scores):
        candidate_score = returned_scores[route_rank]

        if isinstance(candidate_score, (int, float, np.integer, np.floating)):
            return float(candidate_score)

        if isinstance(candidate_score, (list, tuple)):
            for x in candidate_score:
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return float(x)

    return _fallback_route_score(route_seg)


def _flatten_kept_routes(kept_routes: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
    """
    kept_routes = [[...], [...], ...]
    -> child_permutation + route_markers

    marker = 1 nghĩa là bắt đầu route mới.
    marker = 0 nghĩa là tiếp tục route hiện tại.
    """
    child_permutation = []
    child_route_markers = []

    for route_seg in kept_routes:
        for j, vertex in enumerate(route_seg):
            child_permutation.append(vertex)
            child_route_markers.append(1 if j == 0 else 0)

    return child_permutation, child_route_markers


# =========================================================
# WEIGHTED-GREEDY ROUTE SELECTION
# =========================================================

def _collect_route_candidates(parent, route, parent_indices, num_good_routes: int):
    """
    Lấy good routes từ các parent và gán score cho từng route.

    Mỗi candidate có dạng:
        {
            "route": [...],
            "score": float,
            "parent_idx": int,
            "rank": int,
        }
    """
    candidates = []

    for p_idx in parent_indices:
        good_routes, returned_scores = get_good_routes(
            parent[p_idx],
            route[p_idx],
            nodes,
            num_good_routes=num_good_routes,
        )

        for rank, route_seg in enumerate(good_routes):
            route_seg = list(route_seg)

            if len(route_seg) == 0:
                continue

            route_vertices = set(route_seg)
            if len(route_vertices) != len(route_seg):
                continue

            score = _safe_score_from_get_good_routes(
                returned_scores=returned_scores,
                route_rank=rank,
                route_seg=route_seg,
            )

            candidates.append(
                {
                    "route": route_seg,
                    "score": float(score),
                    "parent_idx": p_idx,
                    "rank": rank,
                }
            )

    candidates.sort(key=lambda x: x["score"])
    return candidates


def _weighted_choice_by_score(candidates: Sequence[Dict[str, Any]], temperature: float):
    """
    Chọn một route theo xác suất.
    Score càng nhỏ thì xác suất càng cao.

    temperature nhỏ -> greedy hơn.
    temperature lớn -> random hơn.
    """
    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    scores = np.array([c["score"] for c in candidates], dtype=np.float64)

    score_min = scores.min()
    score_std = scores.std()

    normalized = (scores - score_min) / (score_std + 1e-9)

    temperature = max(float(temperature), 1e-6)
    weights = np.exp(-normalized / temperature)

    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weight_sum

    chosen_idx = np.random.choice(len(candidates), p=weights)
    return candidates[int(chosen_idx)]


def select_good_routes_weighted_greedy(
    route_candidates: Sequence[Dict[str, Any]],
    max_kept_nodes: int = 50,
    temperature: float = 0.7,
) -> List[List[int]]:
    """
    Chọn route giữ lại theo kiểu weighted-greedy.

    Ý tưởng:
    - Route score càng tốt thì xác suất được chọn càng cao.
    - Khi chọn một route, tất cả route còn lại có trùng customer với route đó bị loại.
    - Dừng khi tổng số node đã giữ đạt max_kept_nodes (không giới hạn theo số route).
    """
    available = list(route_candidates)
    kept_routes = []
    used_vertices = set()

    while available and len(used_vertices) < max_kept_nodes:
        feasible = []

        for cand in available:
            route_vertices = set(cand["route"])

            if route_vertices.isdisjoint(used_vertices):
                feasible.append(cand)

        if not feasible:
            break

        chosen = _weighted_choice_by_score(feasible, temperature=temperature)

        if chosen is None:
            break

        chosen_route = list(chosen["route"])
        chosen_vertices = set(chosen_route)

        kept_routes.append(chosen_route)
        used_vertices.update(chosen_vertices)

        # Loại bỏ route đã chọn và mọi route có trùng customer với used_vertices.
        available = [
            cand
            for cand in available
            if set(cand["route"]).isdisjoint(used_vertices)
        ]

    return kept_routes


# =========================================================
# FILL REMAINING CUSTOMERS
# =========================================================

def _normalize_subproblem_coords(depot_xy: np.ndarray, customer_xy: np.ndarray) -> np.ndarray:
    """
    Normalize tọa độ về [0,1] theo bounding box của subproblem.
    """
    all_xy = np.vstack([depot_xy[None, :], customer_xy]).astype(np.float32)

    min_xy = all_xy.min(axis=0, keepdims=True)
    max_xy = all_xy.max(axis=0, keepdims=True)
    scale = np.maximum(max_xy - min_xy, 1e-8)

    return (all_xy - min_xy) / scale


def _random_fill_remaining(
    remaining_vertices: Sequence[int],
    nodes_data,
    capacity_value: float,
) -> Tuple[List[int], List[int]]:
    """
    Fallback random/greedy fill có capacity check.
    """
    permutation = []
    markers = []

    current_load = 0.0

    for idx, vertex in enumerate(remaining_vertices):
        demand = float(nodes_data[vertex]["demand"])

        if idx == 0:
            permutation.append(vertex)
            markers.append(1)
            current_load = demand
        elif current_load + demand <= capacity_value:
            permutation.append(vertex)
            markers.append(0)
            current_load += demand
        else:
            permutation.append(vertex)
            markers.append(1)
            current_load = demand

    return permutation, markers


@torch.no_grad()
def solve_remaining_with_neural(
    remaining_vertices: Sequence[int],
    nodes_data,
    capacity_value: float,
    ckpt_path: Optional[str] = None,
    decode_type: str = "greedy",
) -> Tuple[List[int], List[int]]:
    """
    Dùng neural để giải residual CVRP trên remaining_vertices.

    Returns:
        neural_permutation
        neural_route_markers
    """
    if len(remaining_vertices) == 0:
        return [], []

    remaining_vertices = sorted(remaining_vertices)

    model = _load_neural_model(ckpt_path=ckpt_path)

    depot_xy = np.array(
        [nodes_data[1]["x"], nodes_data[1]["y"]],
        dtype=np.float32,
    )

    customer_xy = np.array(
        [[nodes_data[v]["x"], nodes_data[v]["y"]] for v in remaining_vertices],
        dtype=np.float32,
    )

    all_xy = _normalize_subproblem_coords(depot_xy, customer_xy)

    locs = torch.tensor(
        all_xy,
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    ).unsqueeze(0)

    demands = torch.tensor(
        [[nodes_data[v]["demand"] / capacity_value for v in remaining_vertices]],
        dtype=torch.float32,
        device=_NEURAL_DEVICE,
    )

    env = _make_env(
        num_nodes=len(remaining_vertices),
        capacity_value=capacity_value,
        device=_NEURAL_DEVICE,
    )

    env.reset(batch_size=1, locs=locs, demands=demands)

    # Model mới (POMO + ICAM): solve() tự chạy encoder (precompute) + decode,
    # tự eval() cho BatchNorm dùng running-stats, trả chuỗi action [B, T]
    # (0 = depot). start_nodes=None => decode bình thường từ depot.
    actions = model.solve(env, decode_type=decode_type)
    action_sequence = actions[0].tolist()

    neural_permutation = []
    neural_route_markers = []

    start_new_route = True

    for action in action_sequence:
        if action == 0:
            start_new_route = True
            continue

        global_vertex = remaining_vertices[action - 1]

        neural_permutation.append(global_vertex)
        neural_route_markers.append(1 if start_new_route else 0)

        start_new_route = False

    return neural_permutation, neural_route_markers


def _split_remaining_by_angle(
    remaining_vertices: Sequence[int],
    nodes_data,
    target_size: int = 100,
    max_size: int = 120,
    min_size: int = 30,
) -> List[List[int]]:
    """
    Chia remaining customers thành các residual subproblems theo góc quanh depot.

    Mục đích:
    - Không đưa quá nhiều node vào neural một lần.
    - Giữ mỗi block gần vùng train của neural: khoảng 50--150 node, tốt nhất 80--120.
    - Chia theo không gian thay vì random để route tạo ra ít bị đan chéo.
    """
    remaining_vertices = list(remaining_vertices)

    if len(remaining_vertices) == 0:
        return []

    target_size = max(1, int(target_size))
    max_size = max(target_size, int(max_size))
    min_size = max(1, int(min_size))

    if len(remaining_vertices) <= max_size:
        return [remaining_vertices]

    depot_x = float(nodes_data[1]["x"])
    depot_y = float(nodes_data[1]["y"])

    items = []
    for v in remaining_vertices:
        x = float(nodes_data[v]["x"])
        y = float(nodes_data[v]["y"])
        angle = float(np.arctan2(y - depot_y, x - depot_x))
        radius = float(np.sqrt((x - depot_x) ** 2 + (y - depot_y) ** 2))
        items.append((angle, radius, v))

    # Sweep theo góc quanh depot; radius chỉ để ổn định thứ tự trong cùng vùng góc.
    items.sort(key=lambda t: (t[0], t[1]))
    ordered_vertices = [v for _, _, v in items]

    blocks: List[List[int]] = []
    for i in range(0, len(ordered_vertices), target_size):
        blocks.append(ordered_vertices[i:i + target_size])

    # Nếu block cuối quá nhỏ, xử lý để tránh neural phải decode block quá bé.
    if len(blocks) >= 2 and len(blocks[-1]) < min_size:
        last = blocks.pop()
        merged = blocks[-1] + last

        if len(merged) <= max_size:
            blocks[-1] = merged
        else:
            # Không nhồi vượt max_size; chia đều lại 2 block cuối.
            mid = len(merged) // 2
            blocks[-1] = merged[:mid]
            blocks.append(merged[mid:])

    return blocks


@torch.no_grad()
def solve_remaining_with_neural_chunked(
    remaining_vertices: Sequence[int],
    nodes_data,
    capacity_value: float,
    ckpt_path: Optional[str] = None,
    decode_type: str = "greedy",
    max_neural_nodes: int = 120,
    target_neural_nodes: int = 100,
    min_neural_nodes: int = 30,
    verbose: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Neural fill mới cho GA_upd.

    Nếu |remaining_vertices| nhỏ: decode trực tiếp như code cũ.
    Nếu |remaining_vertices| lớn: chia thành nhiều block theo không gian rồi decode từng block.

    Cách này vẫn đúng logic paper:
    route inheritance -> unassigned set U -> NGM completion -> local search.
    Điểm khác chỉ là U lớn được hoàn thiện bằng nhiều residual subproblems.
    """
    remaining_vertices = list(remaining_vertices)

    if len(remaining_vertices) == 0:
        return [], []

    if len(remaining_vertices) <= max_neural_nodes:
        return solve_remaining_with_neural(
            remaining_vertices=remaining_vertices,
            nodes_data=nodes_data,
            capacity_value=capacity_value,
            ckpt_path=ckpt_path,
            decode_type=decode_type,
        )

    blocks = _split_remaining_by_angle(
        remaining_vertices=remaining_vertices,
        nodes_data=nodes_data,
        target_size=target_neural_nodes,
        max_size=max_neural_nodes,
        min_size=min_neural_nodes,
    )

    if verbose:
        sizes = [len(b) for b in blocks]
        print(f"[NGM CHUNK] remaining={len(remaining_vertices)} | blocks={sizes}")

    final_perm: List[int] = []
    final_markers: List[int] = []

    for block in blocks:
        block_perm, block_markers = solve_remaining_with_neural(
            remaining_vertices=block,
            nodes_data=nodes_data,
            capacity_value=capacity_value,
            ckpt_path=ckpt_path,
            decode_type=decode_type,
        )

        final_perm.extend(block_perm)
        final_markers.extend(block_markers)

    expected = set(remaining_vertices)
    got = set(final_perm)

    if len(final_perm) != len(remaining_vertices):
        raise ValueError(
            f"Chunked neural fill length mismatch: "
            f"got {len(final_perm)}, expected {len(remaining_vertices)}"
        )

    if len(got) != len(final_perm):
        raise ValueError("Chunked neural fill produced duplicated customers.")

    if got != expected:
        missing = sorted(expected - got)[:10]
        extra = sorted(got - expected)[:10]
        raise ValueError(
            f"Chunked neural fill customer mismatch. "
            f"missing={missing}, extra={extra}"
        )

    return final_perm, final_markers


# =========================================================
# MAIN GA CROSSOVER
# =========================================================

def _count_routes(route_markers):
    """
    Đếm số route từ route marker.
    marker = 1 nghĩa là bắt đầu route mới.
    """
    return sum(1 for x in route_markers if x == 1)


def _compute_kept_nodes_by_ratio(dimension, keep_ratio, min_keep=1, max_keep=None):
    """
    Tính số node sẽ giữ lại theo % tổng customer.

    Ví dụ:
    - dimension = 201 (200 customer + 1 depot)
    - keep_ratio = 0.30
    => giữ khoảng 60 node
    """
    total_customers = dimension - 1  # trừ depot

    kept = int(round(total_customers * keep_ratio))

    kept = max(min_keep, kept)

    if max_keep is not None:
        kept = min(kept, max_keep)

    return kept

def GA(
    parent,
    route,
    par1,
    par2,
    par3,
    use_neural_fill: bool = True,
    neural_ckpt_path: Optional[str] = os.path.join(
        BASE_DIR,
        "checkpoints_neural_fill_v3",
        "model_best_sampling.pt",
    ),
    neural_decode_type: str = "sampling",

    # lấy đủ candidate route từ mỗi parent
    num_good_routes_per_parent: int = 16,

    # giữ lại theo tỉ lệ node, không hard-code số route
    keep_node_ratio: float = 0.30,

    # giới hạn an toàn (tính theo node)
    min_kept_nodes: int = 10,
    max_kept_nodes_cap: Optional[int] = None,

    selection_trials: int = 10,
    route_select_temperature: float = 0.8,

    # neural chunking: tốt nhất cho model train trên 50/80/100/120/150 nodes
    max_neural_nodes: int = 120,
    target_neural_nodes: int = 100,
    min_neural_nodes: int = 30,
    verbose_neural_chunk: bool = False,

    # Instance context passed from main. This prevents GA from using a stale
    # hard-coded instance when running a different benchmark.
    dimension_value: Optional[int] = None,
    capacity_value: Optional[float] = None,
    nodes_data: Optional[dict] = None,
):
    """
    GA crossover:
    1. Lấy good routes từ 3 parent.
    2. Chọn route giữ lại bằng weighted-greedy, không chọn route trùng customer.
    3. Remaining customers được fill bằng neural hoặc random.
       Nếu remaining quá lớn, neural fill sẽ chia U thành nhiều subproblem nhỏ.
    4. Thử nhiều lần và trả child tốt nhất.
    """
    _set_instance_context(
        dimension_value=dimension_value,
        capacity_value=capacity_value,
        nodes_data=nodes_data,
    )

    resolved_ckpt_path = _resolve_ckpt_path(neural_ckpt_path)

    parent_indices = [par1, par2, par3]

    dynamic_max_kept_nodes = _compute_kept_nodes_by_ratio(
        dimension=dimension,
        keep_ratio=keep_node_ratio,
        min_keep=min_kept_nodes,
        max_keep=max_kept_nodes_cap,
    )

    route_candidates = _collect_route_candidates(
        parent=parent,
        route=route,
        parent_indices=parent_indices,
        num_good_routes=num_good_routes_per_parent,
    )

    best_fitness = float("inf")
    best_child = None
    best_child_route = None

    warned_neural_failure = False

    all_vertices = set(range(2, dimension + 1))

    n_full = max(1, round(selection_trials * 0.6))
    n_chunked = selection_trials - n_full
    trial_modes = ["full"] * n_full + ["chunked"] * n_chunked

    for trial_mode in trial_modes:
        kept_routes = select_good_routes_weighted_greedy(
            route_candidates=route_candidates,
            max_kept_nodes=dynamic_max_kept_nodes,
            temperature=route_select_temperature,
        )

        used_vertices = get_vertices_from_routes(kept_routes)
        remaining_vertices = sorted(all_vertices - used_vertices)

        child_permutation, child_route_markers = _flatten_kept_routes(kept_routes)

        if remaining_vertices:
            if use_neural_fill:
                try:
                    if trial_mode == "full":
                        fill_perm, fill_markers = solve_remaining_with_neural(
                            remaining_vertices=remaining_vertices,
                            nodes_data=nodes,
                            capacity_value=capacity,
                            ckpt_path=resolved_ckpt_path,
                            decode_type=neural_decode_type,
                        )
                    else:
                        fill_perm, fill_markers = solve_remaining_with_neural_chunked(
                            remaining_vertices=remaining_vertices,
                            nodes_data=nodes,
                            capacity_value=capacity,
                            ckpt_path=resolved_ckpt_path,
                            decode_type=neural_decode_type,
                            max_neural_nodes=max_neural_nodes,
                            target_neural_nodes=target_neural_nodes,
                            min_neural_nodes=min_neural_nodes,
                            verbose=verbose_neural_chunk,
                        )
                except Exception as e:
                    if not warned_neural_failure:
                        print(f"[WARN] Neural fill failed. Fallback to random fill. Error: {e}")
                        warned_neural_failure = True

                    fallback_vertices = list(remaining_vertices)
                    random.shuffle(fallback_vertices)

                    fill_perm, fill_markers = _random_fill_remaining(
                        fallback_vertices,
                        nodes,
                        capacity,
                    )
            else:
                fallback_vertices = list(remaining_vertices)
                random.shuffle(fallback_vertices)

                fill_perm, fill_markers = _random_fill_remaining(
                    fallback_vertices,
                    nodes,
                    capacity,
                )

            child_permutation.extend(fill_perm)
            child_route_markers.extend(fill_markers)

        expected_len = dimension - 1

        if len(child_permutation) != expected_len:
            raise ValueError(
                f"Child permutation length mismatch: "
                f"got {len(child_permutation)}, expected {expected_len}"
            )

        if len(child_route_markers) != expected_len:
            raise ValueError(
                f"Child route marker length mismatch: "
                f"got {len(child_route_markers)}, expected {expected_len}"
            )

        if len(set(child_permutation)) != expected_len:
            raise ValueError("Child has duplicated customers.")

        child_fitness = get_fitness(
            child_permutation,
            child_route_markers,
            nodes,
        )

        if child_fitness < best_fitness:
            best_fitness = child_fitness
            best_child = child_permutation
            best_child_route = child_route_markers

    return best_child, best_child_route, best_fitness