"""
Train POMO cho neural completion module (bản v3).

Tích hợp những gì đã thống nhất:
- POMO: mỗi instance chạy S start khác nhau; baseline = trung bình reward theo
  nhóm start của cùng instance (không cần baseline network).
- Model đã có distance-bias kiểu ICAM (cross-scale) sẵn trong cvrp_model.
- Vehicle penalty (giảm số route K): bật trong CVRPenv, RAMP dần theo thời gian
  để đầu train lo học đi-đường, sau mới ép giảm route. Penalty ở THANG [0,1].
- Curriculum theo thời gian: mở rộng dải size dần nhưng LUÔN giữ size nhỏ.
- Adaptive batch/starts theo N để vừa VRAM 12GB (núm chính: TRAJ_NODE_BUDGET).
- Data rộng (cvrp_data): 3 depot × 3 customer × 7 demand × route-size↔capacity.
- Validation greedy + ×8 augmentation best; checkpoint best/latest + resume.

Điều khiển bằng NGÂN SÁCH THỜI GIAN (không phải epoch) vì chi phí mỗi batch
thay đổi mạnh theo N.

Núm cần để ý trên RTX 3060:
    mem_budget          -> ~ B*S*N^2; nếu OOM thì giảm (vd 3M -> 2M -> 1.5M).
    vehicle_penalty_max -> tune để cân K vs distance (thang [0,1]).
    val_penalty         -> trọng số K khi CHỌN checkpoint best (xem ghi chú config).
"""

import os
import time
import math
import random

import torch
import torch.optim as optim

from cvrp_model import CVRPModel
from cvrp_env import CVRPenv, augment_xy_data_by_8_fold
from cvrp_data import make_batch


# =========================================================
# CONFIG
# =========================================================

def make_config():
    return dict(
        seed=2026,

        # model
        embedding_dim=128,
        num_heads=8,
        num_layers=3,

        # optimization
        lr=1e-4,
        eta_min=1e-6,
        warmup_steps=2000,
        grad_clip=1.0,

        # ngân sách thời gian (2 tuần; để buffer xuống 13 ngày)
        time_budget_hours=13 * 24.0,

        # POMO / bộ nhớ
        # Peak-mem ≈ B*S*N^2 (encoder attn O(M^2) + decode graph O(M*T), T~N),
        # nên ta giữ B*S*N^2 ~ mem_budget (KHÔNG phải B*S*N).
        max_starts=48,
        mem_budget=3_000_000,      # ~ B*S*N^2. GIẢM nếu OOM, TĂNG nếu còn VRAM.
        min_traj=16,               # số trajectory tối thiểu / micro-batch
        min_batch=1,
        target_instances=8,        # số instance hiệu dụng / update (qua grad-accum)

        # vehicle penalty (giảm route) — thang [0,1]
        vehicle_penalty_max=0.10,
        vehicle_penalty_ramp_frac=0.30,   # ramp tuyến tính trong 30% đầu

        # validation / logging / checkpoint
        validate_every_min=30.0,
        log_every=50,
        save_dir="checkpoints_neural_fill_v3",
        resume=True,

        # Chọn checkpoint best theo score = distance + val_penalty * K (thang [0,1]),
        # KHÔNG chỉ distance. val_penalty nên >= vehicle_penalty_max.
        # Lưu ý quy đổi: objective thật F = C*K + dist với C=1000 trên tọa độ [0,1000];
        # chuẩn hóa về [0,1] (chia ~L=1000) thì C/L ~ 1, nên nếu muốn phản ánh đúng
        # mục tiêu thật có thể đặt val_penalty ~ 1.0 (nặng K hơn).
        val_penalty=0.10,
        val_chunk=32,             # số quỹ đạo / chunk khi chạy aug8 (chặn bộ nhớ)

        # validation set (cố định, seed riêng)
        val_sizes=[100, 200, 300],
        val_batch=16,
        val_configs=[
            dict(depot_type="random",   customer_type="random",         demand_type="small",    route_size=8),
            dict(depot_type="eccentric",customer_type="cluster",        demand_type="u1_100",   route_size=5),
            dict(depot_type="central",  customer_type="random_cluster", demand_type="bimodal",  route_size=15),
        ],
    )


# =========================================================
# CURRICULUM
# =========================================================

def current_size_pool(elapsed_frac):
    if elapsed_frac < 0.25:
        return [50, 100]
    if elapsed_frac < 0.55:
        return [50, 100, 150, 200]
    if elapsed_frac < 0.85:
        return [50, 100, 150, 200, 256]
    return [50, 100, 150, 200, 256, 320, 400]


def pomo_config(N, cfg):
    """
    Chọn số instance B và số start S cho size N sao cho B*S*N^2 ~ mem_budget
    (giữ peak-memory bị chặn cho MỌI N, kể cả N lớn).
    Ở N lớn, S (số start) tự giảm thay vì cố ép 48 start.
    """
    traj = max(cfg["min_traj"], cfg["mem_budget"] // (N * N))   # ~ B*S
    S = min(N, cfg["max_starts"], traj)
    S = max(2, S)                                               # cần ≥2 start để có baseline
    B = max(cfg["min_batch"], traj // S)
    return int(B), int(S)


def sample_start_nodes(B, N, S, device):
    """Mỗi instance chọn S customer xuất phát phân biệt (index 1..N)."""
    rand = torch.rand(B, N, device=device)
    idx = rand.argsort(dim=1)[:, :S] + 1   # [B, S] in 1..N
    return idx.reshape(-1)                 # [B*S], khớp repeat_interleave(S)


# =========================================================
# LR + PENALTY SCHEDULE
# =========================================================

def lr_at(step, elapsed_frac, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    # cosine theo phần trăm thời gian đã dùng
    frac = min(max(elapsed_frac, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return cfg["eta_min"] + (cfg["lr"] - cfg["eta_min"]) * cos


def penalty_at(elapsed_frac, cfg):
    ramp = cfg["vehicle_penalty_ramp_frac"]
    factor = min(1.0, elapsed_frac / ramp) if ramp > 0 else 1.0
    return cfg["vehicle_penalty_max"] * factor


# =========================================================
# VALIDATION
# =========================================================

def build_validation_sets(cfg, device):
    val_sets = []

    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(12345)
    torch.manual_seed(12345)

    for N in cfg["val_sizes"]:
        for vc in cfg["val_configs"]:
            locs, demands, capacity, meta = make_batch(
                cfg["val_batch"], N, device,
                depot_type=vc["depot_type"],
                customer_type=vc["customer_type"],
                demand_type=vc["demand_type"],
                route_size=vc["route_size"],
            )
            val_sets.append({"N": N, "locs": locs.detach(),
                             "demands": demands.detach(), "meta": meta})

    random.setstate(py_state)
    torch.random.set_rng_state(torch_state)
    return val_sets


@torch.no_grad()
def validate(model, val_sets, device, val_penalty, chunk=32):
    """
    Đo theo ĐÚNG dạng mục tiêu F = distance + val_penalty * K (thang chuẩn hóa),
    không chỉ distance — để checkpoint "best" không ưu tiên nghiệm ít quãng đường
    nhưng nhiều route.

    Trả dict với distance / số route / score cho cả greedy và aug8.
    aug8 lấy best-of-8 theo SCORE (không phải theo distance).
    aug8 chạy theo chunk để bộ nhớ bị chặn dù val_batch/val_sizes có tăng.
    """
    model.eval()
    agg = {k: 0.0 for k in ("g_dist", "g_routes", "g_score",
                            "a_dist", "a_routes", "a_score")}
    cnt = 0

    for vs in val_sets:
        N = vs["N"]
        locs = vs["locs"]
        demands = vs["demands"]
        B = locs.size(0)

        # ---- greedy ----
        env = CVRPenv(num_nodes=N, device=device)
        env.reset(B, locs, demands)
        model(env, decode_type="greedy")
        g_dist = env.total_distance                  # [B]
        g_routes = env.route_count.float()           # [B]
        g_score = g_dist + val_penalty * g_routes
        agg["g_dist"] += g_dist.mean().item()
        agg["g_routes"] += g_routes.mean().item()
        agg["g_score"] += g_score.mean().item()

        # ---- ×8 augmentation, chạy theo chunk, best-of-8 theo SCORE ----
        aug_locs = augment_xy_data_by_8_fold(locs)   # [8B, N+1, 2]
        aug_dem = demands.repeat(8, 1)               # [8B, N]
        dists, routes = [], []
        total = 8 * B
        for s in range(0, total, chunk):
            e = min(s + chunk, total)
            env2 = CVRPenv(num_nodes=N, device=device)
            env2.reset(e - s, aug_locs[s:e], aug_dem[s:e])
            model(env2, decode_type="greedy")
            dists.append(env2.total_distance)
            routes.append(env2.route_count.float())

        a_dist = torch.cat(dists).view(8, B)
        a_routes = torch.cat(routes).view(8, B)
        a_score_all = a_dist + val_penalty * a_routes        # [8, B]

        best = a_score_all.argmin(dim=0)                     # [B], best aug theo score
        cols = torch.arange(B, device=a_score_all.device)
        agg["a_dist"] += a_dist[best, cols].mean().item()
        agg["a_routes"] += a_routes[best, cols].mean().item()
        agg["a_score"] += a_score_all[best, cols].mean().item()

        cnt += 1

    model.train()
    for k in agg:
        agg[k] /= cnt
    return agg


# =========================================================
# CHECKPOINT
# =========================================================

def save_ckpt(path, model, optimizer, scaler, step, trained_sec, best_score, cfg):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "trained_sec": trained_sec,
        "best_score": best_score,
        "cfg": cfg,
    }, path)


# =========================================================
# TRAIN
# =========================================================

def train(cfg, device):
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    use_amp = torch.cuda.is_available()
    amp_device = "cuda" if use_amp else "cpu"

    os.makedirs(cfg["save_dir"], exist_ok=True)
    latest_path = os.path.join(cfg["save_dir"], "model_latest.pt")
    best_path = os.path.join(cfg["save_dir"], "model_best_sampling.pt")

    model = CVRPModel(
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    trained_sec = 0.0
    best_score = float("inf")

    if cfg["resume"] and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        step = ck.get("step", 0)
        trained_sec = ck.get("trained_sec", 0.0)
        best_score = ck.get("best_score", ck.get("best_aug", float("inf")))
        print(f"[RESUME] step={step} | trained={trained_sec/3600:.2f}h | best_score={best_score:.4f}")

    val_sets = build_validation_sets(cfg, device)
    budget_sec = cfg["time_budget_hours"] * 3600.0

    print("=" * 78)
    print(f"[TRAIN] device={device} | amp={use_amp} | budget={cfg['time_budget_hours']:.1f}h")
    print(f"[TRAIN] mem_budget={cfg['mem_budget']} (~B*S*N^2) | max_starts={cfg['max_starts']}")
    print("=" * 78)

    last_val = time.perf_counter()
    run_start = time.perf_counter()
    running_cost = 0.0
    running_n = 0

    while True:
        now = time.perf_counter()
        trained_sec_live = trained_sec + (now - run_start)
        if trained_sec_live >= budget_sec:
            break
        elapsed_frac = trained_sec_live / budget_sec

        # --- chọn size theo curriculum ---
        N = random.choice(current_size_pool(elapsed_frac))
        B, S = pomo_config(N, cfg)

        # --- LR + penalty theo lịch ---
        cur_lr = lr_at(step, elapsed_frac, cfg)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        pen = penalty_at(elapsed_frac, cfg)

        # Grad-accum: ở N lớn B tụt về 1, ta gộp nhiều micro-batch để số
        # instance hiệu dụng / update >= target_instances (giảm nhiễu gradient).
        grad_accum = max(1, math.ceil(cfg["target_instances"] / B))

        optimizer.zero_grad(set_to_none=True)
        step_cost = 0.0
        step_loss = 0.0

        for _ in range(grad_accum):
            # --- data + replicate cho POMO ---
            locs, demands, capacity, meta = make_batch(B, N, device)
            locs_rep = locs.repeat_interleave(S, dim=0)
            dem_rep = demands.repeat_interleave(S, dim=0)
            starts = sample_start_nodes(B, N, S, device)

            env = CVRPenv(
                num_nodes=N, device=device,
                use_vehicle_penalty=(pen > 0.0), vehicle_penalty=pen,
            )
            env.reset(B * S, locs_rep, dem_rep)

            with torch.amp.autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
                rewards, log_probs = model(env, decode_type="sampling", start_nodes=starts)
                rewards = rewards.view(B, S)
                log_probs = log_probs.view(B, S)

                baseline = rewards.mean(dim=1, keepdim=True)   # POMO shared baseline
                advantage = rewards - baseline
                loss = -(advantage.detach() * log_probs).mean() / grad_accum

            scaler.scale(loss).backward()
            step_cost += (-rewards).mean().item()
            step_loss += loss.item() * grad_accum   # gỡ scale 1/grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        step += 1
        running_cost += step_cost / grad_accum
        running_n += 1

        if step % cfg["log_every"] == 0:
            print(
                f"step {step:6d} | {trained_sec_live/3600:6.2f}h "
                f"({100*elapsed_frac:4.1f}%) | N={N:3d} B={B:3d} S={S:2d} ga={grad_accum} | "
                f"lr={cur_lr:.2e} pen={pen:.3f} | "
                f"cost={running_cost/max(running_n,1):.3f} loss={step_loss/grad_accum:.4f} | "
                f"cfg={meta['depot'][:3]}/{meta['customer'][:4]}/{meta['demand']} r={meta['route_size']}"
            )
            running_cost = 0.0
            running_n = 0

        # --- validation định kỳ theo phút ---
        if (time.perf_counter() - last_val) / 60.0 >= cfg["validate_every_min"]:
            v = validate(model, val_sets, device,
                         val_penalty=cfg["val_penalty"], chunk=cfg["val_chunk"])
            print(
                f"[VAL] step={step} | "
                f"greedy: dist={v['g_dist']:.3f} K={v['g_routes']:.2f} score={v['g_score']:.3f} | "
                f"aug8: dist={v['a_dist']:.3f} K={v['a_routes']:.2f} score={v['a_score']:.3f} | "
                f"best_score={best_score:.4f}"
            )

            cur_trained = trained_sec + (time.perf_counter() - run_start)

            # Cập nhật best TRƯỚC, rồi mới lưu latest -> latest mang best_score mới.
            if v["a_score"] < best_score:
                best_score = v["a_score"]
                save_ckpt(best_path, model, optimizer, scaler, step, cur_trained, best_score, cfg)
                print(f"[VAL] new best score={best_score:.4f} -> saved {best_path}")

            save_ckpt(latest_path, model, optimizer, scaler, step, cur_trained, best_score, cfg)
            last_val = time.perf_counter()

    # lưu lần cuối
    trained_sec += (time.perf_counter() - run_start)
    save_ckpt(latest_path, model, optimizer, scaler, step, trained_sec, best_score, cfg)
    print("=" * 78)
    print(f"[DONE] step={step} | trained={trained_sec/3600:.2f}h | best_score={best_score:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train(make_config(), dev)
