"""
CVRP model (bản nâng cấp).

Định hướng (đã thống nhất):
- Giữ kiến trúc attention encoder-decoder (họ AM/POMO) — train được trên RTX 3060.
- Thêm DISTANCE-BIAS kiểu ICAM: node ở xa bị giảm attention, mức độ phụ thuộc
  scale N (log2 số node). Đây là thứ giúp generalize cross-scale (train ~256,
  chạy được tới ~400) mà không cần đổi pipeline RL.
- POMO-ready: decoder precompute key/value một lần, hỗ trợ multi-start bằng cách
  ép action ở bước đầu (start_nodes). Tính multi-optima + augmentation để ở env.
- Tối ưu số route (giảm K) KHÔNG nằm ở kiến trúc mà ở reward (vehicle_penalty
  trong CVRPenv) + training. Model chỉ cần tối ưu reward mà env trả về.

Khác biệt kỹ thuật so với bản cũ:
- Thay nn.MultiheadAttention bằng attention thủ công để cộng được distance-bias
  hiệu quả (không phải materialize tensor bias [B,H,L,S] riêng).
- decoder.precompute(embeddings) + decoder.forward(...) dùng cache => decode
  nhanh hơn (không chiếu lại key mỗi bước như bản cũ).
- Thêm model.solve(env, decode_type, start_nodes) trả thẳng action_sequence
  để GA chỉ cần đổi 1 chỗ (xem ghi chú cuối file).

Tương thích: model._get_embeddings(locs, demands) vẫn trả embeddings ĐÃ encode
(chạy hết encoder layer), giống bản cũ.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Giá trị khởi tạo nhẹ cho hệ số distance-bias. softplus(-2) ≈ 0.13 nên lúc
# bắt đầu bias yếu (model gần như không thiên vị), training sẽ tự tăng nếu cần.
_DIST_ALPHA_INIT = -2.0


def _scale_factor(num_nodes_including_depot: int) -> float:
    """Hệ số scale theo kích thước bài toán (ICAM conditions on N)."""
    return math.log2(max(int(num_nodes_including_depot), 2))


# =========================================================
# MULTI-HEAD ATTENTION (thủ công, có distance-bias)
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wo = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _split(self, x):
        # [B, L, E] -> [B, H, L, D]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, q_in, k_in, v_in, bias=None, mask=None):
        B = q_in.size(0)

        q = self._split(self.Wq(q_in))
        k = self._split(self.Wk(k_in))
        v = self._split(self.Wv(v_in))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, H, Lq, Lk]

        if bias is not None:
            # bias: [B, Lq, Lk] -> broadcast theo head
            scores = scores + bias.unsqueeze(1)

        if mask is not None:
            # mask: [B, Lk] (True = cấm) -> broadcast [B,1,1,Lk]
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, Lq, D]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embedding_dim)
        return self.Wo(out)


# =========================================================
# ENCODER LAYER
# =========================================================

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden=512):
        super().__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embedding_dim),
        )

        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.norm2 = nn.BatchNorm1d(embedding_dim)

        # Hệ số distance-bias riêng cho mỗi layer (scalar, learnable).
        self.dist_alpha = nn.Parameter(torch.tensor(_DIST_ALPHA_INIT))

    def _bn(self, norm, h):
        # h: [B, M, E] -> BatchNorm trên kênh E
        return norm(h.transpose(1, 2)).transpose(1, 2)

    def forward(self, h, dist=None, scale=1.0):
        bias = None
        if dist is not None:
            # Node càng xa, bias càng âm => attention càng nhỏ.
            bias = -F.softplus(self.dist_alpha) * scale * dist  # [B, M, M]

        h = h + self.mha(h, h, h, bias=bias)
        h = self._bn(self.norm1, h)

        h = h + self.ff(h)
        h = self._bn(self.norm2, h)
        return h


# =========================================================
# DECODER (POMO-ready + distance-bias)
# =========================================================

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8, tanh_clipping=10.0):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.tanh_clipping = tanh_clipping

        # context = graph_embedding + current_node_embedding + remaining_capacity
        self.context_dim = embedding_dim * 2 + 1

        # Glimpse (multi-head)
        self.Wq_glimpse = nn.Linear(self.context_dim, embedding_dim, bias=False)
        self.Wk_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wo_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Output (single-head)
        self.Wq_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Hệ số distance-bias cho decoder.
        self.dist_alpha = nn.Parameter(torch.tensor(_DIST_ALPHA_INIT))

        self._cache = None

    # ---- precompute: chiếu key/value một lần cho cả quá trình decode ----
    def precompute(self, embeddings):
        self._cache = {
            "embeddings": embeddings,                 # [B, M, E]
            "graph": embeddings.mean(dim=1),          # [B, E]
            "Kg": self.Wk_glimpse(embeddings),        # [B, M, E]
            "Vg": self.Wv_glimpse(embeddings),        # [B, M, E]
            "Ko": self.Wk_out(embeddings),            # [B, M, E]
        }

    def _split_kv(self, x):
        # [B, M, E] -> [B, H, M, D]
        B, M, _ = x.shape
        return x.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _split_q(self, x):
        # [B, E] -> [B, H, 1, D]
        B = x.size(0)
        return x.view(B, self.num_heads, self.head_dim).unsqueeze(2)

    def forward(self, current_node, remaining_capacity, mask,
                dist=None, scale=1.0, embeddings=None):
        """
        current_node: [B]
        remaining_capacity: [B]
        mask: [B, M]  (True = cấm chọn)
        dist: [B, M, M] hoặc None (để bật distance-bias)
        embeddings: nếu truyền vào sẽ precompute lại; nếu None dùng cache.
        """
        if embeddings is not None:
            self.precompute(embeddings)

        c = self._cache
        emb = c["embeddings"]
        B, M, E = emb.shape
        idx = torch.arange(B, device=emb.device)

        cur_emb = emb[idx, current_node]  # [B, E]
        context = torch.cat(
            [c["graph"], cur_emb, remaining_capacity.unsqueeze(-1)], dim=-1
        )  # [B, 2E+1]

        # Distance-bias từ current node tới mọi node: [B, M]
        bias = None
        if dist is not None:
            dist_cur = dist[idx, current_node]  # [B, M]
            bias = -F.softplus(self.dist_alpha) * scale * dist_cur

        # ---- Multi-head glimpse ----
        q = self._split_q(self.Wq_glimpse(context))  # [B, H, 1, D]
        k = self._split_kv(c["Kg"])                  # [B, H, M, D]
        v = self._split_kv(c["Vg"])                  # [B, H, M, D]

        comp = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # comp: [B, H, 1, M]
        if bias is not None:
            comp = comp + bias[:, None, None, :]
        comp = comp.masked_fill(mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(comp, dim=-1)
        heads = torch.matmul(attn, v).squeeze(2).contiguous().view(B, E)  # [B, E]
        glimpse = self.Wo_glimpse(heads)  # [B, E]

        # ---- Single-head output ----
        q_out = self.Wq_out(glimpse)  # [B, E]
        logits = torch.matmul(q_out.unsqueeze(1), c["Ko"].transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.embedding_dim)  # [B, M]

        if bias is not None:
            logits = logits + bias

        logits = self.tanh_clipping * torch.tanh(logits)
        logits = logits.masked_fill(mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return probs, logits

    def select_node(self, probs, mask, decode_type="sampling"):
        if decode_type == "greedy":
            selected = probs.argmax(dim=-1)
        elif decode_type == "sampling":
            selected = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Phòng sample nhầm node bị mask do lỗi số học.
            invalid = mask.gather(1, selected.unsqueeze(1)).squeeze(1)
            while invalid.any():
                resampled = torch.multinomial(probs[invalid], num_samples=1).squeeze(1)
                selected[invalid] = resampled
                invalid = mask.gather(1, selected.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        selected_probs = probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        log_prob = torch.log(selected_probs + 1e-12)
        return selected, log_prob


# =========================================================
# MAIN MODEL
# =========================================================

class CVRPModel(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8, num_layers=3, tanh_clipping=10.0):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Encoder embeddings
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed_customers = nn.Linear(3, embedding_dim)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads) for _ in range(num_layers)]
        )

        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            tanh_clipping=tanh_clipping,
        )

        self._init_parameters()

    def _init_parameters(self):
        # uniform(-1/sqrt(d), 1/sqrt(d)) cho các ma trận weight.
        # Các scalar (dist_alpha, dim==0) được bỏ qua => giữ _DIST_ALPHA_INIT.
        for param in self.parameters():
            if param.dim() > 1:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    # ---- Encoder: trả embeddings ĐÃ encode ----
    def _get_embeddings(self, locs, demands, dist=None):
        depot_loc = locs[:, 0:1, :]                       # [B, 1, 2]
        customer_locs = locs[:, 1:, :]                    # [B, n, 2]
        M = locs.size(1)

        # Chấp nhận demands ở 2 dạng:
        #   [B, N+1] (gồm depot ở index 0, như env.get_state trả về)
        #   [B, N]   (chỉ khách, như cvrp_data.make_batch trả về)
        if demands.size(1) == M:
            customer_demands = demands[:, 1:].unsqueeze(-1)   # bỏ depot
        elif demands.size(1) == M - 1:
            customer_demands = demands.unsqueeze(-1)
        else:
            raise ValueError(
                f"demands phải có shape [B,N] hoặc [B,N+1], nhận {tuple(demands.shape)}"
            )

        customer_features = torch.cat([customer_locs, customer_demands], dim=-1)

        h = torch.cat(
            [
                self.init_embed_depot(depot_loc),
                self.init_embed_customers(customer_features),
            ],
            dim=1,
        )  # [B, M, E]

        if dist is None:
            dist = torch.cdist(locs, locs, p=2)

        scale = _scale_factor(h.size(1))

        for layer in self.encoder_layers:
            h = layer(h, dist=dist, scale=scale)

        return h

    # ---- Vòng decode dùng chung cho train (forward) và infer (solve) ----
    def _run_decode(self, env, decode_type, start_nodes=None, record_actions=False):
        state = env.get_state()
        dist = state["dist"]

        embeddings = self._get_embeddings(state["locs"], state["demands"], dist=dist)
        self.decoder.precompute(embeddings)

        scale = _scale_factor(embeddings.size(1))

        B = env.batch_size
        device = embeddings.device

        log_probs_sum = torch.zeros(B, device=device)
        rewards_sum = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        actions = [] if record_actions else None

        step_idx = 0
        while not done.all():
            prev_done = done.clone()
            mask = env.get_mask()

            probs, _ = self.decoder(
                current_node=state["current_node"],
                remaining_capacity=state["remaining_capacity"],
                mask=mask,
                dist=dist,
                scale=scale,
            )

            if step_idx == 0 and start_nodes is not None:
                # POMO multi-start: ép customer xuất phát.
                # Bước ép KHÔNG phải quyết định của policy => log_prob = 0
                # (không đưa gradient cho node bị ép chọn).
                action = start_nodes.to(device).long()
                log_prob = torch.zeros(B, device=device)
            else:
                action, log_prob = self.decoder.select_node(probs, mask, decode_type)

            state, reward, done = env.step(action)

            # Không cộng cho hàng đã done từ trước bước này.
            alive = (~prev_done).float()
            rewards_sum += reward * alive
            log_probs_sum += log_prob * alive

            if record_actions:
                actions.append(action)

            step_idx += 1

        if record_actions:
            actions = torch.stack(actions, dim=1)  # [B, T]

        return rewards_sum, log_probs_sum, actions

    def forward(self, env, decode_type="sampling", start_nodes=None):
        """
        Rollout cho train/validation.
        return:
            tour_rewards: [B]  (= -tour_length [- vehicle_penalty*K nếu env bật])
            total_log_probs: [B]
        """
        rewards, log_probs, _ = self._run_decode(
            env, decode_type, start_nodes=start_nodes, record_actions=False
        )
        return rewards, log_probs

    @torch.no_grad()
    def solve(self, env, decode_type="greedy", start_nodes=None):
        """
        Rollout cho inference (dùng trong GA).
        return: actions [B, T]  — chuỗi node đã thăm, gồm cả depot (=0).
        Từ chuỗi này dựng permutation + route_markers như cũ
        (mỗi lần gặp 0 là mở route mới).

        Tự chuyển sang eval() để BatchNorm dùng running-stats và KHÔNG cập nhật
        chúng lúc inference (no_grad không tự làm việc này), rồi khôi phục mode cũ.
        """
        was_training = self.training
        self.eval()
        try:
            _, _, actions = self._run_decode(
                env, decode_type, start_nodes=start_nodes, record_actions=True
            )
        finally:
            if was_training:
                self.train()
        return actions


# =========================================================
# GHI CHÚ TÍCH HỢP (cho bước sau, KHÔNG sửa ở đây)
# =========================================================
# 1) GA.solve_remaining_with_neural: thay vòng decode thủ công bằng:
#        env.reset(batch_size=1, locs=locs, demands=demands)
#        actions = model.solve(env, decode_type=decode_type)[0].tolist()
#    rồi dựng perm/markers từ `actions` y như logic cũ (gặp 0 -> route mới).
#    Không cần gọi _get_embeddings / decoder thủ công nữa.
#
# 2) Training (train_v2) sẽ:
#    - bật use_vehicle_penalty trong CVRPenv (tune vehicle_penalty trên thang [0,1]),
#    - dùng POMO: nhân bản instance, truyền start_nodes là các customer khác nhau,
#      baseline = mean reward theo nhóm start của cùng instance,
#    - validate với augment_xy_data_by_8_fold + best-of-8.
#   (Phần này để lượt sau theo yêu cầu.)
