import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import itertools
from scipy.optimize import linear_sum_assignment


class ExistDistCosSinLoss(nn.Module):
    def __init__(self, w_exist=1.0, w_dist=1.0, w_dir=1.0, dist_log_space=False, eps=1e-8):
        super().__init__()
        self.w_exist = w_exist
        self.w_dist = w_dist
        self.w_dir = w_dir
        self.dist_log_space = dist_log_space
        self.eps = eps
        self.softplus = nn.Softplus()
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target):
        """
        pred:   [B,2,4] = (z_dist, z_cos, z_sin, logit_exist)
        target: [B,2,4] = (dist,   cos,   sin,  exist in {0,1})
        """
        z_dist  = pred[..., 0]
        z_cos   = pred[..., 1]
        z_sin   = pred[..., 2]
        z_exist = pred[..., 3]

        dist_t  = target[..., 0]
        cos_t   = target[..., 1]
        sin_t   = target[..., 2]
        exist_t = target[..., 3]  # 0/1

        # === existence classification ===
        exist_loss = self.bce_logits(z_exist, exist_t).mean()

        # === mask for present objects ===
        m = (exist_t > 0.5).float()  # [B,2]

        # distance
        if self.dist_log_space:
            dist_p = torch.exp(z_dist)
            dist_err = torch.abs(
                torch.log(torch.clamp(dist_p, min=self.eps)) -
                torch.log(torch.clamp(dist_t,  min=self.eps))
            )
        else:
            dist_p = self.softplus(z_dist)
            dist_err = self.smoothl1(dist_p, dist_t)  # [B,2]

        # direction: unit-vector MSE
        vec = torch.stack([z_cos, z_sin], dim=-1)               # [B,2,2]
        norm = torch.clamp(vec.norm(dim=-1, keepdim=True), min=self.eps)
        unit_p = vec / norm
        unit_t = torch.stack([cos_t, sin_t], dim=-1)
        dir_err = ((unit_p - unit_t) ** 2).sum(dim=-1)          # [B,2]

        # masked reductions
        denom = m.sum() + self.eps
        dist_loss = (m * dist_err).sum() / denom
        dir_loss  = (m * dir_err).sum()  / denom

        return self.w_exist*exist_loss + self.w_dist*dist_loss + self.w_dir*dir_loss

class SortedPosExistLoss(nn.Module):
    """
    pred:   [B, Np, 3] = (x̂, ŷ, z_logit)
    target: [B, Nt, 3] = (x,  y, appear in [0,1])  # soft or hard

    - Sort pred/target by radius sqrt(x^2+y^2), nearest first.
    - Match index-wise after sorting.
    - SmoothL1 on (x,y), weighted by target 'appear'.
    - BCEWithLogits on existence (matched vs appear, unmatched preds as negatives).
    """
    def __init__(
        self,
        beta: float = 1.0,        # SmoothL1 beta
        w_pos: float = 1.0,       # weight for position loss
        w_exist: float = 1.0,     # weight for existence loss
        pos_scale: float = 1.0,   # scale for (x,y), e.g., 1000.0 if mm
        eps: float = 1e-8
    ):
        super().__init__()
        self.w_pos = float(w_pos)
        self.w_exist = float(w_exist)
        self.pos_scale = float(pos_scale)
        self.eps = float(eps)

        # Use reduction='none' so we can apply custom weighting/averaging
        self.smoothl1 = nn.SmoothL1Loss(beta=beta, reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    @staticmethod
    def _sort_indices_near_to_far(xy: torch.Tensor) -> torch.Tensor:
        # xy: [B, N, 2] -> indices [B, N] sorting by r^2 ascending
        r2 = (xy ** 2).sum(dim=-1)         # [B, N]
        return torch.argsort(r2, dim=1)    # [B, N]

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.ndim == 3 and pred.size(-1) >= 3
        assert target.ndim == 3 and target.size(-1) >= 3

        B, Np = pred.size(0), pred.size(1)
        Nt = target.size(1)

        pred_xy = pred[..., :2]
        pred_logit = pred[..., 2]
        tgt_xy = target[..., :2]
        tgt_appear = target[..., 2].clamp(0.0, 1.0)

        # sort near → far
        # if Np > 0:  # do not sort prediction
        #     idx_p = self._sort_indices_near_to_far(pred_xy)
        #     pred_xy = torch.gather(pred_xy, 1, idx_p.unsqueeze(-1).expand(-1, -1, 2))
        #     pred_logit = torch.gather(pred_logit, 1, idx_p)
        if Nt > 0:
            idx_t = self._sort_indices_near_to_far(tgt_xy)
            tgt_xy = torch.gather(tgt_xy, 1, idx_t.unsqueeze(-1).expand(-1, -1, 2))
            tgt_appear = torch.gather(tgt_appear, 1, idx_t)

        M = min(Np, Nt)

        # --- Position loss (matched only), SmoothL1 weighted by appear ---
        if M > 0:
            p_xy_m = pred_xy[:, :M, :] / self.pos_scale
            t_xy_m = tgt_xy[:, :M, :] / self.pos_scale
            appear_m = tgt_appear[:, :M]                 # [B, M]

            # SmoothL1 returns [B, M, 2] with reduction='none'
            s1 = self.smoothl1(p_xy_m, t_xy_m).sum(dim=-1)  # [B, M] sum over x,y
            weight_sum = appear_m.sum().clamp_min(self.eps)
            loss_pos = (s1 * appear_m).sum() / weight_sum
        else:
            loss_pos = pred_xy.new_zeros(())

        # --- Existence loss ---
        loss_exist = pred_xy.new_zeros(())
        if M > 0:
            # matched existence vs appear (soft OK)
            exist_matched = self.bce(pred_logit[:, :M], tgt_appear[:, :M])  # [B, M]
            loss_exist = loss_exist + exist_matched.mean()

        if Np > M:
            # unmatched predictions are negatives
            neg_logits = pred_logit[:, M:]                      # [B, Np-M]
            zeros = torch.zeros_like(neg_logits)
            exist_unmatched = self.bce(neg_logits, zeros)       # [B, Np-M]
            loss_exist = loss_exist + exist_unmatched.mean()

        # total
        loss_total = self.w_pos * loss_pos + self.w_exist * loss_exist

        return loss_total, {
            "loss_pos": loss_pos.detach(),
            "loss_exist": loss_exist.detach(),
            "matched": M,
        }

class HungarianPosExistLoss(nn.Module):
    """
    Hungarian matching with variable counts (N_pred != N_true supported).

    pred:   [B, Np, 3] = (x̂, ŷ, z_exist)               # z_exist: logits
    target: [B, Nt, 3] = (x,  y,  appear in [0,1])      # soft or hard

    Real↔Real cost(i,j):
      C = w_pos * (SmoothL1(x̂_i, x_j) + SmoothL1(ŷ_i, y_j)) * appear_j * pos_scale
        + w_exist * BCEWithLogits(z_exist_i, appear_j)

    Extra preds (Np > Nt): add Nt..Np-1 **dummy targets**:
      C(i, dummy) = w_exist * BCEWithLogits(z_exist_i, 0), position term = 0

    Missing preds (Np < Nt): add Np..Nt-1 **dummy preds**:
      C(dummy, j) = miss_penalty * appear_j    # constant, no gradient

    After assignment:
      - For real↔real matches: compute pos/exist losses with gradients
      - For real↔dummy matches: add BCE(z_exist, 0) for that pred (gradients)
      - For dummy↔real matches: add miss_penalty * appear (no gradients)
    """
    def __init__(self,
                 w_pos=1.0,
                 w_exist=1.0,
                 use_smoothl1=True,
                 beta=1.0,
                 pos_scale=1.0,
                 miss_penalty=1.0,
                 eps=1e-8):
        super().__init__()
        self.w_pos = w_pos
        self.w_exist = w_exist
        self.pos_scale = pos_scale
        self.miss_penalty = miss_penalty
        self.eps = eps

        if use_smoothl1:
            try:
                self.pos_loss_fn = nn.SmoothL1Loss(reduction='none', beta=beta)
            except TypeError:
                self.pos_loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            self.pos_loss_fn = nn.MSELoss(reduction='none')

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    @torch.no_grad()
    def _build_cost_matrix(self, pred_b, tgt_b):
        """
        Build a square cost matrix with dummies if needed.
        pred_b: [Np, 3], tgt_b: [Nt, 3]
        Returns:
          C (np.ndarray) of shape [M, M],
          meta dict with bookkeeping for real/dummy splits.
        """
        xhat = pred_b[:, 0]; yhat = pred_b[:, 1]; zlog = pred_b[:, 2]
        x = tgt_b[:, 0];     y = tgt_b[:, 1];     app  = tgt_b[:, 2]

        Np = pred_b.shape[0]
        Nt = tgt_b.shape[0]
        M = max(Np, Nt)

        # numpy copies (safe on CPU for SciPy)
        xhat_np = xhat.detach().cpu().numpy() if Np > 0 else np.zeros((0,), np.float64)
        yhat_np = yhat.detach().cpu().numpy() if Np > 0 else np.zeros((0,), np.float64)
        zlog_np = zlog.detach().cpu().numpy() if Np > 0 else np.zeros((0,), np.float64)
        x_np    = x.detach().cpu().numpy()   if Nt > 0 else np.zeros((0,), np.float64)
        y_np    = y.detach().cpu().numpy()   if Nt > 0 else np.zeros((0,), np.float64)
        app_np  = app.detach().cpu().numpy() if Nt > 0 else np.zeros((0,), np.float64)

        C = np.zeros((M, M), dtype=np.float64)

        # --- Fill real↔real block ---
        for i in range(Np):
            if Nt == 0: break
            # L1 approx for speed when building costs; final loss uses SmoothL1
            pos_err = (np.abs(xhat_np[i] - x_np) + np.abs(yhat_np[i] - y_np)) * app_np * self.pos_scale
            z = zlog_np[i]
            bce = np.maximum(z, 0.0) - z * app_np + np.log1p(np.exp(-np.abs(z)))  # BCEWithLogits
            C[i, :Nt] = self.w_pos * pos_err + self.w_exist * bce

        # --- Extra preds: real pred ↔ dummy target ---
        # Encourage "no-object" for those preds
        if Np > Nt:
            # vectorized BCE(z_logits, 0) = softplus(z)
            softplus = np.log1p(np.exp(-np.abs(zlog_np))) + np.maximum(zlog_np, 0.0)
            for i in range(Np):
                C[i, Nt:M] = self.w_exist * softplus[i]  # same cost against any dummy target

        # --- Missing preds: dummy pred ↔ real target ---
        # Penalize unassigned real targets proportionally to appear
        if Nt > Np:
            miss = self.miss_penalty * app_np  # [Nt]
            for j in range(Nt):
                C[Np:M, j] = miss[j]  # constant per column j

        # (Optional) dummy↔dummy block remains zeros; it won't be picked if any real cost > 0
        meta = {"Np": Np, "Nt": Nt, "M": M}
        return C, meta

    def forward(self, pred, target):
        """
        pred:   [B, Np, 3]
        target: [B, Nt, 3]
        """
        assert pred.dim() == 3 and target.dim() == 3
        assert pred.shape[0] == target.shape[0], "Batch size mismatch"
        B = pred.shape[0]

        # Accumulators
        total_pos = pred.new_tensor(0.0)
        total_exist = pred.new_tensor(0.0)
        total_miss = pred.new_tensor(0.0)

        for b in range(B):
            pred_b = pred[b]   # [Np,3]
            tgt_b  = target[b] # [Nt,3]

            # Build cost matrix & run Hungarian
            C, meta = self._build_cost_matrix(pred_b, tgt_b)
            row_ind, col_ind = linear_sum_assignment(C)  # arrays of length M
            Np, Nt, M = meta["Np"], meta["Nt"], meta["M"]

            # We’ll compute losses on 3 groups:
            # 1) real pred ↔ real tgt
            # 2) real pred ↔ dummy tgt (no-object)
            # 3) dummy pred ↔ real tgt (miss penalty)

            # --- 1) real↔real ---
            real_pairs = []
            for i, j in zip(row_ind, col_ind):
                if i < Np and j < Nt:  # both real
                    real_pairs.append((i, j))

            if real_pairs:
                idx_p = torch.as_tensor([i for i, _ in real_pairs], device=pred.device, dtype=torch.long)
                idx_t = torch.as_tensor([j for _, j in real_pairs], device=pred.device, dtype=torch.long)

                xhat = pred_b[idx_p, 0]
                yhat = pred_b[idx_p, 1]
                zlog = pred_b[idx_p, 2]

                x = tgt_b[idx_t, 0]
                y = tgt_b[idx_t, 1]
                app = tgt_b[idx_t, 2]

                pos_err = self.pos_loss_fn(xhat, x) + self.pos_loss_fn(yhat, y)  # [K]
                # masked/weighted by appear (soft)
                denom = app.sum() + self.eps
                pos_loss = (pos_err * app * self.pos_scale).sum() / denom

                exist_loss = self.bce(zlog, app).mean()

                total_pos += pos_loss
                total_exist += exist_loss

            # --- 2) real pred ↔ dummy tgt (no-object for those preds) ---
            if Np > Nt:
                # find assignments where j >= Nt and i < Np
                nz = [
                    i for i, j in zip(row_ind, col_ind)
                    if (i < Np and j >= Nt)
                ]
                if nz:
                    idx = torch.as_tensor(nz, device=pred.device, dtype=torch.long)
                    zlog = pred_b[idx, 2]
                    # BCE vs target 0 (absent)
                    exist_noobj = self.bce(zlog, torch.zeros_like(zlog))
                    total_exist += exist_noobj.mean()  # these contribute to exist term

            # --- 3) dummy pred ↔ real tgt (misses) ---
            if Nt > Np:
                # find assignments where i >= Np and j < Nt
                misses = [
                    j for i, j in zip(row_ind, col_ind)
                    if (i >= Np and j < Nt)
                ]
                if misses:
                    idx = torch.as_tensor(misses, device=pred.device, dtype=torch.long)
                    app = tgt_b[idx, 2]  # [K_miss]
                    # constant penalty per missed real target (soft by app)
                    miss_loss = (self.miss_penalty * app).mean()
                    total_miss += miss_loss

        # Average over batch
        loss = (self.w_pos * (total_pos / B)
                + self.w_exist * (total_exist / B)
                + (total_miss / B))
        
        loss_parts = {"position_loss": self.w_pos * (total_pos / B),
                      "existence_loss": self.w_exist * (total_exist / B),
                      "miss_loss": (total_miss / B)}
        
        return loss, loss_parts

def heatmap_loss(pred, target, mask=None):
    """
    MSE loss on two heatmaps
    pred:   (B, C, H, W)  raw values (no sigmoid)
    target: (B, C, H, W)  in [0,1] (e.g., Gaussians)
    mask:   (B, C, 1, 1) or (B, C, H, W), 1=use, 0=ignore (optional)
    """
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum().clamp_min(1))
    return loss.mean(), 0