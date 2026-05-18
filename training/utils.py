import torch
import math

import torch.nn.functional as F

from typing import Dict, Any
from network import AreaAttentionNet, PiVisionNet


# @torch.no_grad()
# def points_to_heatmap(
#     points: torch.Tensor,                 # (N, 3): x, y, exist(0/1)
#     heatmap_size=(100, 100),              # (H, W)
#     coord_range=((0.0, 1000.0), (0.0, 1000.0)),  # ((x_min, x_max), (y_min, y_max)) in world units
#     sigma=2.5,                             # Gaussian sigma
#     sigma_in_world=False,                  # if True, 'sigma' is in world units; else in heatmap pixels
#     accumulate="max",                      # "max" or "sum"
#     clamp_01=True                          # clamp final map to [0,1] (useful for "sum")
# ):
#     """
#     Convert (x,y,exist) points to a single-channel heatmap.
#     - points outside coord_range or with exist==0 are ignored.
#     - y is mapped to vertical axis [0..H-1], x to horizontal [0..W-1].
#     """
#     assert points.ndim == 2 and points.size(1) >= 3, "points must be (N,3)"
#     H, W = int(heatmap_size[0]), int(heatmap_size[1])

#     device = points.device
#     dtype  = points.dtype

#     x_min, x_max = coord_range[0]
#     y_min, y_max = coord_range[1]
#     # Avoid div-by-zero
#     sx = (W - 1) / max(x_max - x_min, 1e-8)
#     sy = (H - 1) / max(y_max - y_min, 1e-8)

#     # Filter valid points: exist==1 and finite & within range (optional)
#     exist = points[:, 2] > 0.5
#     xs = points[:, 0]
#     ys = points[:, 1]
#     finite = torch.isfinite(xs) & torch.isfinite(ys)
#     # Keep even if out of range? Here we drop ones far outside to save compute:
#     in_range = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
#     valid = exist & finite & in_range

#     if valid.sum() == 0:
#         return torch.zeros(H, W, device=device, dtype=dtype)

#     xs = xs[valid]
#     ys = ys[valid]

#     # Map world coords -> heatmap pixel coords
#     xh = (xs - x_min) * sx  # [0..W-1]
#     yh = (ys - y_min) * sy  # [0..H-1]

#     # Sigma in pixels
#     if sigma_in_world:
#         # convert world sigma to pixel sigma; use average scale for isotropic blur
#         # (you can also pass a tuple or different sigmas per axis if desired)
#         sigma_px = float(sigma) * float(0.5 * (sx + sy))
#     else:
#         sigma_px = float(sigma)

#     # Build grid
#     yy, xx = torch.meshgrid(
#         torch.arange(H, device=device, dtype=dtype),
#         torch.arange(W, device=device, dtype=dtype),
#         indexing="ij"
#     )  # (H, W)

#     # Vectorized Gaussians: produce (M, H, W) then reduce
#     M = xh.numel()
#     xx = xx.unsqueeze(0)                   # (1, H, W)
#     yy = yy.unsqueeze(0)                   # (1, H, W)
#     xh = xh.view(M, 1, 1)
#     yh = yh.view(M, 1, 1)

#     dist2 = (xx - xh) ** 2 + (yy - yh) ** 2
#     g = torch.exp(-dist2 / (2.0 * (sigma_px ** 2) + 1e-12))  # (M, H, W)
#     # Normalize each Gaussian peak to 1.0 (optional; common practice)
#     # (Already true since exp(0)=1 at the center.)

#     if accumulate == "max":
#         heat = g.max(dim=0).values
#     elif accumulate == "sum":
#         heat = g.sum(dim=0)
#     else:
#         raise ValueError("accumulate must be 'max' or 'sum'")

#     if clamp_01:
#         heat = heat.clamp_(0, 1)

#     return heat

@torch.no_grad()
def points_to_heatmap(
    points: torch.Tensor,                 # (N, 3): x, y, exist(0/1)
    heatmap_size=(100, 100),
    coord_range=((0.0, 1000.0), (0.0, 1000.0)),
    sigma_min=1.5,
    sigma_max=6.0,
    accumulate="max"
):
    """
    Adaptive Gaussian sigma: larger near origin (0,0), smaller far away.
    """
    assert points.ndim == 2 and points.size(1) >= 3, "points must be (N,3)"
    H, W = heatmap_size
    device = points.device
    dtype  = points.dtype

    x_min, x_max = coord_range[0]
    y_min, y_max = coord_range[1]
    sx = (W - 1) / max(x_max - x_min, 1e-8)
    sy = (H - 1) / max(y_max - y_min, 1e-8)

    exist = points[:, 2] > 0.5
    xs, ys = points[:, 0], points[:, 1]
    valid = exist & torch.isfinite(xs) & torch.isfinite(ys)
    if valid.sum() == 0:
        return torch.zeros(H, W, device=device, dtype=dtype)

    xs = xs[valid]
    ys = ys[valid]

    # compute distance from origin
    r = torch.sqrt(xs**2 + ys**2)
    r_max = torch.sqrt(torch.tensor(x_max**2 + y_max**2, device=device))
    sigma_world = sigma_max - (sigma_max - sigma_min) * (r / r_max).clamp(0, 1)

    # map world coords -> pixel coords
    xh = (xs - x_min) * sx
    yh = (ys - y_min) * sy

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    heat = torch.zeros(H, W, device=device, dtype=dtype)
    for i in range(xh.numel()):
        sigma_px = 0.5 * (sx + sy) * sigma_world[i]
        g = torch.exp(-((xx - xh[i])**2 + (yy - yh[i])**2) / (2.0 * sigma_px**2))
        heat = torch.maximum(heat, g) if accumulate == "max" else heat + g

    if accumulate != "max":
        heat = heat.clamp_(0, 1)

    return heat

@torch.no_grad()
def heatmap_to_points(
    heatmap: torch.Tensor,                 # (H, W)
    coord_range=((0.0, 1000.0), (0.0, 1000.0)),  # ((x_min,x_max),(y_min,y_max))
    threshold=0.3,
    nms_kernel=3,
    max_points=None
):
    """
    Detect local maxima in a heatmap and map them back to world coordinates.
    
    Returns:
        coords_world: (N, 3) tensor [x_world, y_world, score]
    """
    assert heatmap.ndim == 2, "heatmap must be (H, W)"
    H, W = heatmap.shape
    device = heatmap.device
    dtype  = heatmap.dtype

    # --- Non-maximum suppression ---
    pad = nms_kernel // 2
    pooled = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), 
                          kernel_size=nms_kernel, stride=1, padding=pad)
    keep = (heatmap == pooled[0,0]) & (heatmap >= threshold)

    # --- Extract peak positions ---
    ys, xs = torch.where(keep)
    scores = heatmap[ys, xs]

    if ys.numel() == 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)

    # --- Sort by score (descending) ---
    scores, order = scores.sort(descending=True)
    xs, ys = xs[order], ys[order]

    if max_points is not None:
        xs, ys, scores = xs[:max_points], ys[:max_points], scores[:max_points]

    # --- Convert to world coordinates ---
    (x_min, x_max), (y_min, y_max) = coord_range
    x_world = xs.float() / (W - 1) * (x_max - x_min) + x_min
    y_world = ys.float() / (H - 1) * (y_max - y_min) + y_min

    coords_world = torch.stack([x_world, y_world, scores], dim=1)  # (N,3)
    return coords_world


def wrap180(angle_deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    return (angle_deg + 180) % 360 - 180

# --- your helpers unchanged ---
def relative_angle_to(target_x, target_y, self_x, self_y, self_heading_deg):
    dx = target_x - self_x
    dy = target_y - self_y
    world_bearing = math.degrees(math.atan2(dy, dx))       # [-180,180)
    rel = world_bearing - self_heading_deg
    rel = (rel + 180) % 360 - 180                           # wrap to [-180,180)
    return rel

# def relative_angle_to(tx, ty, ta, sx, sy, sa):
#     # vector from self → agent
#     dx, dy = tx - sx, ty - sy
#     angle_to_agent = math.degrees(math.atan2(dy, dx))
    
#     # relative angle from self's head
#     angle_from_self_head = wrap180(angle_to_agent - sa)
    
#     # vector from agent → self
#     dx2, dy2 = sx - tx, sy - ty
#     angle_to_self = math.degrees(math.atan2(dy2, dx2))
    
#     # relative face angle of agent toward self
#     relative_face_to_self = wrap180(angle_to_self - ta)
    
#     return (angle_from_self_head, relative_face_to_self)

def relative_distance_to(target_x, target_y, self_x, self_y):
    return math.hypot(target_x - self_x, target_y - self_y)

def calculate_dist_angles(pos_dict):
    sx, sy, sh = pos_dict['self_pose_x'], pos_dict['self_pose_y'], pos_dict['self_pose_angle']
    agents = sorted({key.split('_pose')[0] for key in pos_dict if '_pose' in key and not key.startswith('self')})

    angles = []
    dists = []
    for agent in agents:
        angles.append(relative_angle_to(pos_dict[f'{agent}_pose_x'], pos_dict[f'{agent}_pose_y'], sx, sy, sh))
        dists.append(relative_distance_to(pos_dict[f'{agent}_pose_x'], pos_dict[f'{agent}_pose_y'], sx, sy))

    return dists, angles

def confidence_from_angle(angle_deg, threshold, binary=True):
    # angle_global, angle_to_self = angle_deg
    a = abs(angle_deg)
    midpoint = threshold
    k = 1.0
    if binary:
        # if a <= threshold - 3:
        #     return 1
        # elif a > threshold:
        #     return 0
        # else:
        #     return int((angle_global * angle_to_self) < 0)
        return int((angle_deg >= threshold[0]) and (angle_deg <= threshold[1]))
    else:
        return 1 / (1 + math.exp(k * (a - midpoint)))
    
def compute_head_rel_xy_angle(data: Dict[str, Any], head_offset: float = 70.0) -> Dict[str, Dict[str, float]]:
    """
    Compute (dx, dy) of other agents' head tips relative to self's head tip,
    and the relative angle of those head tips w.r.t. self's head direction.

    Conventions:
      - Angles are in degrees, wrapped to (-180, 180].
      - dx, dy are coordinates of (agent_head - self_head) in the global frame.
      - angle_deg is: angle_between(self_head -> agent_head line and self's head direction).
        0° = straight ahead of self; +CCW, -CW.

    Args:
        data: dict with keys like:
              'self_pose_x', 'self_pose_y', 'self_pose_angle',
              and for agents: 'car_*_pose_x', 'car_*_pose_y', 'car_*_pose_angle'
        head_offset: distance from pose (x, y) to head tip along heading.

    Returns:
        Dict[agent, {'dx': float, 'dy': float, 'angle_deg': float}]
    """
    # Self head tip
    sx, sy, sa = float(data["self_pose_x"]), float(data["self_pose_y"]), float(data["self_pose_angle"])
    sa_rad = math.radians(sa)
    self_head_x = sx + head_offset * math.cos(sa_rad)
    self_head_y = sy + head_offset * math.sin(sa_rad)

    results: Dict[str, Dict[str, float]] = {}

    for key in data:
        if key.startswith("car_") and key.endswith("_pose_x"):
            agent = key[: -len("_pose_x")]  # e.g., 'car_S2'
            ax = float(data[f"{agent}_pose_x"])
            ay = float(data[f"{agent}_pose_y"])
            aa = float(data[f"{agent}_pose_angle"])

            # Agent head tip in global coords
            aa_rad = math.radians(aa)
            agent_head_x = ax + head_offset * math.cos(aa_rad)
            agent_head_y = ay + head_offset * math.sin(aa_rad)

            # Relative vector from self head -> agent head (global frame)
            dx = agent_head_x - self_head_x
            dy = agent_head_y - self_head_y

            dx_local =  dx * math.cos(-sa_rad) - dy * math.sin(-sa_rad)
            dy_local =  dx * math.sin(-sa_rad) + dy * math.cos(-sa_rad)

            # angle to +x axis (self's forward)
            angle_local = wrap180(math.degrees(math.atan2(dy_local, dx_local)))

            # Angle of that vector relative to self's head direction
            # angle_global = math.degrees(math.atan2(dy, dx))
            # angle_rel = wrap180(angle_global - sa)

            results[agent] = {
                "dx": dx_local,
                "dy": dy_local,
                "angle_deg": angle_local,
            }

    return results

def compute_head_rel_local_with_blocking(data: dict, head_offset: float = 50.0, angle_thresh: float = 5.0, column_keys=["_pose_x", "_pose_y", "_pose_angle"]):
    """
    Compute other agents' head positions relative to self's head in self-head local coordinates.
    Add 'blocked' flag when an agent is behind another within angle_thresh.
    
    Returns:
        dict[agent] = {
            'dx_local': float,
            'dy_local': float,
            'angle_deg': float,  # relative to +x axis (self's head forward)
            'dist': float,       # distance from self's head
            'blocked': int       # 1 if occluded by another agent, else 0
        }
    """
    sx, sy, sa = float(data[f"self{column_keys[0]}"]), float(data[f"self{column_keys[1]}"]), float(data[f"self{column_keys[2]}"])
    sa_rad = math.radians(sa)

    # self head tip
    self_head_x = sx + head_offset * math.cos(sa_rad)
    self_head_y = sy + head_offset * math.sin(sa_rad)

    results = {}

    # --- First compute dx, dy, angle, dist ---
    for key in data:
        if key.startswith("car_") and key.endswith(column_keys[0]):
            agent = key[: -len(column_keys[0])]
            ax = float(data[f"{agent}{column_keys[0]}"])
            ay = float(data[f"{agent}{column_keys[1]}"])
            aa = float(data[f"{agent}{column_keys[2]}"])

            # agent head global
            aa_rad = math.radians(aa)
            agent_head_x = ax + head_offset * math.cos(aa_rad)
            agent_head_y = ay + head_offset * math.sin(aa_rad)

            # global diff
            dx = agent_head_x - self_head_x
            dy = agent_head_y - self_head_y

            # rotate into self's local frame
            dx_local =  dx * math.cos(-sa_rad) - dy * math.sin(-sa_rad)
            dy_local =  dx * math.sin(-sa_rad) + dy * math.cos(-sa_rad)

            # angle and distance in local frame
            angle_local = wrap180(math.degrees(math.atan2(dy_local, dx_local)))
            dist = math.hypot(dx_local, dy_local)

            results[agent] = {
                "dx_local": dx_local,
                "dy_local": dy_local,
                "angle_deg": angle_local,
                "dist": dist,
                "blocked": 0  # initialize
            }

    # --- Now check blocking condition ---
    agents = list(results.keys())
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i == j:
                continue
            ai, aj = agents[i], agents[j]
            ang_diff = wrap180(results[ai]["angle_deg"] - results[aj]["angle_deg"])
            if abs(ang_diff) < angle_thresh:
                # if ai is farther, mark as blocked
                if results[ai]["dist"] > results[aj]["dist"]:
                    results[ai]["blocked"] = 1

    return results

def rel_positions_tensor(pos_dict, threshold, device=None, dtype=torch.float32, column_keys=["_pose_x", "_pose_y", "_pose_angle"]):
    """
    pos_dict keys:
      car_*_pose_x, car_*_pose_y, car_*_pose_angle,
      self_pose_x, self_pose_y, self_pose_angle
    Returns:
      rel_xy:  [N_agents, 2] tensor of (x_rel, y_rel) in self frame
      agents:  list of agent names in the same order as rows in rel_xy
    """
    
    rel_pos_dict = compute_head_rel_local_with_blocking(pos_dict, column_keys=column_keys)
    rel_xy = torch.tensor([[pose['dx_local'], pose['dy_local'], confidence_from_angle(pose['angle_deg'], threshold) * (1 - pose['blocked'])] for pose in rel_pos_dict.values()])
    
    # # find agent prefixes (e.g., 'car_S1', 'car_S2')
    # agents = sorted({k.split("_pose")[0] for k in pos_dict if k.startswith("car")})

    # # self pose
    # sx = torch.tensor(pos_dict["self_pose_x"], device=device, dtype=dtype)
    # sy = torch.tensor(pos_dict["self_pose_y"], device=device, dtype=dtype)
    # sh_deg = torch.tensor(pos_dict["self_pose_angle"], device=device, dtype=dtype)
    # sh_rad = sh_deg * math.pi / 180.0

    # # stack agent positions
    # ax = torch.tensor([pos_dict[f"{a}_pose_x"] for a in agents], device=device, dtype=dtype)  # [N]
    # ay = torch.tensor([pos_dict[f"{a}_pose_y"] for a in agents], device=device, dtype=dtype)  # [N]
    # adeg = [relative_angle_to(pos_dict[f"{a}_pose_x"], pos_dict[f"{a}_pose_y"], pos_dict[f"{a}_pose_angle"], sx, sy, sh_deg) for a in agents]
    
    # aconf = torch.tensor([confidence_from_angle(deg, threshold) for deg in adeg], device=device, dtype=dtype)

    # # translate so self at origin
    # dx = ax - sx  # [N]
    # dy = ay - sy  # [N]

    # # rotate by -self_heading -> align self heading to +x
    # c = torch.cos(-sh_rad)
    # s = torch.sin(-sh_rad)
    # # [N,2] = [ [c, -s], [s, c] ] @ [dx, dy]
    # rx = c * dx - s * dy
    # ry = s * dx + c * dy
    # rel_xy = torch.stack([rx, ry, aconf], dim=-1)  # [N,3]

    return rel_xy

def load_model_from_ckpt(ckpt_path, backbone="resnet18", model_type='coord', out_agents=2, heat_shape=(500, 500), device="cuda"):
    if model_type == 'coord':
        model = PiVisionNet(backbone_name=backbone, pretrained=False, out_agents=out_agents)
    elif model_type == 'heatmap':
        model = AreaAttentionNet(backbone_name=backbone, pretrained=False, heat_shape=heat_shape)
    else:
        raise(Exception('model type invalid'))
    
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # two formats supported:
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.to(device).eval()
    return model