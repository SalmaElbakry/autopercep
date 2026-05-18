import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

import dataloader as pi_loader

from torchvision import transforms

from network import PiVisionNet


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm_imagenet(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: torch.Tensor (C,H,W), ImageNet normalized
    returns: np.uint8 (H,W,3) in [0,255]
    """
    if img_chw.ndim != 3:
        raise ValueError("Expected CHW tensor for image.")
    x = img_chw.detach().cpu()
    x = (x * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    x = (x.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    return x

def parse_label_agents(label: dict):
    """
    Extracts self pose and agent poses from label dict.
    Returns:
      self_pose: (x, y, angle_deg)
      agents: list of dicts {id, x, y, angle_deg}
    """
    # Self
    try:
        self_x = float(label['self_pose_x'])
        self_y = float(label['self_pose_y'])
        self_ang = float(label['self_pose_angle'])
        self_pose = (self_x, self_y, self_ang)
    except KeyError as e:
        raise KeyError(f"Missing self pose key in label: {e}")

    # Agents
    agents = {}
    pat = re.compile(r"^(?P<prefix>.+)_pose_(?P<field>x|y|angle)$")
    for k, v in label.items():
        m = pat.match(k)
        if not m: 
            continue
        prefix = m.group("prefix")
        field = m.group("field")
        if prefix == "self":  # already handled
            continue
        agents.setdefault(prefix, {})[field] = float(v)

    out = []
    for prefix, d in agents.items():
        if all(f in d for f in ("x", "y", "angle")):
            # Try to shorten label e.g. "car_S2" -> "S2"
            short_id = prefix
            m = re.match(r".*_(\w+)$", prefix)
            if m:
                short_id = m.group(1)
            out.append({"id": short_id, "x": d["x"], "y": d["y"], "angle_deg": d["angle"]})
    return self_pose, out

def rotate_translate_points(rel_xy: np.ndarray, origin_xy: tuple, angle_deg: float) -> np.ndarray:
    """
    Convert points in self-frame to world frame:
      world_xy = R(angle_deg) @ rel_xy + origin
    rel_xy: (N,2), origin_xy: (x0, y0)
    """
    if rel_xy.size == 0:
        return rel_xy
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    world = rel_xy @ R.T
    world[:,0] += origin_xy[0]
    world[:,1] += origin_xy[1]
    return world

def arrows_from_pose(ax, x, y, angle_deg, length=80.0, **kwargs):
    """Draw a heading arrow at (x,y) pointing along angle_deg."""
    theta = math.radians(angle_deg)
    dx, dy = length*math.cos(theta), length*math.sin(theta)
    ax.arrow(x, y, dx, dy, head_width=length*0.25, head_length=length*0.35, length_includes_head=True, **kwargs)

def get_sample(dataset, idx):
    """Support (img, label) or dict styles."""
    sample = dataset[idx]
    if isinstance(sample, tuple) and len(sample) == 2:
        img, label = sample
    elif isinstance(sample, dict):
        img = sample.get("image", sample.get("img", None))
        label = sample.get("label", sample)
        if img is None:
            raise ValueError("Could not find 'image' key in dict sample.")
    else:
        raise ValueError("Dataset __getitem__ must return (image, label_dict) or a dict.")
    if not isinstance(label, dict):
        raise ValueError("Label must be a dict containing pose fields.")
    return img, label

def _ray_endpoint_to_axes(ax, x0, y0, angle_deg):
    """
    From (x0,y0), cast a ray at angle_deg and return its intersection with ax's
    current rectangular limits. Returns (x1, y1).
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)

    # Parametric ray: x = x0 + t*c, y = y0 + t*s, t >= 0
    candidates = []

    # Intersect with verticals x = xmin/xmax
    if abs(c) > 1e-12:
        t = (xmin - x0) / c
        if t >= 0:
            y = y0 + t*s
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append((t, xmin, y))
        t = (xmax - x0) / c
        if t >= 0:
            y = y0 + t*s
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append((t, xmax, y))

    # Intersect with horizontals y = ymin/ymax
    if abs(s) > 1e-12:
        t = (ymin - y0) / s
        if t >= 0:
            x = x0 + t*c
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append((t, x, ymin))
        t = (ymax - y0) / s
        if t >= 0:
            x = x0 + t*c
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append((t, x, ymax))

    if not candidates:
        # Fallback: extend a long segment if something degenerate happens
        L = max(xmax - xmin, ymax - ymin) * 2.0
        return (x0 + L*c, y0 + L*s)

    # Pick the nearest positive intersection
    _, x1, y1 = min(candidates, key=lambda tup: tup[0])
    return x1, y1

def _draw_fov_rays(ax, x0, y0, facing_deg, delta_deg=30.0, **plot_kw):
    """
    Draw two rays from (x0,y0) at facing ± delta_deg to the axes bounds.
    """
    for sign in (+1, -1):
        ang = facing_deg + sign * delta_deg
        x1, y1 = _ray_endpoint_to_axes(ax, x0, y0, ang)
        ax.plot([x0, x1], [y0, y1], **plot_kw)


@torch.no_grad()
def make_video(dataset,
               trained_model,
               output_mp4="output.mp4",
               fps=5,
               device=None,
               conf_thresh=0.5,
               arrow_len=80.0,
               point_size=30.0,
               square_size=1400,
               max_frames=None):
    """
    Iterate dataset, run model, render side-by-side (image / world map), and save MP4.
    """
    trained_model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)

    # Prepare figure
    plt.ioff()
    fig = plt.figure(figsize=(10, 5), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1,1.2])
    ax_img = fig.add_subplot(gs[0,0])
    ax_map = fig.add_subplot(gs[0,1])

    writer = FFMpegWriter(fps=fps, bitrate=1800, codec='mpeg4')

    n_total = len(dataset) if max_frames is None else min(max_frames, len(dataset))

    with writer.saving(fig, output_mp4, dpi=150):
        for idx in tqdm(range(n_total), desc="Rendering"):
            # --- Get data
            image_chw, label = get_sample(dataset, idx)
            if isinstance(image_chw, np.ndarray):
                image_chw = torch.from_numpy(image_chw)
            image_chw = image_chw.float()
            # Keep original normalized tensor for model; denorm for display later
            disp_hw3 = denorm_imagenet(image_chw)

            # --- Poses
            self_pose, agents = parse_label_agents(label)
            self_x, self_y, self_ang = self_pose

            # --- Model prediction (relative to self)
            model_in = image_chw.unsqueeze(0).to(device)      # [1,C,H,W]
            pred = trained_model(model_in)                    # expect [1,N,3] or [N,3]
            if pred.ndim == 3:
                pred = pred[0]
            pred = pred.detach().cpu().float().numpy()        # (N,3)
            if pred.size == 0:
                pred_rel = np.zeros((0,2), dtype=np.float32)
            else:
                conf = pred[:,2]
                keep = conf >= conf_thresh
                pred_rel = pred[keep, :2]

            # --- Convert predictions to world frame
            pred_world = rotate_translate_points(pred_rel, (self_x, self_y), self_ang)

            # --- Prepare map data
            # Collect ground-truth agent/world points for bounds
            gt_pts = [(self_x, self_y)]
            gt_pts += [(a["x"], a["y"]) for a in agents]
            gt_pts = np.array(gt_pts, dtype=np.float32)

            all_x = [gt_pts[:,0].min(), gt_pts[:,0].max()] if gt_pts.size else []
            all_y = [gt_pts[:,1].min(), gt_pts[:,1].max()] if gt_pts.size else []
            if pred_world.size:
                all_x += [pred_world[:,0].min(), pred_world[:,0].max()]
                all_y += [pred_world[:,1].min(), pred_world[:,1].max()]

            if all_x and all_y:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)
                margin = 0.15 * max(1.0, max(xmax - xmin, ymax - ymin))
                xmin, xmax = xmin - margin, xmax + margin
                ymin, ymax = ymin - margin, ymax + margin
            else:
                # Fallback window around self
                xmin, xmax = self_x - 300, self_x + 300
                ymin, ymax = self_y - 300, self_y + 300

            # --- Draw left image
            ax_img.clear()
            ax_img.imshow(disp_hw3)
            ax_img.set_title(f"Frame {idx}")
            ax_img.axis("off")

            # --- Draw right world map
            ax_map.clear()
            ax_map.set_aspect("equal", adjustable="box")

            # 🔒 fixed limits instead of auto zoom
            ax_map.set_xlim(-square_size, square_size)
            ax_map.set_ylim(-square_size, square_size)

            ax_map.grid(True, linestyle="--", alpha=0.3)
            ax_map.set_title("World positions (GT agents + predicted targets)")

            # Plot self
            ax_map.scatter([self_x], [self_y], s=point_size*1.5, marker="o", label="self")
            arrows_from_pose(ax_map, self_x, self_y, self_ang, length=arrow_len, alpha=0.8)
            
            # >>> Add ±30° rays relative to self's facing <<<
            # Make sure limits are already set before computing intersections:
            # (you already set fixed limits to [-1400, 1400] above)
            _draw_fov_rays(
                ax_map, self_x, self_y, self_ang, delta_deg=30.0,
                linewidth=0.5, linestyle="-.", alpha=0.8, color='red'
            )           

            # Plot agents (GT)
            if agents:
                ax_map.scatter(
                    [a["x"] for a in agents],
                    [a["y"] for a in agents],
                    s=point_size, marker="o", label="agents (GT)"
                )
                for a in agents:
                    arrows_from_pose(ax_map, a["x"], a["y"], a["angle_deg"], length=arrow_len*0.9, alpha=0.8)
                    ax_map.text(a["x"], a["y"], f"{a['id']}", fontsize=8, ha="left", va="bottom")

            # Plot predictions (rotated into world)
            if pred_world.size:
                ax_map.scatter(pred_world[:,0], pred_world[:,1], s=point_size, marker="x", label="predicted (world)")

                # 🟢 draw connecting lines from self to each detected position
                for (px, py) in pred_world:
                    ax_map.plot([self_x, px], [self_y, py], "r--", linewidth=1, alpha=0.7)

            ax_map.legend(loc="upper right", fontsize=8, frameon=True)

            # --- Write frame
            writer.grab_frame()

    print(f"✅ Saved video to: {output_mp4}")

def load_model_from_ckpt(ckpt_path, backbone="resnet18", out_agents=2, device="cuda"):
    model = PiVisionNet(backbone_name=backbone, pretrained=False, out_agents=out_agents)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # two formats supported:
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.to(device).eval()
    return model

def create_demo_from_folders(model, demo_folder, video_name):
    # --- transforms (adjust size to your backbone) ---
    IMG_SIZE = (128, 128)

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # --- dataset construction ---
    # IMPORTANT: set your paths and timestamp column names if different
    demo_dataset = pi_loader.PiCarDataset(
        roots=demo_folder,
        csv_glob="*.csv",
        transform=train_transform,          # we’ll override for val subset below
        target_transform=None,              # you can plug one in if needed
        strict=False,
        keep_last_n_levels=2, 
        max_time_diff=None,                 # e.g., 0.050 for 50 ms tolerance
        train=False
    )
    
    make_video(demo_dataset, model, output_mp4=video_name)
    
def create_demo_from_imaging_hangar(model, demo_folder, video_name):
    # --- transforms (adjust size to your backbone) ---
    IMG_SIZE = (128, 128)

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=(2.7, 3.7)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # --- dataset construction ---
    # IMPORTANT: set your paths and timestamp column names if different
    demo_dataset = pi_loader.PicarImgPoseDataset(demo_folder, transform=train_transform, train=False)
    print(f"data size: {len(demo_dataset)}")
    
    make_video(demo_dataset, model, output_mp4=video_name, square_size=3000)
    
if __name__ == '__main__':
    # backbone = 'mobilenet_v3_large'
    # backbone = 'resnet18'
    # backbone = 'mobilenet_v3_small'
    
    backbones = ['resnet18']#, 'mobilenet_v3_large', 'mobilenet_v3_small']
    
    for backbone in backbones:
        trained_model = load_model_from_ckpt(f"/home/lab/Documents/picar/picar_ros2/training/checkpoints_imaging_hangar_sorted/best_{backbone}.ckpt", backbone=backbone, out_agents=5)
        trained_model.eval()
        
        # folders = ['/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S1',
        #            '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S2',
        #            '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S3',
        #            '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S4',
        #            '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S5',
        #            '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S7',]
        
        folders = ['/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S5']
        folders = ['/home/lab/Documents/picar/2025_train_data/imaging_hangar']
        
        finish_count = 0
        for folder in folders:
            vid_name = f"demo_videos/{folder.split('/')[-1]}_{backbone}_128.mp4"
            # create_demo_from_folders(trained_model, [folder], vid_name)
            create_demo_from_imaging_hangar(trained_model, folder, vid_name)
            finish_count += 1
            print(f"Video {finish_count}/{len(folders)} done")