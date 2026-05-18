import torch
import dataloader as pi_loader

from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from torchvision import transforms
from typing import Any, Dict, List

from network import PiVisionNet, AreaAttentionNet
from loss import ExistDistCosSinLoss, HungarianPosExistLoss, heatmap_loss, SortedPosExistLoss

import os, re
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from glob import glob
import math
import time

# --- transforms (adjust size to your backbone) ---
# IMG_SIZE = (256, 256)
IMG_SIZE = (128, 128)

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=(1.0, 1.7)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def save_ckpt(path, model, optimizer, epoch, best_val=None, scheduler=None, extra=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,                 # 1-based epoch index
        "best_val": best_val,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0)          # resume from next epoch
    best_val = ckpt.get("best_val", float("inf"))
    extra = ckpt.get("extra", None)
    return start_epoch + 1, best_val, extra     # next_epoch, best_val, extra

def latest_epoch_ckpt(save_dir):
    paths = glob(os.path.join(save_dir, "epoch_*.ckpt"))
    if not paths: return None
    def ep(path):
        m = re.search(r"epoch_(\d+)\.ckpt$", path)
        return int(m.group(1)) if m else -1
    return max(paths, key=ep)


def train_with_resume(
    model, criterion, optimizer, train_loader, val_loader=None,
    epochs=10, device="cuda", log_dir="runs/exp1", scheduler=None,
    log_images_every=200, save_dir="checkpoints", keep_last=5,
    resume_from=None, auto_resume_latest=False
):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    bb_name = model.backbone_name

    # ---- Resume logic ----
    if resume_from is None and auto_resume_latest:
        resume_from = latest_epoch_ckpt(save_dir)
    if resume_from:
        start_epoch, best_val_loss, _ = load_ckpt(
            resume_from, model, optimizer, scheduler, map_location="cpu"
        )
        print(f"Resumed from {resume_from} -> start_epoch={start_epoch}, best_val={best_val_loss:.6f}")
    else:
        start_epoch, best_val_loss = 1, float("inf")

    recent_ckpts = deque(maxlen=keep_last)
    # pre-fill deque with existing recent ckpts if resuming
    for p in sorted(glob(os.path.join(save_dir, "epoch_*.ckpt"))):
        recent_ckpts.append(p)

    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        # ---------------- Train ----------------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", unit="batch")
        for step, (imgs, labels) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(imgs)
            loss, loss_parts = criterion(preds, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler and getattr(scheduler, "_step_every_batch", False):
                scheduler.step()  # for per-step schedulers like OneCycle; set this flag yourself

            running += loss.item()
            avg_loss = running / step
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            # for key in loss_parts.keys():
            #     writer.add_scalar(f"train/{key}", loss_parts[key].item(), global_step)

            if log_images_every and (global_step % log_images_every == 0):
                mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1,3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1,3,1,1)
                vis = torch.clamp(imgs[:16] * std + mean, 0, 1)
                writer.add_image("train/images", make_grid(vis, nrow=4), global_step)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            global_step += 1

        train_loss_epoch = running / max(1, len(train_loader))
        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)

        if scheduler and not getattr(scheduler, "_step_every_batch", False):
            scheduler.step()  # per-epoch schedulers

        # ---------------- Validate ----------------
        val_loss_epoch = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Val   {epoch}/{epochs}", unit="batch"):
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = model(imgs)
                    val_loss, val_loss_parts = criterion(preds, labels)
                    val_running += val_loss.item()
            val_loss_epoch = val_running / max(1, len(val_loader))
            writer.add_scalar("val/loss_epoch", val_loss_epoch, epoch)

            # ---- save best ----
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_path = os.path.join(save_dir, f"best_{bb_name}.ckpt")
                save_ckpt(best_path, model, optimizer, epoch, best_val=best_val_loss, scheduler=scheduler)
                print(f"✅ Saved BEST: {best_path} (val={best_val_loss:.6f})")

        # ---- save last N ----
        # ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.ckpt")
        # save last
        ckpt_path = os.path.join(save_dir, f"last_{bb_name}.ckpt")
        save_ckpt(ckpt_path, model, optimizer, epoch, best_val=best_val_loss, scheduler=scheduler)
        # recent_ckpts.append(ckpt_path)
        # # prune older than keep_last
        # while len(recent_ckpts) > keep_last:
        #     old = recent_ckpts.popleft()
        #     if os.path.exists(old):
        #         os.remove(old)

        print(f"Epoch {epoch}: train={train_loss_epoch:.4f}" +
              (f", val={val_loss_epoch:.4f}" if val_loss_epoch is not None else ""))

    writer.close()

def train_with_tb(
    model, criterion, optimizer, train_loader, val_loader=None,
    epochs=10, device="cuda", log_dir="runs/exp1", scheduler=None,
    log_images_every=200, save_dir="checkpoints", keep_last=5
):
    os.makedirs(save_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import make_grid
    from tqdm import tqdm

    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    best_val_loss = float("inf")
    recent_ckpts = deque(maxlen=keep_last)  # track last N checkpoints
    global_step = 0

    for epoch in range(1, epochs + 1):
        # ---------------- Train ----------------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", unit="batch")

        for step, (imgs, labels) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(imgs)
            loss, loss_parts = criterion(preds, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            avg_loss = running / step
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            for key in loss_parts.keys():
                writer.add_scalar(f"train/{key}", loss_parts[key].item(), global_step)

            # Optional: log images
            if log_images_every and (global_step % log_images_every == 0):
                mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1,3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1,3,1,1)
                vis = torch.clamp(imgs[:16] * std + mean, 0, 1)
                grid = make_grid(vis, nrow=4)
                writer.add_image("train/images", grid, global_step)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            global_step += 1

        train_loss_epoch = running / max(1, len(train_loader))
        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)

        if scheduler is not None:
            scheduler.step()

        # ---------------- Validate ----------------
        val_loss_epoch = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Val   {epoch}/{epochs}", unit="batch"):
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = model(imgs)
                    val_loss, val_loss_parts = criterion(preds, labels)
                    val_running += val_loss.item()

            val_loss_epoch = val_running / max(1, len(val_loader))
            writer.add_scalar("val/loss_epoch", val_loss_epoch, epoch)

            # --- Save best ---
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_path = os.path.join(save_dir, "best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"✅ Saved best model: {best_path} (val_loss={best_val_loss:.4f})")

        # --- Save last N checkpoints ---
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        recent_ckpts.append(ckpt_path)

        # Remove oldest if exceeding keep_last
        while len(recent_ckpts) > keep_last:
            old_ckpt = recent_ckpts.popleft()
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
                print(f"Deleted old checkpoint: {old_ckpt}")

        # Print epoch summary
        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.4f}"
              + (f", val_loss={val_loss_epoch:.4f}" if val_loss_epoch is not None else ""))

    writer.close()
    
def get_coord_dataset(data_folder):
    # --- dataset construction ---
    # IMPORTANT: set your paths and timestamp column names if different
    dataset = pi_loader.PiCarDataset(
        roots=data_folder,
        csv_glob="*.csv",
        transform=train_transform,          # we’ll override for val subset below
        target_transform=None,              # you can plug one in if needed
        strict=False,
        keep_last_n_levels=2, 
        max_time_diff=None,
        train=True
    )

    # --- train/val split ---
    val_fraction = 0.1
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # give val subset a different (non-augmenting) transform
    # random_split returns Subset; override its dataset transform via a wrapper
    # Simple approach: set attribute on the base dataset during loaders creation.
    # We'll pass a transform override via lambda in the DataLoader collate if needed,
    # but easier: duplicate the dataset object for val with same samples.

    # Create a lightweight twin dataset that shares the built samples
    val_dataset = pi_loader.PiCarDataset(
        roots=[]  # we won't rebuild; we copy state below
    )
    # share internal state
    val_dataset.samples = dataset.samples
    val_dataset.transform = val_transform
    val_dataset.target_transform = dataset.target_transform
    val_dataset.__len__ = dataset.__len__
    val_dataset.__getitem__ = dataset.__getitem__

    # Re-wrap the same indices for validation

    val_ds = Subset(val_dataset, val_ds.indices)

    # --- DataLoaders ---
    num_workers = 4
    batch_size = 32

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # collate_fn=smart_collate,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # collate_fn=smart_collate,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_heatmap_dataset(data_folder):
    # --- dataset construction ---
    # IMPORTANT: set your paths and timestamp column names if different
    dataset = pi_loader.PiCarDataset(
        roots=data_folder,
        csv_glob="*.csv",
        transform=train_transform,          # we’ll override for val subset below
        target_transform=None,              # you can plug one in if needed
        strict=False,
        keep_last_n_levels=2, 
        max_time_diff=None,
        train=True,
        label_type='heatmap',
        heatmap_size=(500, 500),
        heatmap_coord_range=((0.0, 2000.0), (-1000.0, 1000.0)),
        heatmap_sigma_range=(45,135),
    )

    # --- train/val split ---
    val_fraction = 0.1
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # give val subset a different (non-augmenting) transform
    # random_split returns Subset; override its dataset transform via a wrapper
    # Simple approach: set attribute on the base dataset during loaders creation.
    # We'll pass a transform override via lambda in the DataLoader collate if needed,
    # but easier: duplicate the dataset object for val with same samples.

    # Create a lightweight twin dataset that shares the built samples
    val_dataset = pi_loader.PiCarDataset(
        label_type='heatmap',
        heatmap_size=(500, 500),
        heatmap_coord_range=((0.0, 2000.0), (-1000.0, 1000.0)),
        heatmap_sigma_range=(45,135),
        train=True,
        roots=[]  # we won't rebuild; we copy state below
    )
    # share internal state
    val_dataset.samples = dataset.samples
    val_dataset.transform = val_transform
    val_dataset.target_transform = dataset.target_transform
    val_dataset.__len__ = dataset.__len__
    val_dataset.__getitem__ = dataset.__getitem__

    # Re-wrap the same indices for validation

    val_ds = Subset(val_dataset, val_ds.indices)

    # --- DataLoaders ---
    num_workers = 4
    batch_size = 32

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # collate_fn=smart_collate,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # collate_fn=smart_collate,
        drop_last=False
    )
    
    return train_loader, val_loader
    
def get_imaging_hangar_dataset(data_folder):
    imaging_hangar_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=(2.7, 3.7)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    dataset = pi_loader.PicarImgPoseDataset(data_folder, transform=imaging_hangar_transform)

    # --- train/val split ---
    val_fraction = 0.1
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    # give val subset a different (non-augmenting) transform
    # random_split returns Subset; override its dataset transform via a wrapper
    # Simple approach: set attribute on the base dataset during loaders creation.
    # We'll pass a transform override via lambda in the DataLoader collate if needed,
    # but easier: duplicate the dataset object for val with same samples.

    # Create a lightweight twin dataset that shares the built samples
    val_dataset = pi_loader.PicarImgPoseDataset("")
    
    # share internal state
    val_dataset._index = dataset._index
    val_dataset.transform = val_transform
    val_dataset.transform = dataset.transform
    val_dataset.__len__ = dataset.__len__
    val_dataset.__getitem__ = dataset.__getitem__

    # Re-wrap the same indices for validation

    val_ds = Subset(val_dataset, val_ds.indices)

    # --- DataLoaders ---
    num_workers = 4
    batch_size = 32

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # collate_fn=smart_collate,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # collate_fn=smart_collate,
        drop_last=False
    )
    
    return train_loader, val_loader

def train_simple():
    data_folder = [#'/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S2',
                    # '/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S1',
                    # '/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S3',
                    # '/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S5',
                    '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S1',
                    '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S2',
                    '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S3',
                    '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S4',
                    '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S5',
                    '/home/lab/Documents/picar/2025_train_data/test_set_7cars/2025_0828_1519-S7']
    
    train_loader, val_loader = get_coord_dataset(data_folder)
    
    backbone = "resnet18"
    # backbone = "mobilenet_v3_large"
    # backbone = "mobilenet_v3_small"
    model = PiVisionNet(backbone_name=backbone, pretrained=True, out_agents=4)  
    # criterion = ExistDistCosSinLoss(w_exist=1.0, w_dist=1.0, w_dir=1.0)        
    # criterion = HungarianPosExistLoss(
    #     w_pos=1.0,
    #     w_exist=1.0,
    #     use_smoothl1=True,
    #     pos_scale=1.0,     # tune if x/y are in pixels
    #     miss_penalty=1.0   # penalty for missed targets (false negatives)
    # )
    criterion = SortedPosExistLoss(
        w_pos=1.0,
        w_exist=1.0,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # optional scheduler example
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    train_with_resume(
        model, criterion, optimizer, train_loader, val_loader,
        epochs=400, device="cuda", log_dir=f"/home/lab/Documents/picar/2025_train_data/runs/5cars_{backbone}_128", save_dir="checkpoints_sorted",
        scheduler=scheduler, keep_last=5, #auto_resume_latest=True,
        # resume_from=f"checkpoints_sorted/last_{backbone}.ckpt"
    )
    
def train_heatmap():
    data_folder = ['/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S2',
                    '/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S1',
                    '/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S3',
                    '/home/lab/Documents/picar/2025_train_data/2025_0821_1457-S5']
    
    train_loader, val_loader = get_heatmap_dataset(data_folder)
    backbone = "resnet18"
    
    model = AreaAttentionNet(backbone_name=backbone, pretrained=True, heat_shape=(500, 500))
    criterion = heatmap_loss
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    train_with_resume(
        model, criterion, optimizer, train_loader, val_loader,
        epochs=100, device="cuda", log_dir=f"/home/lab/Documents/picar/2025_train_data/runs/heatmap_{backbone}_128", save_dir="checkpoints_heatmap",
        scheduler=scheduler, keep_last=5, auto_resume_latest=True,
        #resume_from=f"checkpoints_heatmap/last_{backbone}.ckpt"
    )
    
def train_imaging_hangar():
    train_loader, val_loader = get_imaging_hangar_dataset('/home/lab/Documents/picar/2025_train_data/imaging_hangar')
    backbone = "resnet18"
    
    model = PiVisionNet(backbone_name=backbone, pretrained=True, out_agents=5)
    
    criterion = SortedPosExistLoss(
        w_pos=1.0,
        w_exist=1.0,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # optional scheduler example
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    train_with_resume(
        model, criterion, optimizer, train_loader, val_loader,
        epochs=400, device="cuda", log_dir=f"/home/lab/Documents/picar/2025_train_data/runs/7cars_imaging_hangar_{backbone}_128", save_dir="checkpoints_imaging_hangar_sorted",
        scheduler=scheduler, keep_last=5, #auto_resume_latest=True,
        # resume_from=f"checkpoints_sorted/last_{backbone}.ckpt"
    )

if __name__ == '__main__':
    train_imaging_hangar()