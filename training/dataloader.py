import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import List, Tuple, Dict, Optional

class AutoPercepDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 img_size: Tuple[int, int] = (224, 224),
                 pos_scale: float = 10.0,
                 yaw_scale: float = 180.0,
                 max_neighbors: int = 5,
                 max_distance_meters: float = 15.0, # STRICT FILTER: Ignore anything farther than this
                 visibility_threshold: float = 12.0,
                 unit_scale: float = 1000.0
                 ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.pos_scale = pos_scale
        self.yaw_scale = yaw_scale
        self.max_neighbors = max_neighbors
        self.max_distance_meters = max_distance_meters
        self.visibility_threshold = visibility_threshold
        self.unit_scale = unit_scale
        self.samples = []
        
        print(f"🔍 Scanning for datasets in: {root_dir}")
        print(f"   🚫 Filtering out neighbors > {max_distance_meters}m away...")
        self._load_and_preprocess_data()
        print(f"✅ Loaded {len(self.samples)} valid samples after cleaning and distance filtering.")

    def _load_and_preprocess_data(self):
        """
        Recursively finds CSVs, resolves image paths, converts mm->m,
        cleans NaNs, and computes relative labels.
        """
        # Initialize unit conversion factor (mm to meters)
        self.unit_scale = 1000.0 
        
        # Find all records_*.csv files recursively
        csv_pattern = os.path.join(self.root_dir, "**", "records_*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        if not csv_files:
            raise FileNotFoundError(f"No 'records_*.csv' files found in {self.root_dir}")

        print(f"   Found {len(csv_files)} CSV files.")

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                
                # 1. Validate Columns
                base_cols = ['image', 'self_pose_x', 'self_pose_y', 'self_pose_angle']
                if not all(col in df.columns for col in base_cols):
                    print(f"   ⚠️ Skipping {csv_path}: Missing base pose columns.")
                    continue

                # 2. Drop NaNs in critical SELF columns first
                initial_len = len(df)
                df.dropna(subset=base_cols, inplace=True)
                if len(df) < initial_len:
                    print(f"   - {os.path.basename(csv_path)}: Dropped {initial_len - len(df)} rows with missing self-pose.")

                if len(df) == 0:
                    continue

                # 3. Resolve Image Paths
                csv_dir = os.path.dirname(csv_path)
                subdirs = [d for d in os.listdir(csv_dir) if os.path.isdir(os.path.join(csv_dir, d))]
                
                img_subdir = None
                for sub in subdirs:
                    sample_files = glob.glob(os.path.join(csv_dir, sub, "*.png"))
                    if sample_files:
                        img_subdir = sub
                        break
                
                if not img_subdir:
                    print(f"   ⚠️ No image subfolder found for {csv_path}. Skipping.")
                    continue
                
                img_base_path = os.path.join(csv_dir, img_subdir)

                # 4. Process Row by Row
                valid_rows_count = 0
                for idx, row in df.iterrows():
                    img_name = str(row['image'])
                    if not img_name.endswith('.png'):
                        img_name += '.png'
                    
                    img_full_path = os.path.join(img_base_path, os.path.basename(img_name))
                    
                    if not os.path.exists(img_full_path):
                        continue 

                    # --- EXTRACT & CONVERT SELF POSES (mm -> meters) ---
                    self_x = row['self_pose_x'] / self.unit_scale
                    self_y = row['self_pose_y'] / self.unit_scale
                    self_ang = row['self_pose_angle'] # Degrees

                    neighbors = []
                    # Added S2 since your CSV has it!
                    neighbor_ids = ['S1', 'S2', 'S3'] 
                    
                    for nid in neighbor_ids:
                        x_col = f'car_{nid}_pose_x'
                        y_col = f'car_{nid}_pose_y'
                        ang_col = f'car_{nid}_pose_angle'

                        # Check if columns exist
                        if x_col in row and y_col in row and ang_col in row:
                            # Check for NaNs explicitly before conversion
                            if pd.isna(row[x_col]) or pd.isna(row[y_col]) or pd.isna(row[ang_col]):
                                continue

                            # --- CONVERT NEIGHBOR POSES (mm -> meters) ---
                            nx = row[x_col] / self.unit_scale
                            ny = row[y_col] / self.unit_scale
                            nang = row[ang_col]
                            
                            # Compute Relative Transformations
                            rel_x, rel_y = self._world_to_ego(self_x, self_y, self_ang, nx, ny)
                            rel_yaw = self._compute_relative_yaw(self_ang, nang)
                            
                            # Determine Distance
                            dist = np.sqrt(rel_x**2 + rel_y**2)
                            
                            # FILTER: If distance is > 20m, it's likely a ghost reading, wrong sync, or global offset error
                            if dist > 20.0: 
                                continue

                            exists = 1.0
                            # Visibility based on threshold (e.g., 15m)
                            visible = 1.0 if dist < self.visibility_threshold else 0.0

                            neighbors.append({
                                'x': rel_x,
                                'y': rel_y,
                                'yaw': rel_yaw,
                                'exists': exists,
                                'visible': visible
                            })

                    if not neighbors:
                        continue 

                    self.samples.append({
                        'img_path': img_full_path,
                        'neighbors': neighbors
                    })
                    valid_rows_count += 1

                print(f"   - {os.path.basename(csv_path)}: Kept {valid_rows_count} valid rows after mm-conversion & distance filtering.")

            except Exception as e:
                print(f"   ❌ Error processing {csv_path}: {e}")
                import traceback
                traceback.print_exc()
    def _world_to_ego(self, self_x, self_y, self_ang_deg, nb_x, nb_y):
        theta = np.deg2rad(self_ang_deg)
        dx = nb_x - self_x
        dy = nb_y - self_y
        rel_x = dx * np.cos(theta) + dy * np.sin(theta)
        rel_y = -dx * np.sin(theta) + dy * np.cos(theta)
        return rel_x, rel_y

    def _compute_relative_yaw(self, self_ang_deg, nb_ang_deg):
        diff = nb_ang_deg - self_ang_deg
        while diff <= -180: diff += 360
        while diff > 180: diff -= 360
        return diff

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
            else:
                img = np.zeros((3, self.img_size[1], self.img_size[0]), dtype=np.float32)
        else:
            img = np.zeros((3, self.img_size[1], self.img_size[0]), dtype=np.float32)

        targets = np.zeros((self.max_neighbors, 4), dtype=np.float32)
        vis_mask = np.zeros((self.max_neighbors), dtype=np.float32)
        
        n_actual = min(len(sample['neighbors']), self.max_neighbors)
        
        for i in range(n_actual):
            nb = sample['neighbors'][i]
            targets[i, 0] = nb['x'] / self.pos_scale
            targets[i, 1] = nb['y'] / self.pos_scale
            targets[i, 2] = nb['yaw'] / self.yaw_scale
            targets[i, 3] = nb['exists']
            vis_mask[i] = nb['visible']

        return torch.from_numpy(img), torch.from_numpy(targets), torch.from_numpy(vis_mask), n_actual

    @staticmethod
    def collate_fn(batch):
        images, targets, vis_masks, counts = zip(*batch)
        return torch.stack(images), torch.stack(targets), torch.stack(vis_masks), torch.tensor(counts)

def get_dataloader(root_dir: str, batch_size: int = 8, num_workers: int = 4, **kwargs):
    dataset = AutoPercepDataset(root_dir=root_dir, **kwargs)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return loader, dataset