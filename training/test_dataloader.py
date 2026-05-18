import os
import argparse
import torch
from torch.utils.data import DataLoader
import sys

# Add training module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))
from dataloader import AutoPercepDataset

def main():
    parser = argparse.ArgumentParser(description="Test AutoPercep 2.0 Dataloader")
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Path to parent directory containing run folders (e.g., .../data_clean)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    args = parser.parse_args()

    print(f"🔍 Scanning directory: {args.root_dir}")

    try:
        # Initialize Dataset
        dataset = AutoPercepDataset(
            root_dir=args.root_dir,
            img_size=(224, 224),
            pos_scale=10.0,   
            yaw_scale=180.0,  
            max_neighbors=5,
            visibility_threshold=15.0
        )

        print(f"\n✅ Dataset initialized successfully!")
        print(f"   - Total valid samples found: {len(dataset)}")
        
        if len(dataset) == 0:
            print("\n❌ CRITICAL: No valid samples found.")
            print("   - Check that 'records_*.csv' files exist in subfolders.")
            print("   - Check that image subfolders (e.g., '20250821') exist next to CSVs.")
            print("   - Check for NaN values in your CSVs causing full row drops.")
            return

        # Initialize DataLoader
        loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn
        )

        print(f"   - Batch Size: {args.batch_size}")
        print("-" * 40)

        # Iterate through first 3 batches
        for i, batch in enumerate(loader):
            if i >= 3: 
                break

            images, targets, visibility_masks, neighbor_counts = batch
            
            print(f"\n📦 Batch {i+1}:")
            print(f"   - Image Shape: {images.shape} (Expected: [B, 3, 224, 224])")
            print(f"   - Targets Shape: {targets.shape} (Expected: [B, N, 4])")
            print(f"   - Visibility Mask Shape: {visibility_masks.shape} (Expected: [B, N])")
            print(f"   - Neighbor Counts: {neighbor_counts.tolist()}")
            
            # Deep Dive: First Item, First Neighbor
            b_idx = 0 
            n_idx = 0 
            
            if neighbor_counts[b_idx] > 0:
                exists = targets[b_idx, n_idx, 3].item()
                
                if exists > 0.5: 
                    # Denormalize for human reading
                    x_rel = targets[b_idx, n_idx, 0].item() * 10.0 
                    y_rel = targets[b_idx, n_idx, 1].item() * 10.0
                    yaw_rel = targets[b_idx, n_idx, 2].item() * 180.0
                    vis = visibility_masks[b_idx, n_idx].item()
                    
                    print(f"   \n   🤖 Sample Neighbor Data (Denormalized):")
                    print(f"      • X_Rel: {x_rel:.2f}m")
                    print(f"      • Y_Rel: {y_rel:.2f}m")
                    print(f"      • Yaw_Rel: {yaw_rel:.2f}°")
                    print(f"      • Visible (z_visible): {int(vis)}")

                    # Validation Checks
                    if not (-180.0 <= yaw_rel <= 180.0):
                        print(f"   ⚠️  WARNING: Yaw {yaw_rel} is outside [-180, 180] range!")
                    else:
                        print(f"   ✅ Yaw wrapping is correct.")
                        
                    if abs(x_rel) > 50 or abs(y_rel) > 50:
                        print(f"   ⚠️  WARNING: Position seems unusually large ({x_rel}, {y_rel}). Check units.")
            
            # Image Value Check
            img_min = images.min().item()
            img_max = images.max().item()
            if 0.0 <= img_min <= 1.0 and 0.0 <= img_max <= 1.0:
                print(f"   ✅ Image pixel values normalized correctly [{img_min:.2f}, {img_max:.2f}]")
            else:
                print(f"   ⚠️  WARNING: Image values out of [0, 1] range.")

        print("\n🎉 Dataloader test completed successfully!")
        print("   Ready for Phase 2: Network & Loss Overhaul.")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()