import os
import glob
import pandas as pd

def check_raw_coordinates(root_dir):
    print(f"🔍 Scanning CSVs in: {root_dir}")
    csv_files = glob.glob(os.path.join(root_dir, "**", "records_*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ No CSV files found!")
        return

    for csv_path in csv_files[:3]: # Check first 3 files only
        print(f"\n📄 Checking: {os.path.basename(csv_path)}")
        try:
            df = pd.read_csv(csv_path)
            
            # Define columns to check
            cols = ['self_pose_x', 'self_pose_y', 'car_S1_pose_x', 'car_S1_pose_y']
            available_cols = [c for c in cols if c in df.columns]
            
            if not available_cols:
                print("   ⚠️ Missing expected coordinate columns.")
                continue

            # Show first 3 valid rows
            valid_df = df.dropna(subset=available_cols).head(3)
            
            if len(valid_df) == 0:
                print("   ⚠️ No valid rows with coordinates found.")
                continue

            print("   --- Sample Raw Values (First 3 rows) ---")
            print(valid_df[available_cols].to_string(index=False))
            
            # Check ranges
            min_vals = valid_df[available_cols].min()
            max_vals = valid_df[available_cols].max()
            
            print(f"\n   📊 Min values: \n{min_vals}")
            print(f"   📊 Max values: \n{max_vals}")
            
            range_span = max_vals - min_vals
            if range_span.max() > 100:
                print("\n   ⚠️ WARNING: Coordinate span is > 100 units. These are likely Global Coordinates (e.g., UTM/MoCap World Frame).")
                print("   💡 FIX NEEDED: You must subtract a global offset (e.g., the minimum X/Y of the whole dataset) to make them local.")
            else:
                print("\n   ✅ Coordinates look like local meters.")

        except Exception as e:
            print(f"   ❌ Error reading {csv_path}: {e}")

if __name__ == "__main__":
    # Point to your data parent folder
    import sys
    if len(sys.argv) > 1:
        check_raw_coordinates(sys.argv[1])
    else:
        # Default path from your previous command
        check_raw_coordinates(r"C:\University\SS26\Project\data")