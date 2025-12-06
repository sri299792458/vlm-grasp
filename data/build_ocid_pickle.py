import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

# === CONFIGURATION ===
ROOT_DIR = Path("/scratch.global/kanth042/IRS_project/OCID_grasp/") 
SPLITS = {
    "train": "data_split/training_0.txt",
    "val": "data_split/validation_0.txt"
}

def corners_to_grasp(corners):
    """
    Convert 4 corners [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 
    to [cx, cy, w, h, theta, target_idx]
    """
    # Assumption 1: Points are ordered sequentially (P1->P2->P3->P4)
    # Verified by dot product check on sample data.
    
    # 1. Center: Average of all 4 points
    cx, cy = np.mean(corners, axis=0)

    # 2. Dimensions:
    # Assumption 2: P1->P2 is 'Width', P2->P3 is 'Height'
    p1, p2, p3 = corners[0], corners[1], corners[2]
    
    # Vector 1 (Width)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    w = np.linalg.norm([dx, dy])
    
    # Vector 2 (Height)
    h = np.linalg.norm(p3 - p2)

    # 3. Orientation (Theta):
    # Angle of the 'Width' vector relative to X-axis
    theta = np.degrees(np.arctan2(dy, dx))
    
    # Normalize to [-90, 90) convention
    while theta >= 90: theta -= 180
    while theta < -90: theta += 180

    # Assumption 3: Class ID is unknown (0) because master file lacks labels
    target_idx = 0 

    return [cx, cy, w, h, theta, target_idx]

def parse_grasp_file(txt_path):
    """Parses OCID format: 4 corners per grasp."""
    if not txt_path.exists():
        return []

    grasps = []
    with open(txt_path, 'r') as f:
        # Robust parsing: collects all float tokens regardless of newlines
        all_nums = []
        for line in f:
            all_nums.extend(map(float, line.strip().split()))
    
    # Validation: Must be multiple of 8 (4 points * 2 coords)
    if len(all_nums) == 0 or len(all_nums) % 8 != 0:
        return []

    # Reshape: (N_grasps, 4_points, 2_coords)
    points = np.array(all_nums).reshape(-1, 4, 2)

    for corners in points:
        grasps.append(corners_to_grasp(corners))
    
    return np.array(grasps)

def process_split(split_name, txt_file):
    print(f"\nProcessing {split_name} from {txt_file}...")
    samples = []
    
    if not (ROOT_DIR / txt_file).exists():
        print(f"Skipping {split_name}: File not found.")
        return

    with open(ROOT_DIR / txt_file, 'r') as f:
        lines = f.readlines()

    missing_grasps = 0
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        parts = line.split(',')
        if len(parts) != 2: continue
            
        seq_path, filename = parts
        
        # Paths
        img_path = ROOT_DIR / seq_path / 'rgb' / filename
        grasp_filename = filename.replace('.png', '.txt')
        
        # Search for Annotations (Capital A matches your find command)
        candidates = [
            ROOT_DIR / seq_path / 'Annotations' / grasp_filename,
            ROOT_DIR / seq_path / 'annotations' / grasp_filename,
            ROOT_DIR / seq_path / 'grasps' / grasp_filename
        ]
        
        grasp_path = None
        for cand in candidates:
            if cand.exists():
                grasp_path = cand
                break
        
        if grasp_path and img_path.exists():
            try:
                img = np.array(Image.open(img_path).convert('RGB'))
                grasps = parse_grasp_file(grasp_path)
                
                if len(grasps) > 0:
                    samples.append({
                        'image': img,
                        'grasps': grasps,
                        'mask': None,
                        'target': 'unknown',
                        'image_id': f"{seq_path}/{filename}"
                    })
                else:
                    missing_grasps += 1
            except Exception as e:
                print(f"Error: {e}")
        else:
            missing_grasps += 1
            
        if idx % 200 == 0:
            print(f"  Processed {idx}/{len(lines)}...")

    print(f"Finished {split_name}. Total samples: {len(samples)}")
    if missing_grasps > 0:
        print(f"WARNING: {missing_grasps} images skipped.")

    out_path = ROOT_DIR / f"{split_name}.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    for split, txt_file in SPLITS.items():
        process_split(split, txt_file)