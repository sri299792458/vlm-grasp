"""
OCID-VLG dataset loader.
Language-guided grasp detection with referring expressions.

Download from: https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view
"""
import os
import json
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class OCIDVLGDataset:
    """
    OCID-VLG: Vision-Language-Grasping dataset.
    
    Each sample contains:
        - img: (480, 640, 3) RGB image
        - sentence: Referring expression (e.g., "the red apple on the left")
        - grasps: (N, 6) array of grasps FOR THE REFERRED OBJECT ONLY
        - mask: (480, 640) segmentation mask of target object
        - target: Object class name
        - target_idx: Object class index
    
    Versions:
        - 'multiple': Multiple referring expressions per target object
        - 'unique': One referring expression per target object  
        - 'novel-instances': Test on unseen object instances
        - 'novel-classes': Test on unseen object classes
    """
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 version: str = 'multiple',
                 transform_img=None,
                 transform_grasp=None):
        """
        Args:
            root_dir: Path to OCID-VLG dataset root
            split: 'train', 'val', or 'test'
            version: 'multiple', 'unique', 'novel-instances', or 'novel-classes'
            transform_img: Optional transform for images
            transform_grasp: Optional transform for grasps (None = return raw grasps)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.version = version
        self.transform_img = transform_img
        self.transform_grasp = transform_grasp
        
        # Load annotations
        self.samples = self._load_annotations()
        print(f"OCID-VLG [{version}/{split}]: Loaded {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON files."""
        # 1. Try User's Structure: root/refer/version/train_expressions.json
        p1 = self.root_dir / "refer" / self.version / f"{self.split}_expressions.json"
        
        # 2. Try Standard Structure: root/version/split.json
        p2 = self.root_dir / self.version / f"{self.split}.json"
        
        # 3. Try Alternative: root/version_split.json
        p3 = self.root_dir / f"{self.version}_{self.split}.json"

        # Check existing paths
        final_path = None
        for p in [p1, p2, p3]:
            if p.exists():
                final_path = p
                print(f"Loading annotations from: {final_path}")
                break
        
        if final_path is None:
            raise FileNotFoundError(
                f"Could not find annotations. Searched:\n"
                f"1. {p1}\n2. {p2}\n3. {p3}\n"
                f"Root: {self.root_dir}"
            )
        
        with open(final_path, 'r') as f:
            raw_data = json.load(f)
            
        # Handle "data" wrapper (Common in referring expression datasets)
        if isinstance(raw_data, dict) and 'data' in raw_data:
            print("Found 'data' wrapper in JSON, unwrapping...")
            annotations = raw_data['data']
        else:
            annotations = raw_data
        
        return annotations
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load RGB image."""
        # FIX: The JSON uses "seq06,result..." but filesystem is "seq06/result..."
        # We need to handle this discrepancy.
        img_path_fixed = img_path.replace(',', '/')
        
        full_path = self.root_dir / img_path_fixed
        if not full_path.exists():
            # Fallback: maybe it really is a comma? (Unlikely given the debug output)
            full_path_original = self.root_dir / img_path
            if full_path_original.exists():
                full_path = full_path_original
            else:
                # Try finding it in 'images' subdir
                alt_path = self.root_dir / "images" / img_path_fixed
                if alt_path.exists():
                    full_path = alt_path
                else:
                    # FIX: OCID structure often puts images in 'rgb' folder
                    # e.g., seq06/rgb/result_...
                    # We need to insert 'rgb' before the filename
                    parent_dir = (self.root_dir / img_path_fixed).parent
                    filename = (self.root_dir / img_path_fixed).name
                    rgb_path = parent_dir / "rgb" / filename
                    
                    if rgb_path.exists():
                        full_path = rgb_path
                    else:
                        # DEBUGGING BLOCK: Deep Path Walker 
                        # (Keep this just in case, but use the fixed path)
                        print(f"\n[ERROR] Image not found: {full_path}")
                        pass # The rest of the debug block can stay or execute on failure
        
        # If we reached here without existing, let the debug block below run or crash
        
        if not full_path.exists():
                # DEBUGGING BLOCK: Deep Path Walker
                print(f"\n[ERROR] Image not found: {full_path}")
                print(f"  Root: {self.root_dir.resolve()}")
                
                # Check path components using the FIXED path
                parts = Path(img_path_fixed).parts
                current = self.root_dir
                for i, part in enumerate(parts):
                    next_level = current / part
                    if not next_level.exists():
                        print(f"  [BREAKPOINT] Found '{current.name}' but could not find '{part}' inside it.")
                        # Check ignore case
                        candidates = list(current.glob(part + '*'))
                        if candidates:
                            print(f"  Did you mean? {[c.name for c in candidates]}")
                        else:
                            # List sibling dirs
                            try:
                                siblings = [p.name for p in current.iterdir()]
                                print(f"  Contents of '{current.name}': {siblings[:10]} ... ({len(siblings)} items)")
                            except Exception as e:
                                print(f"  Could not list contents: {e}")
                        break
                    
                    # If this is the last part (the file) and it doesn't exist (but folder does)
                    if i == len(parts) - 1 and not next_level.exists():
                         print(f"  [BREAKPOINT] File '{part}' missing in '{current.name}'.")
                         siblings = [p.name for p in current.iterdir()]
                         print(f"  Folder contents: {siblings[:10]}")
                         
                    current = next_level


        img = Image.open(full_path).convert('RGB')
        return np.array(img)
    
    def _load_depth(self, depth_path: str) -> np.ndarray:
        """Load depth image."""
        full_path = self.root_dir / depth_path
        depth = np.load(full_path) if depth_path.endswith('.npy') else np.array(Image.open(full_path))
        return depth.astype(np.float32)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load segmentation mask."""
        full_path = self.root_dir / mask_path
        mask = np.array(Image.open(full_path))
        return mask.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with:
                'image': (480, 640, 3) RGB image
                'sentence': str, referring expression
                'grasps': list of grasp dicts [{cx, cy, w, h, theta, target_idx}, ...]
                'mask': (480, 640) segmentation mask (or None)
                'target': str, object class name
                'target_idx': int, object class index
                'bbox': (4,) bounding box [x1, y1, x2, y2]
                'image_id': str, unique identifier
        """
        sample = self.samples[idx]
        
        # Load image (Support 'image_filename' from new schema)
        if 'img_path' in sample:
            image = self._load_image(sample['img_path'])
        elif 'image_filename' in sample:
            image = self._load_image(sample['image_filename'])
        elif 'img' in sample:
            image = np.array(sample['img'])
        else:
            raise KeyError(f"No image path found in sample keys: {sample.keys()}")
        
        # Apply image transform if provided
        if self.transform_img is not None:
            image = self.transform_img(image)
        
        # Get referring expression (Support 'question' from new schema)
        sentence = sample.get('sentence', sample.get('question', ''))
        
        # Get grasps for this specific target object
        grasps_raw = sample.get('grasps', [])
        
        # Convert to list of dicts
        grasps = []
        
        for g in grasps_raw:
            # Handle 4-point polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # This is common in Cornell/OCID datasets
            if isinstance(g, list) and len(g) == 4 and isinstance(g[0], list):
                g = np.array(g)
                
                # Center
                center = g.mean(axis=0)
                cx, cy = center[0], center[1]
                
                # Edges
                # Edge 0-1
                dx1 = g[1][0] - g[0][0]
                dy1 = g[1][1] - g[0][1]
                len1 = np.sqrt(dx1**2 + dy1**2)
                
                # Edge 1-2
                dx2 = g[2][0] - g[1][0]
                dy2 = g[2][1] - g[1][1]
                len2 = np.sqrt(dx2**2 + dy2**2)
                
                # Convention: w is the longer side? Or w is opening?
                # Usually standard is: w = length along gripper axis, h = opening usually fixed or variable?
                # Let's map len1 -> w, len2 -> h (arbitrary but consistent)
                # And theta is angle of len1
                w, h = len1, len2
                angle = np.degrees(np.arctan2(dy1, dx1)) # FIX: variable name was angle_deg
                
                target_idx = 0 # Default if not provided
                
            # Handle 5/6-tuple format: [cx, cy, w, h, angle, (id)]
            elif len(g) >= 5:
                # Ensure it's flat list/array, not nested
                if isinstance(g[0], list): 
                     # Should not happen if logic above caught 4-points, but safety check
                     continue
                     
                cx, cy, w, h, angle = g[:5]
                target_idx = g[5] if len(g) > 5 else 0
            
            else:
                continue # Unknown format

            # Normalize angle to [-90, 90)
            angle = ((angle + 90) % 180) - 90
            
            grasps.append({
                'cx': float(cx),
                'cy': float(cy),
                'w': float(w),
                'h': float(h),
                'theta': float(angle),
                'target_idx': int(target_idx),
            })
        
        # Load mask if available
        mask = None
        if 'mask_path' in sample:
            mask = self._load_mask(sample['mask_path'])
        elif 'mask' in sample and sample['mask'] is not None:
            mask = np.array(sample['mask'])
        
        # Get other metadata
        target = sample.get('target', 'unknown')
        target_idx = sample.get('target_idx', 0)
        bbox = sample.get('bbox', None)
        if bbox is not None:
            bbox = np.array(bbox)
        
        return {
            'image': image,
            'sentence': sentence,
            'grasps': grasps,
            'mask': mask,
            'target': target,
            'target_idx': target_idx,
            'bbox': bbox,
            'image_id': sample.get('image_id', f"sample_{idx}"),
        }


class OCIDVLGPickleDataset:
    """
    Alternative loader if OCID-VLG is stored as pickle files.
    This matches the format from the official dataset.py in the repo.
    """
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 version: str = 'multiple'):
        """
        Args:
            root_dir: Path to OCID-VLG root (where the .pkl files are)
            split: 'train', 'val', or 'test'
            version: 'multiple', 'unique', 'novel-instances', 'novel-classes'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.version = version
        
        # Load the pickle file
        self.samples = self._load_pickle()
        print(f"OCID-VLG [{version}/{split}]: Loaded {len(self.samples)} samples")
    
    def _load_pickle(self) -> List[Dict]:
        """Load from pickle file."""
        # Try different naming conventions
        possible_paths = [
            self.root_dir / self.version / f"{self.split}.pkl",
            self.root_dir / f"{self.version}_{self.split}.pkl",
            self.root_dir / f"{self.split}.pkl",
        ]
        
        for pkl_path in possible_paths:
            if pkl_path.exists():
                print(f"Loading from {pkl_path}")
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
        
        raise FileNotFoundError(
            f"Could not find pickle file. Tried:\n" +
            "\n".join(str(p) for p in possible_paths)
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns sample in standardized format.
        
        Official OCID-VLG format:
            'img': (480, 640, 3), np.uint8
            'depth': (480, 640), np.float32
            'sentence': str
            'target': str
            'target_idx': int
            'bbox': (4,), np.int16
            'mask': (480, 640), np.float32
            'grasps': (N, 6), np.float32
            'grasp_masks': dict with pos, qua, ang, wid
        """
        sample = self.samples[idx]
        
        # Get image
        image = sample['img']
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Get sentence (referring expression)
        sentence = sample.get('sentence', '')
        
        # Convert grasps to list of dicts
        grasps_raw = sample.get('grasps', np.array([]))
        grasps = []
        for i in range(len(grasps_raw)):
            cx, cy, w, h, angle = grasps_raw[i][:5]
            target_idx = grasps_raw[i][5] if len(grasps_raw[i]) > 5 else 0
            
            # Normalize angle to [-90, 90)
            angle = ((angle + 90) % 180) - 90
            
            grasps.append({
                'cx': float(cx),
                'cy': float(cy),
                'w': float(w),
                'h': float(h),
                'theta': float(angle),
                'target_idx': int(target_idx),
            })
        
        return {
            'image': image,
            'sentence': sentence,
            'grasps': grasps,
            'mask': sample.get('mask', None),
            'target': sample.get('target', 'unknown'),
            'target_idx': sample.get('target_idx', 0),
            'bbox': sample.get('bbox', None),
            'image_id': f"{self.split}_{idx}",
        }


def inspect_ocid_vlg(root_dir: str, version: str = 'multiple', n_samples: int = 5):
    """
    Quick inspection of OCID-VLG dataset.
    """
    print("="*60)
    print("OCID-VLG Dataset Inspection")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = OCIDVLGPickleDataset(root_dir, split=split, version=version)
            print(f"\n{split}: {len(dataset)} samples")
            
            # Show a few examples
            for i in range(min(n_samples, len(dataset))):
                sample = dataset[i]
                print(f"\n  Sample {i}:")
                print(f"    Sentence: '{sample['sentence']}'")
                print(f"    Target: {sample['target']}")
                print(f"    Num grasps: {len(sample['grasps'])}")
                if sample['grasps']:
                    g = sample['grasps'][0]
                    print(f"    First grasp: cx={g['cx']:.1f}, cy={g['cy']:.1f}, θ={g['theta']:.1f}°")
        
        except Exception as e:
            print(f"\n{split}: Error - {e}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to OCID-VLG dataset')
    parser.add_argument('--version', type=str, default='multiple',
                        choices=['multiple', 'unique', 'novel-instances', 'novel-classes'])
    
    args = parser.parse_args()
    inspect_ocid_vlg(args.root, args.version)
