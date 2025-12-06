"""
OCID-Grasp dataset loader.
Reads from pickle/HDF5 and converts to our format.
"""
import os
import pickle
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class OCIDGraspDataset:
    """
    Load OCID-Grasp annotations.

    Expected format (from stefan-ainetter/grasp_det_seg_cnn):
        - grasps: (N, 6) array with [center_x, center_y, width, height, angle, target_idx]
        - img: (480, 640, 3) RGB image
        - mask: (480, 640) segmentation mask
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir: Path to OCID-Grasp dataset
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Load data
        self.samples = self._load_data()
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_data(self) -> List[Dict]:
        """
        Load dataset from disk.

        Try multiple formats:
        1. Pickle files (*.pkl)
        2. Individual images + annotations
        3. HDF5 files
        """
        samples = []

        # Option 1: Load from pickle (common format)
        pkl_path = self.data_dir / f"{self.split}.pkl"
        if pkl_path.exists():
            print(f"Loading from {pkl_path}")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Convert to our format
            for item in data:
                samples.append(self._convert_sample(item))
            return samples

        # Option 2: Load from individual files
        img_dir = self.data_dir / self.split / 'rgb'
        ann_dir = self.data_dir / self.split / 'annotations'

        if img_dir.exists() and ann_dir.exists():
            print(f"Loading from {img_dir}")
            for img_path in sorted(img_dir.glob('*.png')):
                ann_path = ann_dir / f"{img_path.stem}.pkl"
                if not ann_path.exists():
                    continue

                # Load image
                img = np.array(Image.open(img_path).convert('RGB'))

                # Load annotations
                with open(ann_path, 'rb') as f:
                    ann = pickle.load(f)

                sample = {
                    'image': img,
                    'grasps': ann.get('grasps', []),
                    'mask': ann.get('mask', None),
                    'target': ann.get('target', 'unknown'),
                    'image_id': img_path.stem,
                }
                samples.append(sample)
            return samples

        # Option 3: Try to auto-detect format
        print(f"Warning: Could not find data in {self.data_dir}")
        print("Please organize data as:")
        print("  - {split}.pkl, or")
        print("  - {split}/rgb/*.png + {split}/annotations/*.pkl")

        return samples

    def _convert_sample(self, item: Dict) -> Dict:
        """Convert raw dataset item to our standardized format."""
        # Extract image
        if 'img' in item:
            img = item['img']
        elif 'image' in item:
            img = item['image']
        else:
            raise KeyError("No 'img' or 'image' key in sample")

        # Ensure RGB
        if img.dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)

        # Extract grasps: (N, 6) with [cx, cy, w, h, angle, target_idx]
        grasps_raw = item.get('grasps', np.array([]))

        # Convert to list of dicts
        grasps = []
        for i in range(len(grasps_raw)):
            cx, cy, w, h, angle, target_idx = grasps_raw[i]

            # OCID-Grasp uses angle in degrees, sometimes in different conventions
            # Standardize to [-90, 90)
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
            'image': img,
            'grasps': grasps,
            'mask': item.get('mask', None),
            'target': item.get('target', 'unknown'),
            'sentence': item.get('sentence', ''),  # For OCID-VLG
            'image_id': item.get('image_id', 'img_unknown'),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def split_dataset(data_dir: str, train_ratio: float = 0.8,
                  val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test if not already split.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    from sklearn.model_selection import train_test_split

    # Load all samples
    all_samples_path = Path(data_dir) / 'all_samples.pkl'

    if not all_samples_path.exists():
        # Try to find samples
        pkl_files = list(Path(data_dir).glob('*.pkl'))
        if len(pkl_files) == 1 and pkl_files[0].name != 'all_samples.pkl':
            all_samples_path = pkl_files[0]
        else:
            raise FileNotFoundError(f"Could not find dataset at {data_dir}")

    with open(all_samples_path, 'rb') as f:
        all_samples = pickle.load(f)

    # Split
    train_val, test = train_test_split(
        all_samples, test_size=(1 - train_ratio - val_ratio), random_state=42
    )

    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=42
    )

    print(f"Split dataset: {len(train)} train, {len(val)} val, {len(test)} test")

    # Save splits
    output_dir = Path(data_dir)
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(output_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val, f)
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test, f)

    return train, val, test
