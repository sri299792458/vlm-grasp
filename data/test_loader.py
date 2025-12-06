#!/usr/bin/env python3
"""
Test data loader and visualize samples.
Quick sanity check before training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from configs.config import CONFIG
from data.torch_dataset import GraspVLMDataset
from data.grasp_quantizer import GraspQuantizer
from inference.visualize import draw_grasp_rectangle


def test_loader():
    """Test data loading and quantization."""
    config = CONFIG.copy()

    # Initialize quantizer
    quantizer = GraspQuantizer(
        img_width=config['image_size'][1],
        img_height=config['image_size'][0],
        x_bins=config['x_bins'],
        y_bins=config['y_bins'],
        theta_bins=config['theta_bins'],
        w_bins=config['w_bins'],
        h_bins=config['h_bins'],
    )

    # Load train dataset
    print("Loading dataset...")
    dataset = GraspVLMDataset(
        config['ocid_grasp_path'],
        split='train',
        config=config,
        quantizer=quantizer
    )

    print(f"Dataset size: {len(dataset)}")

    # Visualize a few samples
    n_vis = min(10, len(dataset))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_vis):
        sample = dataset[i]

        # Extract info
        messages = sample['messages']
        grasp_bins = sample['grasp_bins']
        grasp_continuous = sample['grasp_continuous']

        # Get image from messages
        user_content = messages[1]['content']
        for item in user_content:
            if item['type'] == 'image':
                image = np.array(item['image'])
                break

        # Get assistant response (quantized grasp string)
        grasp_string = messages[2]['content']

        print(f"\nSample {i}:")
        print(f"  Grasp string: {grasp_string}")
        print(f"  Bins: {grasp_bins}")

        # Decode bins back to continuous
        grasp_decoded = quantizer.decode(grasp_bins)

        print(f"  Original: {grasp_continuous}")
        print(f"  Decoded:  {grasp_decoded}")

        # Draw both on image
        img_vis = image.copy()

        # Original in blue
        img_vis = draw_grasp_rectangle(img_vis, grasp_continuous,
                                       color=(0, 0, 255), thickness=2)

        # Decoded in green
        img_vis = draw_grasp_rectangle(img_vis, grasp_decoded,
                                       color=(0, 255, 0), thickness=2)

        axes[i].imshow(img_vis)
        axes[i].set_title(f"Sample {i}\nBlue=orig, Green=quantized")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('IRS/data_loader_test.png', dpi=150, bbox_inches='tight')
    print("\n\nVisualization saved to data_loader_test.png")

    # Test encode-decode round trip
    print("\n" + "="*60)
    print("Testing quantization round-trip...")
    print("="*60)

    errors_cx = []
    errors_cy = []
    errors_theta = []
    errors_w = []
    errors_h = []

    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        original = sample['grasp_continuous']

        bins = quantizer.encode(original)
        decoded = quantizer.decode(bins)

        errors_cx.append(abs(original['cx'] - decoded['cx']))
        errors_cy.append(abs(original['cy'] - decoded['cy']))
        errors_theta.append(abs(original['theta'] - decoded['theta']))
        errors_w.append(abs(original['w'] - decoded['w']))
        errors_h.append(abs(original['h'] - decoded['h']))

    print(f"Mean quantization errors (over {len(errors_cx)} samples):")
    print(f"  Center X: {np.mean(errors_cx):.2f} px")
    print(f"  Center Y: {np.mean(errors_cy):.2f} px")
    print(f"  Angle:    {np.mean(errors_theta):.2f} deg")
    print(f"  Width:    {np.mean(errors_w):.2f} px")
    print(f"  Height:   {np.mean(errors_h):.2f} px")

    print("\nData loader test passed! ✓")


if __name__ == '__main__':
    test_loader()
