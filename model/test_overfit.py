#!/usr/bin/env python3
"""
Overfitting test: train on tiny subset to verify pipeline works.
Should reach ~100% accuracy on 10-100 samples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import TrainingArguments, Trainer
from configs.config import CONFIG
from data.torch_dataset import GraspVLMDataset
from data.grasp_quantizer import GraspQuantizer
from model.qwen3_grasp_model import load_model_and_processor
from train.data_collator import GraspDataCollator


def overfit_test(n_samples: int = 10, n_steps: int = 200):
    """
    Overfit on tiny dataset to verify training works.

    Args:
        n_samples: Number of samples to overfit on
        n_steps: Number of training steps
    """
    config = CONFIG.copy()

    print("="*60)
    print(f"OVERFITTING TEST: {n_samples} samples, {n_steps} steps")
    print("="*60)

    # Load tiny dataset
    quantizer = GraspQuantizer(
        img_width=config['image_size'][1],
        img_height=config['image_size'][0],
        x_bins=config['x_bins'],
        y_bins=config['y_bins'],
        theta_bins=config['theta_bins'],
        w_bins=config['w_bins'],
        h_bins=config['h_bins'],
    )

    full_dataset = GraspVLMDataset(
        config['ocid_grasp_path'],
        split='train',
        config=config,
        quantizer=quantizer
    )

    # Take tiny subset
    tiny_dataset = torch.utils.data.Subset(full_dataset, range(n_samples))

    print(f"Overfitting on {len(tiny_dataset)} samples")

    # Load model
    model, processor = load_model_and_processor(config, use_qlora=False)

    # Data collator
    data_collator = GraspDataCollator(processor=processor)

    # Training args for overfitting
    training_args = TrainingArguments(
        output_dir="./checkpoints/overfit_test",
        max_steps=n_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,  # Higher LR for fast overfitting
        logging_steps=10,
        dataloader_num_workers=8,
        save_steps=1000,  # Don't save checkpoints
        bf16=config['bf16'],
        gradient_checkpointing=config['gradient_checkpointing'],
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tiny_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nTraining...")
    trainer.train()

    # Test inference on training samples
    print("\n" + "="*60)
    print("Testing inference on training samples...")
    print("="*60)

    from inference.predict_grasp import GraspPredictor
    from inference.visualize import draw_grasp_rectangle
    import numpy as np
    from PIL import Image

    # Save model
    trainer.save_model("./checkpoints/overfit_test/final")

    # Load for inference
    model.eval()

    correct = 0
    total = 0

    for i in range(min(n_samples, 5)):
        sample = full_dataset[i]

        # Get image
        messages = sample['messages']
        user_content = messages[1]['content']
        for item in user_content:
            if item['type'] == 'image':
                image = item['image']
                break

        # Get ground truth
        gt_bins = sample['grasp_bins']
        gt_continuous = sample['grasp_continuous']

        # Predict
        from model.constrained_decoding import DigitsOnlyLogitsProcessor, parse_grasp_output
        from data.chat_formatter import format_inference_messages

        inference_msgs = format_inference_messages(np.array(image))
        
        # FIX: Separate Text and Image for the processor
        text_prompt = processor.apply_chat_template(
            inference_msgs, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors='pt'
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False,
                logits_processor=[DigitsOnlyLogitsProcessor(processor.tokenizer)]
            )

        generated_text = processor.decode(outputs[0], skip_special_tokens=True)

        print(f"\nSample {i}:")
        print(f"  Ground truth: {quantizer.bins_to_string(gt_bins)}")
        print(f"  Generated:    {generated_text}")

        # Parse and check
        pred_bins = parse_grasp_output(generated_text)

        if pred_bins is not None:
            # Check if close
            match = (
                abs(pred_bins['x'] - gt_bins['x']) <= 5 and
                abs(pred_bins['y'] - gt_bins['y']) <= 5 and
                abs(pred_bins['theta'] - gt_bins['theta']) <= 5 and
                abs(pred_bins['w'] - gt_bins['w']) <= 5 and
                abs(pred_bins['h'] - gt_bins['h']) <= 5
            )

            if match:
                correct += 1
                print("  ✓ MATCH")
            else:
                print("  ✗ MISMATCH")

        else:
            print("  ✗ INVALID OUTPUT")

        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n\nOverfitting accuracy: {correct}/{total} = {accuracy*100:.1f}%")

    if accuracy >= 0.6:
        print("✓ Overfitting test PASSED (>60% accuracy)")
    else:
        print("✗ Overfitting test FAILED (<60% accuracy)")
        print("  This suggests an issue with the training pipeline.")

    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=200)

    args = parser.parse_args()

    overfit_test(args.n_samples, args.n_steps)
