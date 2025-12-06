#!/usr/bin/env python3
"""
Main training script for VLM grasp prediction.
Research prototype style: simple, hackable, fast iteration.

Usage:
    python train/train_grasp_lora.py [--config configs/config.py]
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader

from configs.config import CONFIG
from data.torch_dataset import GraspVLMDataset
from data.grasp_quantizer import GraspQuantizer
from model.qwen3_grasp_model import load_model_and_processor
from train.data_collator import GraspDataCollator, compute_grasp_metrics


def main(args):
    config = CONFIG.copy()

    # Update from args
    if args.data_dir:
        config['ocid_grasp_path'] = args.data_dir
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir

    print("="*60)
    print("VLM Grasp Training")
    print("="*60)
    print(f"Data dir: {config['ocid_grasp_path']}")
    print(f"Model: {config['model_name']}")
    print(f"Checkpoint dir: {config['checkpoint_dir']}")
    print(f"LoRA rank: {config['lora_r']}")
    print(f"Batch size: {config['train_batch_size']} x {config['gradient_accumulation_steps']} = {config['train_batch_size'] * config['gradient_accumulation_steps']}")
    print("="*60)

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

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GraspVLMDataset(
        config['ocid_grasp_path'],
        split='train',
        config=config,
        quantizer=quantizer
    )

    val_dataset = GraspVLMDataset(
        config['ocid_grasp_path'],
        split='val',
        config=config,
        quantizer=quantizer
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Load model
    print("\nLoading model...")
    # Matches overfit settings (16-bit LoRA)
    model, processor = load_model_and_processor(config, use_qlora=False)

    # Data collator
    data_collator = GraspDataCollator(processor=processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['checkpoint_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler'],
        warmup_steps=config['warmup_steps'],
        logging_dir=f"{config['checkpoint_dir']}/logs",
        logging_steps=config['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        save_strategy="steps",
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=config['bf16'],
        gradient_checkpointing=config['gradient_checkpointing'],
        dataloader_num_workers=8, 
        remove_unused_columns=False,
        
        # --- THE FIX IS HERE ---
        prediction_loss_only=True,  # Critical: Discards logits to prevent 90GB OOM
        # -----------------------
        
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.run_name,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_grasp_metrics,
    )

    # Train!
    print("\nStarting training...")
    print("="*60)

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config['checkpoint_dir'] + "/final")

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to OCID-Grasp dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint output directory')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--run_name', type=str, default='grasp-vlm-lora',
                        help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    args = parser.parse_args()
    main(args)