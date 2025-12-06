#!/bin/bash
#SBATCH --job-name=qwen_grasp_train
#SBATCH --time=12:00:00          # 12 hours is a safe buffer (expect ~4-6h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # 16 CPU cores for 8 dataloader workers
#SBATCH --mem=64G                # 64GB RAM
#SBATCH --partition=msigpu       # Your partition
#SBATCH --gres=gpu:a100:1        # Requesting 1 A100 GPU
#SBATCH --account=stat8105       # Your account
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# 1. Setup Logging & Environment
mkdir -p logs
module load cuda/12.1.1            
module load conda
source activate IRS              

# 2. Debug Info
echo "Job started on $(hostname) at $(date)"
echo "--------------------"
nvidia-smi
echo "--------------------"

# 3. FAST I/O SETUP (Critical Speed Boost)
# Copy dataset from slow network drive to fast local NVMe ($TMPDIR)
echo "Setting up local data in: $TMPDIR"
mkdir -p $TMPDIR/OCID_grasp

# Copy pickle files (Metadata)
cp /scratch.global/kanth042/IRS_project/OCID_grasp/*.pkl $TMPDIR/OCID_grasp/

# Copy image folders (The heavy lifting)
echo "Copying images..."
cp -r /scratch.global/kanth042/IRS_project/OCID_grasp/ARID* $TMPDIR/OCID_grasp/

echo "Data copy complete. Storage size:"
du -sh $TMPDIR/OCID_grasp

# 4. Run Training
# Note: --data_dir points to the fast local copy ($TMPDIR/OCID_grasp)
# The config file handles batch size (8), workers (8), and reduced epochs (3)
python -u IRS/train/train_grasp_lora.py \
    --data_dir $TMPDIR/OCID_grasp \
    --checkpoint_dir /scratch.global/kanth042/IRS_project/checkpoints/run2 \
    --run_name grasp-vlm-run2

echo "--------------------"
echo "Job finished at $(date)"
