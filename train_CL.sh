#!/bin/bash
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # One task per GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --time=12:00:00
#SBATCH --job-name="Train_her2st_svg"
#SBATCH --array=1                      # 0: kidney, 1: her2st, etc.
#SBATCH --output=out_files_CL/%x_%A_%a_log.out
#SBATCH --error=out_files_CL/%x_%A_%a_err.out

# === Dataset mapping
datasets=("kidney" "her2st")
out_slide=("NCBI703,NCBI714,NCBI693" "SPA148,SPA136")


DATASET_NAME=${datasets[$SLURM_ARRAY_TASK_ID]}
SLIDE_OUT=${out_slide[$SLURM_ARRAY_TASK_ID]}


echo "Running dataset: $DATASET_NAME"
echo "Slide out: $SLIDE_OUT"
echo "SLURM_JOBID=$SLURM_JOBID | Task ID=$SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"

# === Environment setup
ulimit -s unlimited
module purge
module load conda
conda activate ./conda/envs/hest

# === Paths and config
DATA_PATH=./hest_datasets/${DATASET_NAME}/
RESULTS_DIR=./${DATASET_NAME}_results/CL/runs/


# === Run training
python3 train_CL.py \
    --slide_out "$SLIDE_OUT" \
    --data_path "$DATA_PATH" \
    --expr_name "$DATASET_NAME" \
    --gene_list_filename svg_200genes.txt \
    --batch_size 512 \
    --results_dir "$RESULTS_DIR" \
    --total_epochs 200 \
    --num_aug_ratio 7 \
    --dim 200

echo "✅ Done with $DATASET_NAME!"
