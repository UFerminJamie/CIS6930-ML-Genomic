#!/bin/bash
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1                      # One task per GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --time=1:00:00
#SBATCH --job-name="eval_her2st_svg"
#SBATCH --array=1                      # 0: kidney, 1: her2st, etc.
#SBATCH --output=out_files/%x_%A_%a_log.out
#SBATCH --error=out_files/%x_%A_%a_err.out

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
# var_200genes.txt, deg_200genes.txt
GENE=svg_200genes.txt 
RUN=027

# === Run training
python3 CL/eval.py \
    --slide_out "$SLIDE_OUT" \
    --expr_name "$DATASET_NAME" \
    --gene_list_filename $GENE\
    --run "$RUN" \

echo "✅ Done with $DATASET_NAME!"
