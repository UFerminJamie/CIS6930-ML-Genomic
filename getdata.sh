#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=6:00:00
#SBATCH --job-name="HEST"
#SBATCH --array=0-3     # 0: kidney, 1: her2st, 2: mouse_brain, 3: PRAD
#SBATCH --output=out_files/%x_%A_%a_log.out

# === Dataset mapping
datasets=("kidney" "her2st" "mouse_brain" "PRAD")

DATASET_NAME=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Running dataset: $DATASET_NAME"
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"

ulimit -s unlimited
ml

module load conda
conda activate /blue/pinaki.sarder/j.fermin/conda/envs/hest

# === Run download and preprocess
echo "Starting dataset download..."
python3 dataset_download_hest1k.py \
    --dataset_name $DATASET_NAME \
    --data_path ./hest_datasets

if [ $? -ne 0 ]; then
    echo "dataset_download_hest1k.py failed or it has been downloaded already. Exiting."
    exit 1
fi

echo "Starting dataset preprocessing..."
python3 dataset_preprocess.py \
    --dataset_name $DATASET_NAME \
    --data_path ./hest_datasets \
    --device cuda \
    --top_k 200 \
    --criteria var mean \
    --generate_embedding False

if [ $? -ne 0 ]; then
    echo "dataset_preprocess.py failed. Exiting."
    exit 1
fi

echo "Done with $DATASET_NAME!"