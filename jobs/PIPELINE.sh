#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_pipeline
#SBATCH --output=/scratch/elena/9Li/results/log/pipeline_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/pipeline_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "================================================================"
echo "Starting full Li9 analysis pipeline submission on Hyperion"
echo "Time: $(date)"
echo "================================================================"

JOBS_DIR="/scratch/elena/9Li/jobs"

# Configuración adaptativa del rango del array según la muestra
SAMPLE_FLAG=""
ARRAY_RANGE="0-287%10"  # Para SIGNAL: 288 chunks en total, máximo 10 concurrentes
if [[ "$1" == "--bkg" ]]; then
    SAMPLE_FLAG="--bkg"
    ARRAY_RANGE="0-564%10"  # Para BKG: 565 chunks en total, máximo 10 concurrentes
    echo ">> CONFIGURATION SET TO: BACKGROUND (BKG) <<"
else
    echo ">> CONFIGURATION SET TO: SIGNAL (DEFAULT) <<"
fi

echo "Using Array Range: $ARRAY_RANGE"

# ------------------------------------------------------------------------------
# STAGE 1: Submit Load Sliding Windows
# ------------------------------------------------------------------------------
echo "Submitting Stage 1: Load sliding windows"
JOB_OUT_1=$(sbatch --array=${ARRAY_RANGE} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/load_SW_all_files.sh)
JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')
echo "--> Deployed Stage 1 Job ID: $JOB_ID_1"

# ------------------------------------------------------------------------------
# STAGE 2: Submit Multilateration -> Waits for Stage 1
# ------------------------------------------------------------------------------
echo "Submitting Stage 2: Multilateration"
JOB_OUT_2=$(sbatch --array=${ARRAY_RANGE} --dependency=afterok:${JOB_ID_1} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/run_multilat_all.sh)
JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')
echo "--> Deployed Stage 2 Job ID: $JOB_ID_2 (Dependent on $JOB_ID_1)"


# ------------------------------------------------------------------------------
# STAGE 4: Submit Final Refinement -> Waits for Stage 3
# ------------------------------------------------------------------------------
echo "Submitting Stage 4: Vertex reconstruction refinement"
JOB_OUT_4=$(sbatch --array=${ARRAY_RANGE} --dependency=afterok:${JOB_ID_3} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/submit_refinement.sh)
JOB_ID_4=$(echo "$JOB_OUT_4" | awk '{print $4}')
echo "--> Deployed Stage 4 Job ID: $JOB_ID_4 (Dependent on $JOB_ID_3)"

# ------------------------------------------------------------------------------
# STAGE 5: Submit Final Fiducial Volume Cut & Merge -> Waits for Stage 4
# ------------------------------------------------------------------------------
echo "Submitting Stage 5: Final cut --> FV"
# Añadimos --array=0-11 aquí también
JOB_OUT_5=$(sbatch --array=0-11 --dependency=afterok:${JOB_ID_4} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/fv_cut.sh)
JOB_ID_5=$(echo "$JOB_OUT_5" | awk '{print $4}')
echo "--> Deployed Stage 5 Job ID: $JOB_ID_5 (Dependent on $JOB_ID_4)"

echo "================================================================"
echo "All job arrays successfully registered in the Slurm Controller!"
echo "Master pipeline script execution completed."
echo "================================================================"