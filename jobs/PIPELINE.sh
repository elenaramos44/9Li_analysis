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

# ------------------------------------------------------------------------------
# STAGE 1: Submit Load Sliding Windows (Array 0-845)
# ------------------------------------------------------------------------------
echo "Submitting Stage 1: Load sliding windows"
JOB_OUT_1=$(sbatch ${JOBS_DIR}/load_SW_all_files.sh)
JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')
echo "--> Deployed Stage 1 Job ID: $JOB_ID_1"

# ------------------------------------------------------------------------------
# STAGE 2: Submit Multilateration (Array 0-845) -> Waits for Stage 1
# ------------------------------------------------------------------------------
echo "Submitting Stage 2: Multilateration"
JOB_OUT_2=$(sbatch --dependency=afterok:${JOB_ID_1} ${JOBS_DIR}/run_multilat_all.sh)
JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')
echo "--> Deployed Stage 2 Job ID: $JOB_ID_2 (Dependent on $JOB_ID_1)"

# ------------------------------------------------------------------------------
# STAGE 3: Submit CSV Processing (Array 0-11) -> Waits for Stage 2
# ------------------------------------------------------------------------------
echo "Submitting Stage 3: CSV Processing to pkl"
JOB_OUT_3=$(sbatch --dependency=afterok:${JOB_ID_2} ${JOBS_DIR}/procesado_csv_multilat.sh)
JOB_ID_3=$(echo "$JOB_OUT_3" | awk '{print $4}')
echo "--> Deployed Stage 3 Job ID: $JOB_ID_3 (Dependent on $JOB_ID_2)"

# ------------------------------------------------------------------------------
# STAGE 4: Submit Final Refinement (Array 0-845) -> Waits for Stage 3
# ------------------------------------------------------------------------------
echo "Submitting Stage 4: Vertex reconstruction refinement"
JOB_OUT_4=$(sbatch --dependency=afterok:${JOB_ID_3} ${JOBS_DIR}/submit_refinement.sh)
JOB_ID_4=$(echo "$JOB_OUT_4" | awk '{print $4}')
echo "--> Deployed Stage 4 Job ID: $JOB_ID_4 (Dependent on $JOB_ID_3)"

# ------------------------------------------------------------------------------
# STAGE 5: Submit Final Fiducial Volume Cut & Merge (Array 0-11) -> Waits for Stage 4
# ------------------------------------------------------------------------------
echo "Submitting Stage 5: Final cut --> FV"
JOB_OUT_5=$(sbatch --dependency=afterok:${JOB_ID_4} ${JOBS_DIR}/fv_cut.sh)
JOB_ID_5=$(echo "$JOB_OUT_5" | awk '{print $4}')
echo "--> Deployed Stage 5 Job ID: $JOB_ID_5 (Dependent on $JOB_ID_4)"

echo "================================================================"
echo "All job arrays successfully registered in the Slurm Controller!"
echo "Master pipeline script execution completed."
echo "================================================================"