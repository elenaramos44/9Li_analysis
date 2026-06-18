#!/bin/bash
JOBS_DIR="/scratch/elena/9Li/jobs"
SAMPLE_FLAG="--bkg"
ARRAY_RANGE="0-564%10"

# 1. El ID del Stage 1 que ya está corriendo en tu clúster
JOB_ID_1="5957773"

echo "Enlazando el resto del pipeline al Job ID: $JOB_ID_1"

# ------------------------------------------------------------------------------
# STAGE 2: Multilateration -> Espera al 1
# ------------------------------------------------------------------------------
echo "Submitting Stage 2..."
JOB_OUT_2=$(sbatch --array=${ARRAY_RANGE} --dependency=afterok:${JOB_ID_1} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/run_multilat_all.sh)
JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')
echo "--> Deployed Stage 2 Job ID: $JOB_ID_2"

# ------------------------------------------------------------------------------
# STAGE 3: PKL Normalization -> Espera al 2
# ------------------------------------------------------------------------------
echo "Submitting Stage 3..."
JOB_OUT_3=$(sbatch --dependency=afterok:${JOB_ID_2} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/procesado_csv_multilat.sh)
JOB_ID_3=$(echo "$JOB_OUT_3" | awk '{print $4}')
echo "--> Deployed Stage 3 Job ID: $JOB_ID_3"

# ------------------------------------------------------------------------------
# STAGE 4: Vertex refinement -> Espera al 3
# ------------------------------------------------------------------------------
echo "Submitting Stage 4..."
JOB_OUT_4=$(sbatch --array=${ARRAY_RANGE} --dependency=afterok:${JOB_ID_3} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/submit_refinement.sh)
JOB_ID_4=$(echo "$JOB_OUT_4" | awk '{print $4}')
echo "--> Deployed Stage 4 Job ID: $JOB_ID_4"

# ------------------------------------------------------------------------------
# STAGE 5: Final cut -> Espera al 4
# ------------------------------------------------------------------------------
echo "Submitting Stage 5..."
JOB_OUT_5=$(sbatch --dependency=afterok:${JOB_ID_4} --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" ${JOBS_DIR}/fv_cut.sh)
JOB_ID_5=$(echo "$JOB_OUT_5" | awk '{print $4}')
echo "--> Deployed Stage 5 Job ID: $JOB_ID_5"