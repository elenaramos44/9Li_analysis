#!/bin/bash
JOBS_DIR="/scratch/elena/9Li/jobs"
SAMPLE_FLAG="--bkg"
ARRAY_RANGE="0-564%10"

# El ID del Stage 3 que SÍ ha entrado en cola y del que depende el Stage 4
JOB_ID_3="5958973"

echo "Enlazando los últimos stages al Stage 3 Job ID: $JOB_ID_3"

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