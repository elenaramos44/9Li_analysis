#!/bin/bash
# ==============================================================================
# Automatización de Stage 5 para volúmenes fiduciales (FV_2, FV_3, FV_4)
# ==============================================================================

SUBMIT_SCRIPT="submit_stage5_Gd.sh"

echo "======================================================================"
echo "Lanzando Stage 5 para nuevos Volúmenes Fiduciales (Signal y Background)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Definición de volúmenes fiduciales adicionales
# Formato: "TAG XMIN XMAX YMIN YMAX ZMIN ZMAX"
# ------------------------------------------------------------------------------
FV_CONFIGS=(
    "FV_2 -20.0 20.0 -60.0 -20.0 -130.0 0.0"
    "FV_3 -20.0 20.0  20.0  60.0 -130.0 0.0"
    "FV_4 -20.0 20.0 -20.0  20.0    0.0 130.0"
)

for config in "${FV_CONFIGS[@]}"; do
    # Extraer parámetros de la configuración
    read -r TAG XMIN XMAX YMIN YMAX ZMIN ZMAX <<< "$config"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "Procesando $TAG -> X:[$XMIN, $XMAX], Y:[$YMIN, $YMAX], Z:[$ZMIN, $ZMAX]"
    echo "----------------------------------------------------------------------"

    # Construir el string de argumentos geométricos para merge_and_fv_cut.py
    GEO_ARGS="--fvtag $TAG --xmin $XMIN --xmax $XMAX --ymin $YMIN --ymax $YMAX --zmin $ZMIN --zmax $ZMAX"

    # 1. Enviar muestra SIGNAL
    echo "--> [SIGNAL] Lanzando $TAG..."
    EXTRA_ARGS="$GEO_ARGS" sbatch $SUBMIT_SCRIPT

    # 2. Enviar muestra BACKGROUND
    echo "--> [BACKGROUND] Lanzando $TAG..."
    EXTRA_ARGS="--bkg $GEO_ARGS" sbatch $SUBMIT_SCRIPT

done

echo ""
echo "======================================================================"
echo "¡Todos los volúmenes fiduciales han sido enviados exitosamente a Slurm!"
echo "======================================================================"