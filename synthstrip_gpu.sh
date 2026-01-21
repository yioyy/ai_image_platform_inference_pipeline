#!/usr/bin/env bash

# ============================================================
#   SynthStrip Docker Wrapper (GPU version)
#   Usage:
#       ./synthstrip_gpu.sh input.nii.gz output.nii.gz
#   Requirements:
#       - NVIDIA GPU
#       - nvidia-container-toolkit installed
# ============================================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_image> <output_image>"
    exit 1
fi

INPUT=$1
OUTPUT=$2

# Convert to absolute paths
INPUT_ABS=$(readlink -f "$INPUT")
OUTPUT_ABS=$(readlink -f "$OUTPUT")
WORKDIR=$(dirname "$INPUT_ABS")

echo "=== SynthStrip GPU Mode ==="
echo "Input : $INPUT_ABS"
echo "Output: $OUTPUT_ABS"
echo "Workdir: $WORKDIR"
echo "============================"

docker run --rm \
    --gpus all \
    -v "$WORKDIR":/data \
    freesurfer/synthstrip \
    -i /data/"$(basename "$INPUT_ABS")" \
    -o /data/"$(basename "$OUTPUT_ABS")"

