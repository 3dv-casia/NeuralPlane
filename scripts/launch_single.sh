#!/bin/bash

helpFunction_launch_eval()
{
    echo "Launch a single job per gpu."
    echo "Usage: $0 -i <scene id> -t <timestamp> -g gpu_id"
    echo -e "\t-i scene id"
    echo -e "\t-t <timestamp>: if using launch_train_blender.sh will be of format %Y-%m-%d_%H%M%S"
    echo -e "\t-g: gpu id to launch the job on"
    exit 1 # Exit program after printing help
}

while getopts "i:t:g:" opt; do
    case "$opt" in
        i ) scene="$OPTARG" ;;
        t ) timestamp="$OPTARG" ;;
        g ) gpu_id="$OPTARG" ;;
        ? ) helpFunction_launch_eval ;;
    esac
done

if [ -z "$scene" ]; then
    echo "Missing scene id"
    helpFunction_launch_eval
fi
if [ -z "$timestamp" ]; then
    echo "Missing timestamp specification"
    helpFunction_launch_eval
fi
if [ -z "$gpu_id" ]; then
    echo "Missing gpu id"
    helpFunction_launch_eval
fi

echo "scene: $scene on GPU $gpu_id"
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

config_path="configs/${timestamp}/${scene}.yaml"

# pre-procssing
np-pre-seg --config ${config_path} --skipping

# Training
np-train --config ${config_path}

# Exporting
np-export --config ${config_path}

# Evaluation
np-eval --config ${config_path}