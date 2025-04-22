#!/bin/bash

helpFunction_launch_eval()
{
   echo "Usage: $0 -m <data split> -t <timestamp> [-s] [<gpu_list>]"
   echo -e "\t-m data split file: 8 scenes or 100 scenes according to Airplane"
   echo -e "\t-t <timestamp>: if using launch_train_blender.sh will be of format %Y-%m-%d_%H%M%S"
   echo -e "\t-s: Launch a single evaluation job per gpu."
   echo -e "\t<gpu_list> [OPTIONAL] list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

single=false
while getopts "m:t:s" opt; do
    case "$opt" in
        m ) data_split="$OPTARG" ;;
        t ) timestamp="$OPTARG" ;;
        s ) single=true ;;
        ? ) helpFunction_launch_eval ;;
    esac
done

if [ -z "$data_split" ]; then
    echo "Missing method name"
    helpFunction_launch_eval
fi

if [ -z "$timestamp" ]; then
    echo "Missing timestamp specification"
    helpFunction_launch_eval
fi

shift $((OPTIND-1))

# Deal with gpu's. If passed in, use those.
GPU_IDX=("$@")
if [ -z "${GPU_IDX[0]+x}" ]; then
    echo "no gpus set... finding available gpus"
    # Find available devices
    num_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    START=0
    END=${num_device}-1
    GPU_IDX=()

    for (( id=START; id<=END; id++ )); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo '[0-9]+')
        if [[ $free_mem -gt 10000 ]]; then
            GPU_IDX+=( "$id" )
        fi
    done
fi
echo "available gpus... ${GPU_IDX[*]}"

SCENES=($(cat $data_split))
len=${#GPU_IDX[@]}
GPU_PID=()

mkdir -p "logs/${timestamp}"

# kill all the background jobs if terminated:
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for scene in "${SCENES[@]}"; do
    if "$single" && [ -n "${GPU_PID[$idx]+x}" ]; then
        wait "${GPU_PID[$idx]}"
    fi
    export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}

    config_path="configs/${timestamp}/${scene}.yaml"

    bash scripts/launch_single.sh -i $scene -t $timestamp -g ${GPU_IDX[$idx]} > "logs/${timestamp}/${scene}.txt" 2>&1 & GPU_PID[$idx]=$!

    echo "Launched ${config_path} on gpu ${GPU_IDX[$idx]}"

    # update gpu
    ((idx=(idx+1)%len))
done
wait
echo "Done."