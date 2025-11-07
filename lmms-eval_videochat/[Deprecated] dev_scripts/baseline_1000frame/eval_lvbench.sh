TASK=lvbench
MODEL_NAME=videochat_flash
MAX_NUM_FRAMES=1024
# CKPT_PATH=OpenGVLab/VideoChat-Flash-Qwen2-7B_res448
CKPT_PATH=../llava-train_videochat/checkpoints/baseline_1000frame/stage1-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B_stage3_short-long_mix_sft_small.yaml/
BASE_PATH=benchmark_data/model/VideoChat-Flash-Qwen2_5-2B_res448_ver2/
cache_path=${CKPT_PATH}/cache

# Hyperparameters
export EXTRA_PARAM_INNER_CONDENSER_ID="[10, 20]"
export EXTRA_PARAM_INNER_CONDENSER_TYPE="nexthalf"

export EXTRA_PARAM_INNER_STRIDE="4"

export EXTRA_PARAM_FPS="8"
#--------------------

# Eval Setting
export EXTRA_OVERWRITE_MAX_NUM_FRAMES="1024"
export EXTRA_OVERWRITE_MIN_NUM_FRAMES="256"
#--------------------


# 1. 建立 cache_path
mkdir -p "${cache_path}"
# 2. 把 BASE_PATH 里的所有文件（除了 model.safetensors）copy 到 cache_path
for f in "${BASE_PATH}"/*; do
    if [ "$(basename "${f}")" != "model.safetensors" ]; then
        cp -r "${f}" "${cache_path}"
    fi
done
# 3. 把 ckpt 里的 model.safetensors 拷贝进 cache_path
cp "${CKPT_PATH}/model.safetensors" "${cache_path}/"

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8

accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$cache_path,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES} 

rm -r "${cache_path}"