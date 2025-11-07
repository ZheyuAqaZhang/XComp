TASK=lvbench
MODEL_NAME=videochat_flash
MAX_NUM_FRAMES=64
# CKPT_PATH=OpenGVLab/VideoChat-Flash-Qwen2-7B_res448
CKPT_PATH=benchmark_data/model/VideoChat-Flash-Qwen2_5-2B_res448
BASE_PATH=benchmark_data/model/VideoChat-Flash-Qwen2_5-2B_res448_ver2/
cache_path=${CKPT_PATH}/cache1

# Hyperparameters
export EXTRA_PARAM_INNER_CONDENSER_ID="[10, 20]"
export EXTRA_PARAM_INNER_CONDENSER_TYPE="nexthalf"

export EXTRA_PARAM_INNER_GROUPS="[1, 1, 1]"

export EXTRA_PARAM_INNER_STRIDE="1"

export EXTRA_PARAM_FPS="8"
#--------------------

# Eval Setting
export EXTRA_OVERWRITE_MAX_NUM_FRAMES="64"
export EXTRA_OVERWRITE_MIN_NUM_FRAMES="32"
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
MASTER_PORT=$((18000 + $RANDOM % 1000))
NUM_GPUS=4

# set LOGDIR to current directory, with current file name but replace .sh with .log
LOGDIR=$(pwd)/$(basename $0 .sh).log

accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$cache_path,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES} >"${LOGDIR}" 2>&1

rm -r "${cache_path}"