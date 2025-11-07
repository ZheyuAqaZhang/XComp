if [ $# -lt 2 ]; then
  echo "Usage: $0 <TASK> <MAX_NUM_FRAMES>"
  exit 1
fi

# 使用传入的参数
TASK="$1"
MAX_NUM_FRAMES="$2"

MODEL_NAME=videochat_flash
# CKPT_PATH=OpenGVLab/VideoChat-Flash-Qwen2-7B_res448
CKPT_PATH=../llava-train_videochat/checkpoints/baseline_1000frame_cos/stagesuf-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B_stage3_short-long_mix_sft_mid2.yaml
BASE_PATH=benchmark_data/model/VideoChat-Flash-Qwen2_5-2B_res448_ver2/
cache_path="${CKPT_PATH}/cache"

# Hyperparameters
export EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE="Cosine" 
export EXTRA_PARAM_ALWAYS_SUFFIX="True"
export EXTRA_PARAM_INNER_CONDENSER_ID="[10, 20]"
export EXTRA_PARAM_INNER_CONDENSER_TYPE="nexthalf"
export EXTRA_PARAM_INNER_STRIDE="4"
export EXTRA_PARAM_FPS="8"

# Eval Setting
export EXTRA_OVERWRITE_MAX_NUM_FRAMES="${MAX_NUM_FRAMES}"
export EXTRA_OVERWRITE_MIN_NUM_FRAMES="128"

# 1. 建立 cache_path
mkdir -p "${cache_path}"

# 2. 拷贝 BASE_PATH 中除 model.safetensors 外的所有文件到 cache_path
for f in "${BASE_PATH}"/*; do
    if [ "$(basename "${f}")" != "model.safetensors" ]; then
        cp -r "${f}" "${cache_path}"
    fi
done

# 3. 拷贝 ckpt 中的 model.safetensors 到 cache_path
cp "${CKPT_PATH}/model.safetensors" "${cache_path}/"

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + RANDOM % 100))
NUM_GPUS=1

accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$cache_path,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}

# 清除 cache
yes | sudo rm -r "${cache_path}"
