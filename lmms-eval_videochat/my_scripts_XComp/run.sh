#!/bin/bash

start_time=$(date +%s)

# 模型参数（可以从外部传入，否则使用默认值）
TASK=${TASK:-lvbench}
MODEL_NAME=${MODEL_NAME:-videochat_flash}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-512}
CKPT_PATH=${CKPT_PATH:-../llava-train_videochat/checkpoints/baseline_1000frame_cos/stagesuf-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B_stage3_short-long_mix_sft_mid2.yaml}
BASE_PATH=${BASE_PATH:-benchmark_data/model/VideoChat-Flash-Qwen2_5-2B_res448_ver7}
cache_path=${cache_path:-${CKPT_PATH}/cache7}

# 超参数（可通过外部环境变量传入）
export EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE=${EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE:-"Cosine"}
export EXTRA_PARAM_ALWAYS_SUFFIX=${EXTRA_PARAM_ALWAYS_SUFFIX:-"True"}
# export EXTRA_PARAM_ATTENTION_DROP=${EXTRA_PARAM_ATTENTION_DROP:-"0.5"}
export EXTRA_NEED_SCORE=${EXTRA_NEED_SCORE:-"True"}
export EXTRA_PARAM_SELECT_AND_DROP=${EXTRA_PARAM_SELECT_AND_DROP:-"2048"}
export EXTRA_PARAM_DUPLICATE=${EXTRA_PARAM_DUPLICATE:-"8"}

export EXTRA_PARAM_SCORE_TYPE=${EXTRA_PARAM_SCORE_TYPE:-"segments_8_4"}
export EXTRA_PARAM_QUERY_BIAS=${EXTRA_PARAM_QUERY_BIAS:-"27"}

export EXTRA_PARAM_INNER_CONDENSER_ID=${EXTRA_PARAM_INNER_CONDENSER_ID:-"[10, 20]"}
export EXTRA_PARAM_INNER_CONDENSER_TYPE=${EXTRA_PARAM_INNER_CONDENSER_TYPE:-"nexthalf"}

export EXTRA_PARAM_INNER_STRIDE=${EXTRA_PARAM_INNER_STRIDE:-"4"}
export EXTRA_PARAM_FPS=${EXTRA_PARAM_FPS:-"8"}

# Eval 设置
export EXTRA_OVERWRITE_MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-"512"}
export EXTRA_OVERWRITE_MIN_NUM_FRAMES=${EXTRA_OVERWRITE_MIN_NUM_FRAMES:-"128"}

# 建立 cache_path
mkdir -p "${cache_path}"

# 拷贝除 model.safetensors 外的所有文件
for f in "${BASE_PATH}"/*; do
    if [ "$(basename "${f}")" != "model.safetensors" ]; then
        cp -r "${f}" "${cache_path}"
    fi
done

# 拷贝 model.safetensors 到 cache_path
cp "${CKPT_PATH}/model.safetensors" "${cache_path}/"

# 显示任务名称
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

# Job 参数
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 1000))
NUM_GPUS=${NUM_GPUS:-4}

# 启动评估

# 如果有环境变量 LIMIT，则执行
if [ -n "$LIMIT" ]; then
accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$cache_path,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES} \
    --limit $LIMIT
else
# 否则执行
accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$cache_path,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}
fi

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "脚本执行时间: ${elapsed_time} 秒"