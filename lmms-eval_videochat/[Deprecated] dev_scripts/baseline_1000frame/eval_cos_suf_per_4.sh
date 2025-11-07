TASK=mvbench
MODEL_NAME=videochat_flash
MAX_NUM_FRAMES=1024
# CKPT_PATH=OpenGVLab/VideoChat-Flash-Qwen2-7B_res448
CKPT_PATH=../llava-train_videochat/checkpoints/baseline_1000frame_cos/stagesuf-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B_stage3_short-long_mix_sft_mid2.yaml
BASE_PATH=benchmark_data/model/VideoChat-Flash-Qwen2_5-2B_res448_ver2/
cache_path=${CKPT_PATH}/cache2

# Hyperparameters
# export EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE="Cosine" 
# export EXTRA_PARAM_ALWAYS_SUFFIX="True"
export EXTRA_PARAM_INNER_CONDENSER_ID="[30]"
export EXTRA_PARAM_INNER_CONDENSER_TYPE="nexthalf"

export EXTRA_PARAM_INNER_STRIDE="4"

export EXTRA_PARAM_FPS="4"
#--------------------

# Eval Setting
export EXTRA_OVERWRITE_MAX_NUM_FRAMES="1024"
export EXTRA_OVERWRITE_MIN_NUM_FRAMES="64"
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



# |               Tasks               |Version|Filter|n-shot|     Metric     |Value |   |Stderr |
# |-----------------------------------|-------|------|-----:|----------------|-----:|---|-------|
# |mvbench                            |N/A    |none  |     0|mvbench_accuracy|68.725|±  |15.8499|
# | - mvbench_action_antonym          |Yaml   |none  |     0|mvbench_accuracy|87.000|±  |N/A    |
# | - mvbench_action_count            |Yaml   |none  |     0|mvbench_accuracy|58.000|±  |N/A    |
# | - mvbench_action_localization     |Yaml   |none  |     0|mvbench_accuracy|52.500|±  |N/A    |
# | - mvbench_action_prediction       |Yaml   |none  |     0|mvbench_accuracy|76.000|±  |N/A    |
# | - mvbench_action_sequence         |Yaml   |none  |     0|mvbench_accuracy|85.500|±  |N/A    |
# | - mvbench_character_order         |Yaml   |none  |     0|mvbench_accuracy|64.000|±  |N/A    |
# | - mvbench_counterfactual_inference|Yaml   |none  |     0|mvbench_accuracy|78.500|±  |N/A    |
# | - mvbench_egocentric_navigation   |Yaml   |none  |     0|mvbench_accuracy|33.500|±  |N/A    |
# | - mvbench_episodic_reasoning      |Yaml   |none  |     0|mvbench_accuracy|62.000|±  |N/A    |
# | - mvbench_fine_grained_action     |Yaml   |none  |     0|mvbench_accuracy|46.500|±  |N/A    |
# | - mvbench_fine_grained_pose       |Yaml   |none  |     0|mvbench_accuracy|75.500|±  |N/A    |
# | - mvbench_moving_attribute        |Yaml   |none  |     0|mvbench_accuracy|92.500|±  |N/A    |
# | - mvbench_moving_count            |Yaml   |none  |     0|mvbench_accuracy|70.000|±  |N/A    |
# | - mvbench_moving_direction        |Yaml   |none  |     0|mvbench_accuracy|38.000|±  |N/A    |
# | - mvbench_object_existence        |Yaml   |none  |     0|mvbench_accuracy|88.500|±  |N/A    |
# | - mvbench_object_interaction      |Yaml   |none  |     0|mvbench_accuracy|85.000|±  |N/A    |
# | - mvbench_object_shuffle          |Yaml   |none  |     0|mvbench_accuracy|43.000|±  |N/A    |
# | - mvbench_scene_transition        |Yaml   |none  |     0|mvbench_accuracy|92.500|±  |N/A    |
# | - mvbench_state_change            |Yaml   |none  |     0|mvbench_accuracy|67.000|±  |N/A    |
# | - mvbench_unexpected_action       |Yaml   |none  |     0|mvbench_accuracy|79.000|±  |N/A    |

# |Groups |Version|Filter|n-shot|     Metric     |Value |   |Stderr |
# |-------|-------|------|-----:|----------------|-----:|---|------:|
# |mvbench|N/A    |none  |     0|mvbench_accuracy|68.725|±  |15.8499|