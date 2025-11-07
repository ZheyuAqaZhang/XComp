TASK=mvbench
MODEL_NAME=videochat_flash
MAX_NUM_FRAMES=512
CKPT_PATH=OpenGVLab/VideoChat-Flash-Qwen2-7B_res448

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=8


accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args pretrained=$CKPT_PATH,max_num_frames=$MAX_NUM_FRAMES \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}

