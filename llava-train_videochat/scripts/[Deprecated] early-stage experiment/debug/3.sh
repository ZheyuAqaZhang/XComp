export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

# Most Important Part
STAGE_ID="1"
EXP_NAME="debug"
DATA_VERSION="data/stage1_init_connector_iv1m.yaml"
#--------------------

# Hyperparameters

#--------------------

DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")

VISION_MODEL_VERSION="umt-hd-large"
VISION_MODEL_VERSION_CLEAN="umt-hd-large"

LLM_VERSION="/home/ec2-user/workspace/VideoChat-Flash-forked/llava-train_videochat/checkpoints/baseline"
LLM_VERSION_CLEAN="Qwen2_5_1_5B"

mm_projector_type=tome16_mlp_hd64
PROMPT_VERSION="qwen_2"

MID_RUN_NAME=stage${STAGE_ID}-${VISION_MODEL_VERSION}-${mm_projector_type}_${LLM_VERSION_CLEAN}_${DATA_VERSION_CLEAN}
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CHECKPOINT_DIR="./checkpoints/${EXP_NAME}/${MID_RUN_NAME}"
if [ -d "${CHECKPOINT_DIR}" ]; then
  all_checkpoints=( $(ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null || true) )
  if [ ${#all_checkpoints[@]} -gt 1 ]; then
    sorted=( $(printf '%s\n' "${all_checkpoints[@]}" | sort -t '-' -k2n) )
    # sorted[0] 即数字最小（最早）的 checkpoint，保留；其余删除
    for ckpt in "${sorted[@]:1}"; do
      rm -rf "${ckpt}"
      echo "Removed ${ckpt}"
    done
  fi
fi

mkdir -p ${CHECKPOINT_DIR}/logs/train.log
NUM_GPU=2

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPU} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ${CHECKPOINT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 160 \
    --frames_lowbound 64 \
    --time_msg short \
    --local_num_frames 8 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 4 \
    --verbose_logging True >> ${CHECKPOINT_DIR}/logs/train.log



# You can delete the sdpa attn_implementation if you want to use flash attn
# Originally, frames_upbound was 512
# Originally, mm_local_num_frames was 8