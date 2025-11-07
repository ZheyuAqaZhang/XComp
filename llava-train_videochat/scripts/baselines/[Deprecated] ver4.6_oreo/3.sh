export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

# Most Important Part
STAGE_ID="3"
EXP_NAME="oreo"
DATA_VERSION="data/oreo/stage3_short-long_mix_sft_oreo_mid.yaml"
#--------------------

# Hyperparameters
export EXTRA_PARAM_INNER_CONDENSER_ID="[10, 20]"
export EXTRA_PARAM_INNER_CONDENSER_TYPE="nexthalf"
export EXTRA_PARAM_OUTER_LEARNABLE_TOKEN="4"

export EXTRA_PARAM_INNER_STRIDE="4"
export EXTRA_PARAM_INNER_STRIDE_LIST="[5,4]"
export EXTRA_PARAM_IMAGE_LAST="Middle"
#--------------------

DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")

VISION_MODEL_VERSION="umt-hd-large"
VISION_MODEL_VERSION_CLEAN="umt-hd-large"

# LLM_VERSION="/home/ec2-user/workspace/VideoChat-Flash-forked/llava-train_videochat/checkpoints/spatial_temporal/stage1-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B_stage1_init_connector_iv1m.yaml"
LLM_VERSION="/home/ec2-user/workspace/VideoChat-Flash-forked/llava-train_videochat/checkpoints/baseline-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B"
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

mkdir -p ${CHECKPOINT_DIR}/logs
NUM_GPU=8

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPU} --master_port=29502 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_condenser,mm_language_model" \
    --mm_vision_tower_lr=6e-7 \
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
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 3e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 128 \
    --frames_lowbound 16 \
    --time_msg short \
    --local_num_frames 4 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 4 \
    --attn_implementation flash_attention_2 \
    --verbose_logging True


# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPU} \
#     llava/train/train_mem.py \
#     --deepspeed scripts/zero1.json \
#     --model_name_or_path ${LLM_VERSION} \
#     --version ${PROMPT_VERSION} \
#     --data_path ${DATA_VERSION} \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,mm_condenser" \
#     --mm_vision_select_layer -2 \
#     --mm_projector_type ${mm_projector_type} \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --run_name $MID_RUN_NAME \
#     --output_dir ${CHECKPOINT_DIR} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 1 \
#     --learning_rate 1e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 8192 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 12 \
#     --lazy_preprocess True \
#     --report_to tensorboard \
#     --attn_implementation flash_attention_2 \
#     --frames_upbound 4 \
#     --time_msg short \
#     --local_num_frames 4 \
#     --sample_type middle \
#     --vision_encode_type video_image \
#     --mm_pos_num_frames 4 \
#     --mm_local_num_frames 4 \
#     --mm_vision_tower_lr=2e-6 \
#     --verbose_logging True

# You can delete the sdpa attn_implementation if you want to use flash attn
# Originally, frames_upbound was 512
# attn_implementation was sdpa
    # --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model,mm_condenser" \