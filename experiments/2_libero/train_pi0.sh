ulimit -u 2048

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPUS=2 #8
PER_DEVICE_BATCH_SIZE=64 #128

ACCELERATE_ARGS="--num_machines 1 --machine_rank 0 --num_processes=${GPUS} --multi_gpu --mixed_precision=bf16 --dynamo_backend no"
# ACCELERATE_ARGS="--num_machines 1 --machine_rank 0 --num_processes=${GPUS} --mixed_precision=bf16 --dynamo_backend no"

# datasets
dataset=experiments/2_libero/data_configs/data-libero-spatial.yaml
dataset_name=$(basename ${dataset%.*})-euler-frontview

# hparams
lr=5e-5
mlr=5e-5
vlr=2e-5

chunk_size=50
epoch=50

run_name=${dataset_name}_ck${chunk_size}_lr${lr}_vlr${vlr}_mlr${mlr}_gpu${GPUS}_bs${PER_DEVICE_BATCH_SIZE}_epoch${epoch}

model_name_or_path=lerobot/pi0_base
pi_state_action_norm_file=robot_data/libero/lerobot_point_lmdb/libero_spatial_no_noops/robot_state_action_stats/euler.json

output_dir=$SCRATCH/datasets/PointAct_exprs/libero/pi0/${run_name}-tune.vision.tower
# output_base=null # with time in directory name
echo $output_dir

# Determine TF32 support
TF32_SUPPORT="False"
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
echo "GPU Compute Capability: $COMPUTE_CAP"
# Check if Ampere or newer (compute capability >= 8.0)
if [ "$MAJOR" -ge 8 ]; then
    TF32_SUPPORT="True"
fi
echo "TF32_SUPPORT: $TF32_SUPPORT"


accelerate launch $ACCELERATE_ARGS scripts/train_pi.py \
    --output_dir ${output_dir} \
    --data-path ${dataset} \
    --model-class pi0 \
    ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
    --pi_state_action_norm_file ${pi_state_action_norm_file} \
    --pi_train_expert_only False \
    --freeze-vision-tower False \
    --max-state-dim 32 \
    --chunk-size ${chunk_size} \
    --dataloader-num-workers 8 \
    --bf16 True \
    --tf32 ${TF32_SUPPORT} \
    --fp16 False \
    --num-train-epochs ${epoch} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --learning-rate ${lr} \
    --merger-lr ${mlr} \
    --vision-lr ${vlr} \
    --weight-decay 0.001 \
    --warmup-steps 0.03 \
    --lr-scheduler-type cosine \
    --gradient-checkpointing True \
    --save-strategy steps \
    --logging-steps 10 \
    --save-steps 1000 \
    --save-total-limit 10 \
    --run-name ${run_name} \
    --attn-implementation flash_attention_2 \
    --log_level info \
    --report-to tensorboard \
    --max_grad_norm 3 \
    --use_robot_state True 
    # --report-to none / tensorboard / wandb
