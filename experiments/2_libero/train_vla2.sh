ulimit -u 2048

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH=$(pwd):$PYTHONPATH

GPUS=1
PER_DEVICE_BATCH_SIZE=128

# Build accelerate arguments based on GPU count
if [ $GPUS -eq 1 ]; then
    echo "Single GPU mode"
    ACCELERATE_ARGS="--num_processes 1 --num_machines 1"
elif [ $GPUS -gt 1 ] && [ $NUM_NODES -eq 1 ]; then
    echo "Multi-GPU single node mode"
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines 1 --machine_rank 0"
else
    echo "Multi-GPU multi-node mode"
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines ${NUM_NODES} --machine_rank ${SLURM_NODEID}  --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

# datasets
dataset=experiments/2_libero/data_configs/data-libero-spatial.yaml
dataset_name=$(basename ${dataset%.*})-rot6d-image.frontview

# hparams
lr=5e-5
mlr=5e-5
vlr=2e-5

chunk_size=16
epoch=50

model_name_or_path=
run_name=${dataset_name}_ck${chunk_size}_lr${lr}_gpu${GPUS}_bs${PER_DEVICE_BATCH_SIZE}_epoch${epoch}

output_dir=$SCRATCH/datasets/PointAct_exprs/libero/vla2/${run_name}-freeze.vlm

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


accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --model_class VLADualFlowMatchingModel \
    --output_dir ${output_dir} \
    ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
    --vlm-name-or-path Qwen/Qwen2.5-VL-3B-Instruct \
    --data-path ${dataset} \
    --chunk-size ${chunk_size} \
    --dataloader-num-workers 8 \
    --freeze-vision-tower True \
    --freeze-llm True \
    --freeze-merger True \
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
    --color_aug True \
    --max_grad_norm 3 \
    --use_robot_state True 
