#!/bin/bash

export sif_image=/scratch/shichen/singularity_images/nvcuda_v2.sif
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}
export PYTHONPATH=$(pwd):$PYTHONPATH

ckpt_dir=$1
ckpt_step=$2
seed=$3 # 100, 200
pred_rot_type=$4
options=$5 # --args.remove_arm --args.save_video --args.project_action_on_image --args.verbose --args.save_obs_out --args.num_episodes 100

host=localhost
port=$((10000 + RANDOM % 10000))
ckpt_file=${ckpt_dir}/checkpoint-${ckpt_step}

num_denoise_steps=10

set -euo pipefail

cleanup() {
  echo "[cleanup] stopping server..."
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  echo "[cleanup] done"
}

trap cleanup INT TERM EXIT

# check if port is in use
is_used() {
    lsof -i TCP:$1 >/dev/null 2>&1
}
while is_used "$port"; do
    echo "Port $port in use, generating new..."
    port=$(( (RANDOM % 10000) + 10000 ))  # 10000~20000
done

# Start server
server_python_bin=$HOME/miniconda3/envs/pointact/bin/python

${server_python_bin} scripts/run_server.py \
    --args.seed ${seed} \
    --args.pretrained_path ${ckpt_dir}/checkpoint-${ckpt_step} \
    --args.num_denoise_steps ${num_denoise_steps} \
    --args.host ${host} --args.port ${port} &

SERVER_PID=$!

echo "Server started, PID=$SERVER_PID"

# wait for server
sleep 20

# Start client
client_python_bin=$HOME/miniconda3/envs/rlbench/bin/python

singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    xvfb-run -a ${client_python_bin} \
    experiments/10_rlbench/run_rlbench_client.py \
    --args.taskvar_file $HOME/codes/GroundedVLA/PointAct/assets/hybridvla_10tasks.json \
    --args.seed ${seed} \
    --args.host ${host} --args.port ${port} \
    --args.replan_steps 1 --args.max_steps 25 \
    --args.pretrained_path ${ckpt_file} \
    --args.clip_within_workspace \
    --args.pred_rot_type ${pred_rot_type} \
    --args.save_dir ${ckpt_dir}/preds/seed${seed} \
    --args.num_workers 4 \
    --args.num_episodes 25 \
    ${options} 

echo ${ckpt_file}
echo "Client finished"

kill $SERVER_PID
wait $SERVER_PID

echo "Server exited"
