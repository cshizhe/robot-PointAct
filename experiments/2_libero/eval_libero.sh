#!/bin/bash

task_suite_name=$1  # spatial, object, goal, 10
ckpt_dir=$2
ckpt_step=$3
pred_rot_type=$4 # rot6d, euler
options=$5  # --args.use_depth --args.verbose --args.delta_action --args.save_video 

host=localhost
port=$((10000 + RANDOM % 10000))

task_suite_name=libero_${task_suite_name}

seed=7
num_denoise_steps=10

export PYTHONPATH=$(pwd):$PYTHONPATH

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
    port=$(( RANDOM % 10000 + 10000 ))  # 10000~20000
done
echo "port=$port"

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
client_python_bin=$HOME/miniconda3/envs/libero/bin/python

${client_python_bin} experiments/2_libero/run_libero_client.py \
    --args.seed ${seed} \
    --args.host ${host} --args.port ${port} \
    --args.task_suite_name ${task_suite_name} \
    --args.num_trials_per_task 50 \
    --args.pred_rot_type ${pred_rot_type} \
    --args.replan_steps 8 \
    --args.save_dir ${ckpt_dir}/results/checkpoint-${ckpt_step} \
    ${options}

echo ${ckpt_dir}/checkpoint-${ckpt_step}
echo "Client finished"

kill $SERVER_PID
wait $SERVER_PID

echo "Server exited"
