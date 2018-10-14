export CUDA_VISIBLE_DEVICES=0;
OMP_NUM_THREADS=1 time python main.py \
--env-name "PongNoFrameskip-v4" \
--log-dir "./agentLog" \
--save-dir "./saved_model" \
--total-timestep 1e8 \
--memory-size     10000 \
--learning-starts 10000 \
--seed 1234 \
--use-double-dqn \
--use-prioritized-buffer \
2>&1 | tee myLog.txt;

# Default to use Vanilla DQN
# --use-double-dqn
# --use-prioritized-buffer
