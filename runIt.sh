export CUDA_VISIBLE_DEVICES=3;
OMP_NUM_THREADS=1 time python main.py \
--env-name "SpaceInvadersNoFrameskip-v4" \
--log-dir "./agentLog" \
--save-dir "./saved_model" \
--total-timestep 1e8 \
--memory-size     50000 \
--learning-starts 20000 \
--target-update   8192 \
--seed 1234 \
--use-double-dqn \
--use-prioritized-buffer \
--use-n-step \
--use-duel \
--use-noisy-net \
--use-C51 \
--use-QR-C51 \
2>&1 | tee myLog.txt;

# Default to use Vanilla DQN
# --use-double-dqn
# --use-prioritized-buffer
