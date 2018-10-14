export CUDA_VISIBLE_DEVICES=0;
OMP_NUM_THREADS=1 time python main.py \
2>&1 | tee myLog.txt;

# Default to use Vanilla DQN
# --use-double-dqn
# --use-prioritized-buffer
