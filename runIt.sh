export CUDA_VISIBLE_DEVICES=0;
SEED=1;
GAME="Assault";
NAME="DQN-C51-"$GAME"-SEED-"$SEED;
LOG_NAME="./TXT_LOGS/myLog_"$NAME".txt";

if [ ! -f  $LOG_NAME ]; then

OMP_NUM_THREADS=1 time python main.py \
--env-name $GAME"NoFrameskip-v4" \
--log-dir "./agentLog_"$SEED \
--save-dir "./saved_model_"$SEED \
--total-timestep 1e8 \
--memory-size     50000 \
--learning-starts 20000 \
--target-update   8192 \
--seed $SEED \
--use-double-dqn \
--use-C51 \
2>&1 | tee $LOG_NAME;

else
	echo "Danger close. The log file at -- $LOG_NAME -- exists!!"
fi

# --use-QR-C51 \
# --use-prioritized-buffer \
# --use-n-step \
# --use-duel \
# --use-noisy-net \
# --use-C51 \
# --use-QR-C51 \

# Default to use Vanilla DQN
# --use-double-dqn
# --use-prioritized-buffer
