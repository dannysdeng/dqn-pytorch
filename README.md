# dqn-pytorch
Towards learning Rainbow-DQN and all that good stuffs.

Run the codes with

     time ./runIt.sh

A few notes (Bug might exist): 

   1. Testing results for 10M is shown in Figure 1. 
   
      The comparisons are for "DDQN + C51", "DDQN + QR-C200" and "DDQN + IQN-64-64-32".
   
   2. Rainbow in this repo runs kinda slow on my machine (_TITAN Xp with Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz_). It only reached 75 FPS  when using a prioritized memory of size 50,000 on the game _Spaceinvaders_, so it seems that it may not be able to finish 200M within 10 days. (Could be because of the sub-optimal way that is used for pushing new transitions into the PER buffer.)
     
[Figure 1]
![alt text](https://raw.githubusercontent.com/dannysdeng/dqn-pytorch/master/test.png)

Useful references:

[0] IQN implementation reference: https://github.com/google/dopamine/tree/master/dopamine

[1] Very helpful pytorch code base: https://github.com/qfettes/DeepRL-Tutorials

[2] Tutorial on C51 https://mtomassoli.github.io/2017/12/08/distributional_rl/

[3] Hyperparam of target interval: lower might be better: https://www.noob-programmer.com/openai-retro-contest/how-to-score-6k-in-leaderboard/

[4] APX-DPG: something better than rainbow: https://arxiv.org/pdf/1803.00933.pdf

