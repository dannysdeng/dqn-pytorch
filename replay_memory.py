from collections import namedtuple
import random
from utils import SumTree

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # args is like def push(self, state, action, next_state, reward), 
    # So Transition(state, action, next_state, reward) becomes what we need to store
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayBuffer():
    # Remember to deal with EMPTY MEMORY PROBLEM by pre-filling the prioritized memory with Random (St0, At0, St1, Rt0)
    def __init__(self, args, capacity, beta):
        self.capacity = capacity
        # We may want better data structure: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
        self.memory   = SumTree(capacity) 

        self.e     = 0.01
        self.alpha = 0.6 # tradeoff between taking only experience with high-priority samples
        self.beta  = 0.4 # Importance Sampling, from 0.4 -> 1.0 over the course of training
        self.beta_increment = 0.001
        self.abs_error_clipUpper = 1.0

    def push(self, *args):      
        # Find the max priority. Recall that treeArr is of size 2*capacity - 1. 
        # And all the priorioties lie on the leaves of the tree
        all_priority = self.memory.treeArr[-self.memory.capacity:]
        max_priority = np.max(max_priority)
        if max_priority == 0:
            max_priority = self.abs_error_clipUpper
        # Setting maximum priority for new transitions. Total priority will be updated
        transition = Transition(*args)
        self.memory.push(max_priority, transition) 

    def sample(self, batch_size):
        """
        Let N = batch_size

        1. First, divide the range [0, priority_total] into N ranges
        2. Next,  uniformly sample one value per range (out of N ranges)
        3. Then, go and search the SumTree, 
                the transitions with (priority score == sampled values) are retrieved
        4. Finally, calculate Importance Sampling weight, W_is, for each of the element in the minibatch
        """
        this_batch = []
        batch_index     = np.empty((n,  ), dtype=np.int32)
        batch_weight_IS = np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment by dividing the ranges
        total_priority = self.memory.get_total_priority()
        priority_segment = total_priority / batch_size

        # Increase the beta each time we sample a new mini-batch until it reaches 1.0
        self.beta = min(self.beta + self.beta_increment, 1.0)

        # Calculate the max_weight
        all_priority = self.memory.treeArr[-self.memory.capacity:]
        prob_min     = min(all_priority) / total_priority
        """
             [     prob_min * n ]^(-beta)             [   1      1        ]^(beta)
             |------------------|             --->    |  --- * -----------|
             [          1       ]                     [   N     prob_min  ]
        """
        # Getting the MAX of importance sampling weight for nomalization
        max_weight   = (prob_min * n)**(-self.beta)
        # 
        for i in range(batch_size):
            # A value is sample from each range
            A = A_the_ith_range = priority_segment *  i
            B = B_the_ith_range = priority_segment * (i + 1)
            sampled_value = np.random.uniform(A, B)

            # The transition that corresponds to the "sampled_value" is retrieved
            index, priority, data = self.memory.get_leaf(sampled_value)
            transition = data

            # P(j) --> stochastic priority
            stochastic_p = priority / total_priority

            """
                Importance Sampling Weight:
                     [   1      1        ]^(beta)
                     |  --- * -----------|
                     [   N     prob_min  ]                
            """
            this_weight_IS = (stochastic_p * n) ** (-self.beta)
            this_weight_IS /= max_weight # Normalize the max priority

            batch_weight_IS[i, 0] = this_weight_IS
            batch_index[i]        = index
            this_batch.append(transition)
        return this_batch, batch_index, batch_weight_IS

    def update_priority_on_tree_for_a_batch(self, tree_idx, abs_TD_error):
        """ 
            A bunch of tree indices and a bunch of TD_errors
        """
        abs_TD_errors  += self.e # p_t = |delta_t| + e
        clipped_errors = np.minimum(abs_TD_errors, self.abs_error_clipUpper)
        pt_alpha       = np.power(clipped_errors, self.alpha)
        for index, prob in zip(tree_idx, pt_alpha):
            self.memory.update(index, prob)
    # Remember to deal with EMPTY MEMORY PROBLEM













