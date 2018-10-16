import torch
import torch.nn as nn

import numpy as np
import math

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))    

# Utils for prioritized replay buffer and sampling
# Segment tree data structure where parent node values are sum/max of children node values
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
class SumTree(object):
    def __init__(self, capacity):
        """ 
            Initialize the tree with all nodes  = 0 
            Initialize the data with all values = 0 
        """
        self.capacity     = capacity
        self.position     = 0
        self.dataArr = np.zeros(capacity, dtype=object)
        self.treeArr = np.zeros(2*capacity - 1)
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity 
        """   tree:
                            0
                           / \
                          0   0
                         / \ / \
                        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
    def push(self, priority, data):
        """ Look at what index we want to put the new transition at """
        tree_index = self.position + self.capacity - 1
        """                
            tree:
                        0
                       / \
                      0   0
                     / \ / \
            tree_index  0 0  0    

        We fill the leaves from left to right
        """
        self.dataArr[self.position] = data    # Update data frame
        self.update(tree_index, priority)  # Update the leaf, using the function below
        #
        self.position += 1
        if self.position >= self.capacity:
            self.position = 0
        #

    def update(self, tree_index, priority):
        """
            Change_of_Score = new priority score - former priority score
        """
        delta_score = priority - self.treeArr[tree_index]
        self.treeArr[tree_index] = priority

        # Propagate this change through tree
        """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE "INDEXES" NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
        """     
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.treeArr[tree_index] += delta_score

    def get_leaf(self, v):
        """
        Return the leaf_index, that is the "priority value" of the transition at that leaf.

            Tree structure and array storage:
            Tree index:
                 0         -> storing priority sum
                / \
              1     2
             / \   / \
            3   4 5   6    -> storing priority for experiences
            Array type for storing:
            [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            #
            # If we reach bottom, end the search
            if left_child_index >= len(self.treeArr):
                LEAF_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.treeArr[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.treeArr[left_child_index]
                    parent_index = right_child_index
            #
        # The corresponding data index: 
        data_index = LEAF_index - self.capacity + 1
        return LEAF_index, self.treeArr[LEAF_index], self.dataArr[data_index]

    def get_total_priority(self):
        return self.treeArr[0] # The root node contains the total priority
