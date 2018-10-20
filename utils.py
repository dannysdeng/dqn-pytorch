import operator
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


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

#
# Utils for prioritized replay buffer and sampling
# Segment tree data structure where parent node values are sum/max of children node values
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
class SumTree(object):
    """ Deprecated """
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
            left_child_index  = 2 * parent_index + 1
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



def PER_pre_fill_memory(envs):
    """
    Pre-filling the memory buffer if we are doing Prioritized Experience Replay 
    """
    state = envs.reset()
    print_now('[Warning] Begin to pre-fill [Prioritized Experience Replay Memory]')
    for j in range(args.memory_size):
        action = torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)
        st_0 = copy.deepcopy(state)      # IMPORTANT. Make a deep copy as state will be come next_state AUTOMATICALLY
        next_state, reward, done, info = envs.step(action)
        st_1 = copy.deepcopy(next_state)
        # We only ensure one environment here
        # -------------------------------------------------------------------######    
        if USE_N_STEP:
            st_0, action, st_1, reward = n_step_preprocess(st_0, action, st_1, reward, done[0])
        # -------------------------------------------------------------------######    
        if done[0]:
            memory.push(st_0, action, None, reward)
            print_now('Pre-filling Replay Memory %d / %d -- action: %d' % (j+1, args.memory_size, action.item()))
        elif st_0 is not None:
            memory.push(st_0, action, st_1, reward)      
            print_now('Pre-filling Replay Memory %d / %d -- action: %d' % (j+1, args.memory_size, action.item()))
        state = next_state  
    return state
    #        
