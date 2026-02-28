"""
Tabular Q-Learning Scheduler.
"""
import random
from collections import defaultdict
import statistics

from config import NUM_NODES, PENALTY_OVERLOAD, LOAD_VARIANCE_WEIGHT

class QLearningScheduler:
    """
    Reinforcement Learning based scheduler using Tabular Q-Learning.
    """
    def __init__(self, num_nodes=NUM_NODES, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-values to 0.0
        self.q_table = defaultdict(lambda: [0.0] * self.num_nodes)

    def discretize_load(self, load):
        """
        Groups the current load into 4 buckets:
        0 = Empty (0)
        1 = Low (1-3)
        2 = Medium (4-7)
        3 = High (8+)
        """
        if load <= 0:
            return 0
        elif load <= 3:
            return 1
        elif load <= 7:
            return 2
        else:
            return 3

    def get_state(self, nodes, task_size):
        """
        Creates the state representation tuple:
        (Load1, Load2, Load3, Load4, Load5, TaskSize)
        """
        loads = tuple(self.discretize_load(n.current_load) for n in nodes)
        return loads + (task_size,)

    def select_node(self, state, is_training=True):
        """
        Selects an action using an epsilon-greedy policy.
        """
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.num_nodes - 1)

        q_values = self.q_table[state]
        return q_values.index(max(q_values))

    def update(self, state, action, reward, next_state):
        """
        Performs the Q-learning update step.
        """
        best_next = max(self.q_table[next_state])
        old_value = self.q_table[state][action]

        self.q_table[state][action] = old_value + self.alpha * (
            reward + self.gamma * best_next - old_value
        )
        
    def calculate_reward(self, nodes, chosen_node_latency, chosen_node_overloaded):
        """
        Calculates the exact reward functionality.
        Reward = - Latency - (0.3 * LoadVariance)
        Overload penalty = -10
        """
        # Calculate variance across current loads
        current_loads = [n.current_load for n in nodes]
        variance = statistics.variance(current_loads) if len(current_loads) > 1 else 0.0
        
        reward = -chosen_node_latency - (LOAD_VARIANCE_WEIGHT * variance)
        if chosen_node_overloaded:
            reward += PENALTY_OVERLOAD
            
        return reward
