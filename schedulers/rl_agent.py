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
        self._burst_overload_count = 0
        self._total_overload_count = 0

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
    
    def reset_burst_metrics(self):
        self._burst_overload_count = 0
        self._total_overload_count = 0
    
    def log_overload(self, step, is_overloaded):
        if is_overloaded:
            self._total_overload_count += 1
            if 200 <= step <= 300:
                self._burst_overload_count += 1
    
    def print_burst_metrics(self):
        print("\n[RL_BURST_METRICS]")
        print(f"BurstOverloadCount: {self._burst_overload_count}")
        print(f"TotalOverloadCount: {self._total_overload_count}")
        
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

    def rl_diagnostics(self):
        """
        Computes and prints Q-table diagnostics.
        """
        import math
        
        num_states = len(self.q_table)
        max_abs_q = 0.0
        has_nan = False
        has_inf = False
        
        for state, q_values in self.q_table.items():
            for q_val in q_values:
                if math.isnan(q_val):
                    has_nan = True
                if math.isinf(q_val):
                    has_inf = True
                max_abs_q = max(max_abs_q, abs(q_val))
        
        print("[RL_DIAGNOSTICS]")
        print(f"States: {num_states}")
        print(f"MaxAbsQ: {max_abs_q}")
        print(f"NaN: {has_nan}")
        print(f"Inf: {has_inf}")

    def run_stability_check(self, train_callable, seed):
        """
        Runs training 3 times with same seed to check stability.
        """
        import math
        
        # Store initial epsilon
        initial_epsilon = self.epsilon
        
        runs_data = []
        
        for run_idx in range(3):
            # Reset Q-table before each run
            self.q_table = defaultdict(lambda: [0.0] * self.num_nodes)
            
            # Reset epsilon to initial value before each run
            self.epsilon = initial_epsilon
            
            # Run training with same seed
            random.seed(seed)
            train_callable()
            
            # Compute diagnostics after this run
            num_states = len(self.q_table)
            max_abs_q = 0.0
            has_nan = False
            has_inf = False
            
            for state, q_values in self.q_table.items():
                for q_val in q_values:
                    if math.isnan(q_val):
                        has_nan = True
                    if math.isinf(q_val):
                        has_inf = True
                    max_abs_q = max(max_abs_q, abs(q_val))
            
            runs_data.append({
                'states': num_states,
                'max_abs_q': max_abs_q,
                'nan': has_nan,
                'inf': has_inf
            })
        
        # Check consistency
        consistent = (
            runs_data[0]['states'] == runs_data[1]['states'] == runs_data[2]['states'] and
            runs_data[0]['nan'] == runs_data[1]['nan'] == runs_data[2]['nan'] and
            runs_data[0]['inf'] == runs_data[1]['inf'] == runs_data[2]['inf']
        )
        
        print("[RL_STABILITY_CHECK]")
        print(f"Run1: States={runs_data[0]['states']}")
        print(f"Run2: States={runs_data[1]['states']}")
        print(f"Run3: States={runs_data[2]['states']}")
        print(f"Consistent: {consistent}")
