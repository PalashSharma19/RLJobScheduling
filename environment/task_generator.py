"""
Environment component responsible for generating user tasks.
"""
import random
from config import (
    NORMAL_ARRIVAL_PROB, BURST_ARRIVAL_PROB, BURST_WINDOW_START, BURST_WINDOW_END,
    TASK_SIZE_MIN, TASK_SIZE_MAX
)

class TaskGenerator:
    """
    Generates tasks dynamically over time, incorporating bursty arrival patterns.
    """
    
    def generate_task(self, current_timestep):
        """
        Determine if a task arrives based on timestep probabilities,
        and generate its size.
        
        Returns:
            task_size (int): Size of the task, or 0 if no task arrives.
        """
        is_burst = BURST_WINDOW_START <= current_timestep <= BURST_WINDOW_END
        arrival_prob = BURST_ARRIVAL_PROB if is_burst else NORMAL_ARRIVAL_PROB
        
        if random.random() < arrival_prob:
            return random.randint(TASK_SIZE_MIN, TASK_SIZE_MAX)
        return 0
