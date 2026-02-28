"""
Environment simulation for edge computing nodes.
"""
from config import BASE_LATENCY, LATENCY_ALPHA, OVERLOAD_THRESHOLD, LOAD_DECREASE_RATE, MAX_CAPACITY

class EdgeNode:
    """
    Represents an edge server node that processes tasks.
    """
    def __init__(self, node_id, max_capacity=MAX_CAPACITY):
        self.node_id = node_id
        self.max_capacity = max_capacity
        self.current_load = 0

    def assign_task(self, task_size):
        """Increase the current load by the assigned task's size."""
        self.current_load += task_size

    def process_time_step(self):
        """Process tasks, reducing the current load deterministically per timestep."""
        if self.current_load > 0:
            # Load decreases deterministically by 1 per timestep if > 0
            self.current_load -= LOAD_DECREASE_RATE

    def compute_latency(self):
        """Calculate the latency based on current load."""
        return BASE_LATENCY + (LATENCY_ALPHA * self.current_load)

    def is_overloaded(self):
        """Check if the node's current load exceeds the overload threshold."""
        return self.current_load > OVERLOAD_THRESHOLD

    def __repr__(self):
        return f"Node {self.node_id} | Load: {self.current_load}/{self.max_capacity}"
