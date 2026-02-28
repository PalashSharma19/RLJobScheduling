"""
Baseline schedulers for performance comparison.
"""
import random

class RandomScheduler:
    """Selects a node entirely at random."""
    def select_node(self, nodes):
        """Returns a randomly chosen node."""
        return random.choice(nodes)


class RoundRobinScheduler:
    """Distributes tasks sequentially among nodes."""
    def __init__(self, num_nodes):
        self.pointer = 0
        self.num_nodes = num_nodes

    def select_node(self, nodes):
        """Returns the next node in the round-robin sequence."""
        node = nodes[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_nodes
        return node


class LeastLoadedScheduler:
    """Selects the node that currently has the lowest processing load."""
    def select_node(self, nodes):
        """Returns the node with the minimum current_load."""
        return min(nodes, key=lambda n: n.current_load)
