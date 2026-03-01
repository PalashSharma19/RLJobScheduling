"""
Metrics collection and training/evaluation wrappers.
"""
from simulation.run_simulation import run_simulation
from schedulers.rl_agent import QLearningScheduler
from config import RANDOM_SEED

def evaluate_scheduler(scheduler, runs=10):
    """
    Evaluates a baseline scheduler over multiple runs to handle variance.
    """
    latencies = []
    overloads = []
    tasks = []

    for _ in range(runs):
        avg_latency, overload_count, task_count = run_simulation(scheduler, is_training=False)
        latencies.append(avg_latency)
        overloads.append(overload_count)
        tasks.append(task_count)

    return (
        sum(latencies) / runs,
        sum(overloads) / runs,
        sum(tasks) / runs,
    )

def train_rl(episodes=250):
    """
    Trains the RL Q-learning Agent for multiple episodes.
    """
    agent = QLearningScheduler()

    def single_training_run():
        for _ in range(episodes):
            # Run episode with training=True
            run_simulation(agent, is_training=True)

            # Decay epsilon post-episode
            agent.epsilon = max(0.05, agent.epsilon * 0.98)

    agent.run_stability_check(single_training_run, seed=RANDOM_SEED)
    
    agent.rl_diagnostics()

    return agent

def evaluate_rl(agent):
    """
    Evaluates a trained RL agent by turning off exploration.
    """
    # Temporarily store and disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # Reset burst metrics before evaluation
    agent.reset_burst_metrics()
    
    # Evaluate via normal multiple runs logic
    avg_latency, overload_count, task_count = evaluate_scheduler(agent, runs=10)
    
    # Print burst metrics after evaluation
    agent.print_burst_metrics()
    
    # Restore epsilon 
    agent.epsilon = original_epsilon
    
    return avg_latency, overload_count, task_count
