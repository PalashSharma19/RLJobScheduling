"""
Main Entry Point
Runs baseline testing, RL training, RL Evaluation, and generates plots.
"""
from config import NUM_NODES, RANDOM_SEED
from schedulers.baselines import RandomScheduler, RoundRobinScheduler, LeastLoadedScheduler
from metrics.metrics import evaluate_scheduler, train_rl, evaluate_rl
from plots.plot_results import plot_latency
import random

def main():
    print("--- Edge Grid Scheduler Refactoring ---")
    
    # Set seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Init schedulers
    random_scheduler = RandomScheduler()
    rr_scheduler = RoundRobinScheduler(NUM_NODES)
    least_scheduler = LeastLoadedScheduler()

    print("\n[1/4] Evaluating Baseline Schedulers...")
    r_latency, r_overload, r_tasks = evaluate_scheduler(random_scheduler, runs=10)
    print(f"Random      -> Latency: {r_latency:.2f}, Overloads: {r_overload:.2f}, Tasks: {r_tasks:.2f}")

    rr_latency, rr_overload, rr_tasks = evaluate_scheduler(rr_scheduler, runs=10)
    print(f"RoundRobin  -> Latency: {rr_latency:.2f}, Overloads: {rr_overload:.2f}, Tasks: {rr_tasks:.2f}")

    l_latency, l_overload, l_tasks = evaluate_scheduler(least_scheduler, runs=10)
    print(f"LeastLoaded -> Latency: {l_latency:.2f}, Overloads: {l_overload:.2f}, Tasks: {l_tasks:.2f}")

    print("\n[2/4] Training Tabular Q-Learning Agent...")
    # Training
    rl_agent = train_rl(episodes=250)
    print("Training Complete.")

    print("\n[3/4] Evaluating RL Agent...")
    # Evaluation
    rl_latency, rl_overload, rl_tasks = evaluate_rl(rl_agent)
    print(f"RL Agent    -> Latency: {rl_latency:.2f}, Overloads: {rl_overload:.2f}, Tasks: {rl_tasks:.2f}")

    print("\n[4/4] Generating Charts...")
    results = {
        "Random": (r_latency, r_overload, r_tasks),
        "RoundRobin": (rr_latency, rr_overload, rr_tasks),
        "LeastLoaded": (l_latency, l_overload, l_tasks),
        "RL Agent": (rl_latency, rl_overload, rl_tasks),
    }

    # Plot and save to root of the project
    plot_latency(results, save_path="latency_comparison.png")
    print("Chart saved as latency_comparison.png!")
    print("Done!")

if __name__ == "__main__":
    main()
