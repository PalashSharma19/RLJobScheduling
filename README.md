# Adaptive RL Edge Scheduler

This project evaluates different task scheduling algorithms for Edge Computing networks. It implements baseline schedulers (Random, Round-Robin, Least Loaded) versus a Tabular Q-Learning Reinforcement Learning agent.

## Refactoring Log

This repository standardizes and merges two differing implementations of the same simulation base. The discrepancies removed include:

1. **Variables and Constants Standardization**: Consolidated to a strict final spec. `config.py` acts as the single source of truth for max capacity, burst probabilities, normal probabilities, and all magic variables.
2. **Latency Discrepancies**: Latency formula mathematically unified to `1 + 0.5 * current_load`. Alpha parameters removed from constructor injection to enforce identical evaluation criteria. 
3. **Overload Parameters**: Defined overload specifically as `> 10`, with determinism rather than probability for load dropping (decreases precisely by 1 every timestep).
4. **State Confusion**: The state space representation was locked down into a distinct bucketed tuple sequence of `(Load1, Load2, Load3, Load4, Load5, TaskSize)`, mapping load values continuously to integer buckets (`0, 1, 2, 3`).
5. **Reward Uniqueness**: Set explicitly as `-Latency - (0.3 * LoadVariance)` alongside an explicit `-10` logic step for overloads.
6. **Simulation Loops**: Previously there existed separated loops scattered in `edge_simulation.py` and `metrics.py` (during training wrappers). It has been stripped to one single functional sequence inside `run_simulation.py` which takes a boolean `is_training` to indicate RL adjustments or evaluation behavior respectively. Multiple loops allowed inconsistent task loads to distort outputs.

## Usage

Simply run:
```bash
python main.py
```

This will run evaluators, train the RL, run a strict evaluation over the agent out of sample, and generate a latency chart.
