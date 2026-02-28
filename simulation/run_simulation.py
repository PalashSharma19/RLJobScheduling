"""
Centralized simulation loop.
"""
from config import NUM_NODES, MAX_CAPACITY, TOTAL_TIMESTEPS
from environment.nodes import EdgeNode
from environment.task_generator import TaskGenerator

def run_simulation(scheduler, is_training=False, num_nodes=NUM_NODES, 
                   max_capacity=MAX_CAPACITY, time_steps=TOTAL_TIMESTEPS):
    """
    Runs a single simulation episode for the given scheduler.
    
    If `is_training` is True, expects the scheduler to be an RL agent
    and calls its `update()` method to learn.
    """
    nodes = [EdgeNode(i, max_capacity) for i in range(num_nodes)]
    task_gen = TaskGenerator()

    total_latency = 0.0
    overload_count = 0
    task_count = 0

    for step in range(1, time_steps + 1):
        # 1. Check if task arrives
        task_size = task_gen.generate_task(step)

        if task_size > 0:
            task_count += 1
            
            # 2. Get state (needed if RL)
            state = None
            if hasattr(scheduler, "get_state"):
                state = scheduler.get_state(nodes, task_size)
            
            # 3. Choose Node
            if hasattr(scheduler, "get_state"):
                action = scheduler.select_node(state, is_training=is_training)
                chosen_node = nodes[action]
            else:
                chosen_node = scheduler.select_node(nodes)

            # 4. Assign task
            chosen_node.assign_task(task_size)

            # 5. Compute latency and Overload
            latency = chosen_node.compute_latency()
            total_latency += latency
            
            is_overloaded = chosen_node.is_overloaded()
            if is_overloaded:
                overload_count += 1
                
            # 6. Update agent if training
            if is_training and hasattr(scheduler, "update"):
                reward = scheduler.calculate_reward(nodes, latency, is_overloaded)
                next_state = scheduler.get_state(nodes, task_size)
                scheduler.update(state, action, reward, next_state)

        # 7. Process time step for all nodes
        for node in nodes:
            node.process_time_step()

    avg_latency = total_latency / task_count if task_count > 0 else 0.0

    return avg_latency, overload_count, task_count
