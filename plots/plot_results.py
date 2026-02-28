"""
Plotting utilities.
"""
import matplotlib.pyplot as plt

def plot_latency(results, save_path=None):
    """
    Plots the average latency for each tested scheduler.
    """
    names = list(results.keys())
    latencies = [results[name][0] for name in names]

    plt.figure()
    plt.bar(names, latencies, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel("Schedulers")
    plt.ylabel("Average Latency")
    plt.title("Scheduler Latency Comparison")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
