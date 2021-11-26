import tracemalloc

import numpy as np
import quantities as pq
from elephant.spike_train_generation import homogeneous_poisson_process
import matplotlib.pyplot as plt
from online_statistics import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient


def omfr_plot_memory_benchmark_for_instantiation(buffer_sizes, used_memory,
                                                 std_error_memory):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(f"Memory Usage for Instancing Online MFR",
                 y=0.93, fontsize=18)
    # rescale to MB
    current = np.divide(used_memory[:, 0], 10 ** 6)
    peak = np.divide(used_memory[:, 1], 10 ** 6)
    std_error_memory = np.divide(std_error_memory, 10 ** 6)
    ax1.errorbar(buffer_sizes, current, yerr=std_error_memory[:, 0],
                 ecolor='red', color="blue", marker="o", markerfacecolor="blue",
                 markeredgecolor='black',
                 label="current memory usage after instantiation")
    ax1.errorbar(buffer_sizes, peak, yerr=std_error_memory[:, 1], ecolor='red',
                 color="orange", marker="o", markerfacecolor="orange",
                 markeredgecolor='black',
                 label="peak memory usage while instantiation")
    ax1.set_xlabel(f"Buffer Size [s]", fontsize=16)
    ax1.set_ylabel(f"Memory Usage [MB]", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f"plots/omfr_investigate_memory_usage_at_instantiation.pdf")
    plt.show()


def omfr_benchmark_memory(num_repetitions=100):
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]  # in s
    memory_for_instantiation = []
    for rep in range(num_repetitions):
        mem_for_inst_within_rep = []
        for j, bs in enumerate(buffer_sizes):
            # measure memory usage for instantiation
            tracemalloc.start()
            omfr = OnlineMeanFiringRate(buffer_size=bs)
            current_i, peak_i = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_for_inst_within_rep.append((current_i, peak_i))
        memory_for_instantiation.append(mem_for_inst_within_rep)

    # calculating mean and standard error for memory usage at instantiation
    mean_memory_for_instantiation = np.mean(memory_for_instantiation, axis=0)
    std_error_of_memory_for_instantiation = \
        np.std(memory_for_instantiation, axis=0) / np.sqrt(num_repetitions)

    omfr_plot_memory_benchmark_for_instantiation(
        buffer_sizes=buffer_sizes, used_memory=mean_memory_for_instantiation,
        std_error_memory=std_error_of_memory_for_instantiation)


if __name__ == "__main__":
    omfr_benchmark_memory()
