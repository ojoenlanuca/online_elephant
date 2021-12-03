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


def omfr_benchmark_memory_at_instantiation(num_repetitions=100):
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


def trace_memory_usage_during_simulation_of_MFR_ISI_PCC(
        num_buffers=100, buffer_size=1, firing_rate=50):
    omfr_memory_usage = []
    oisi_memory_usage = []
    opcc_memory_usage = []
    st_list = [homogeneous_poisson_process(
                    firing_rate*pq.Hz, t_start=buffer_size*i*pq.s,
                    t_stop=(buffer_size*i+buffer_size)*pq.s).magnitude
               for i in range(num_buffers)]
    tracemalloc.start()
    omfr = OnlineMeanFiringRate()
    oisi = OnlineInterSpikeInterval()
    opcc = OnlinePearsonCorrelationCoefficient()
    for i in range(num_buffers):
        omfr.update_mfr(spike_buffer=st_list[i])
        oisi.update_isi(spike_buffer=st_list[i])
        opcc.update_pcc(spike_buffer1=st_list[i], spike_buffer2=st_list[i])
        snapshot = tracemalloc.take_snapshot()
        # filter OMFR trace
        omfr_filter_instantiation = tracemalloc.Filter(
            inclusive=True, filename_pattern="*/benchmark_memory.py",
            lineno=_get_line_number(omfr))
        omfr_filter_update = tracemalloc.Filter(
            inclusive=True, filename_pattern="*/benchmark_memory.py",
            lineno=_get_line_number(omfr)+4)
        omfr_snapshot_filtered = snapshot.filter_traces(
            filters=[omfr_filter_instantiation, omfr_filter_update])
        omfr_filterd_stats = omfr_snapshot_filtered.statistics("lineno")
        omfr_memory_usage.append((omfr_filterd_stats[0].size,
                                  omfr_filterd_stats[1].size))
        # filter OISI trace
        oisi_filter_instantiation = tracemalloc.Filter(
            inclusive=True, filename_pattern="*/benchmark_memory.py",
            lineno=_get_line_number(oisi))
        oisi_filter_update = tracemalloc.Filter(
            inclusive=True, filename_pattern="*/benchmark_memory.py",
            lineno=_get_line_number(oisi)+4)
        oisi_snapshot_filtered = snapshot.filter_traces(
            filters=[oisi_filter_instantiation, oisi_filter_update])
        oisi_filterd_stats = oisi_snapshot_filtered.statistics("lineno")
        oisi_memory_usage.append((oisi_filterd_stats[0].size,
                                  oisi_filterd_stats[1].size))
        # filter OPCC trace
        opcc_filter_instantiation = tracemalloc.Filter(
            inclusive=True, filename_pattern="*/benchmark_memory.py",
            lineno=_get_line_number(opcc))
        opcc_filter_update = tracemalloc.Filter(
            inclusive=True, filename_pattern="*/benchmark_memory.py",
            lineno=_get_line_number(opcc)+4)
        opcc_snapshot_filtered = snapshot.filter_traces(
            filters=[opcc_filter_instantiation, opcc_filter_update])
        opcc_filterd_stats = opcc_snapshot_filtered.statistics("lineno")
        opcc_memory_usage.append((opcc_filterd_stats[0].size,
                                  opcc_filterd_stats[1].size))
    tracemalloc.stop()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"Memory Usage of Online MFR, ISI and PCC",
                 y=0.93, fontsize=22)
    # plot MFR memory usage results
    ax1.plot(np.arange(0, num_buffers), np.asarray(omfr_memory_usage)[:, 0],
             color="lightblue", marker="o", markerfacecolor="lightblue",
             markeredgecolor='black', label="OMFR @ instance")
    ax1.plot(np.arange(0, num_buffers), np.asarray(omfr_memory_usage)[:, 1],
             color="blue", marker="o", markerfacecolor="blue",
             markeredgecolor='black', label="OMFR @ update")
    # plot ISI memory usage results
    ax1.plot(np.arange(0, num_buffers), np.asarray(oisi_memory_usage)[:, 0],
             color="lightcoral", marker="o", markerfacecolor="lightcoral",
             markeredgecolor='black', label="OISI @ instance")
    ax1.plot(np.arange(0, num_buffers), np.asarray(oisi_memory_usage)[:, 1],
             color="red", marker="o", markerfacecolor="red",
             markeredgecolor='black', label="OISI @ update")
    # plot PCC memory usage results
    ax1.plot(np.arange(0, num_buffers), np.asarray(opcc_memory_usage)[:, 0],
             color="lightgreen", marker="o", markerfacecolor="lightgreen",
             markeredgecolor='black', label="OPCC @ instance")
    ax1.plot(np.arange(0, num_buffers), np.asarray(opcc_memory_usage)[:, 1],
             color="green", marker="o", markerfacecolor="green",
             markeredgecolor='black', label="OPCC @ update")

    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel(f"Number of Buffers", fontsize=22)
    ax1.set_ylabel(f"Memory Usage [Bytes]", fontsize=22)
    plt.legend(fontsize=18)
    plt.savefig(f"plots/trace_memory_usage_during_simulation_"
                f"of_MFR_ISI_PCC.pdf")
    plt.show()


def _get_line_number(instance):
    """Returns the line number of code, where an object instance was created /
    it's used memory was allocated."""
    location = tracemalloc.get_object_traceback(instance).format()
    line_number = int(location[0].split(",")[1].split(" ")[2])
    return line_number


if __name__ == "__main__":
    # omfr_benchmark_memory_at_instantiation()
    trace_memory_usage_during_simulation_of_MFR_ISI_PCC()
