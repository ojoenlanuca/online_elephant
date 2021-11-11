from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_generation import homogeneous_poisson_process

from analysis import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient

from multiprocessing import Pool


class BenchmarkOnlineMeanFiringRate:
    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def do_benchmark_omfr(self, buffer_size, num_buffers, firing_rate):
        repetition_runtimes = []
        for r in range(self.num_repetitions):
            # simulate buffered reading/transport of spiketrains,
            # i.e. create spiketrain in loop and call calculate_mfr()
            omfr = OnlineMeanFiringRate()
            buffer_runtimes = []
            for i in range(num_buffers):
                # create spiketrain
                st = homogeneous_poisson_process(
                    firing_rate*pq.Hz, t_start=buffer_size*i*pq.s,
                    t_stop=(buffer_size*i+buffer_size)*pq.s)
                # measure runtime for one buffer
                tic1 = perf_counter_ns()
                omfr.calculate_mfr(spike_buffer=st)
                toc1 = perf_counter_ns()
                buffer_runtimes.append((toc1-tic1)*1e-9)

            # add sum of buffer_runtimes to repetition_runtime list
            repetition_runtimes.append(sum(buffer_runtimes))

        # calculate average runtime per buffer
        average_time_per_buffer = np.mean(repetition_runtimes) / num_buffers
        print(f"average runtime per buffer for online_mfr: "
              f"{average_time_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return average_time_per_buffer


def omfr_investigate_buffer_size():
    buffer_sizes = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    average_times_per_buffer = []
    BOMFR10 = BenchmarkOnlineMeanFiringRate(10)
    for b in buffer_sizes:
        average_times_per_buffer.append(BOMFR10.do_benchmark_omfr(
            buffer_size=b, num_buffers=100, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer size influence on runtime of online MFR")
    ax1.semilogx(buffer_sizes, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=buffer_sizes[0],
               xmax=buffer_sizes[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("buffer size in sec")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.0001,
                 max(average_times_per_buffer)+0.0001)
    ax1.legend()
    plt.savefig("plots/omfr_investigate_buffer_size.svg")
    plt.show()


def omfr_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    average_times_per_buffer = []
    BOMFR10 = BenchmarkOnlineMeanFiringRate(10)
    for f in firing_rates:
        average_times_per_buffer.append(BOMFR10.do_benchmark_omfr(
            buffer_size=1, num_buffers=100, firing_rate=f))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Firing rate influence on runtime of online MFR")
    ax1.semilogx(firing_rates, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=firing_rates[0],
               xmax=firing_rates[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("firing rate in Hz")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.0001,
                 max(average_times_per_buffer)+0.0001)
    ax1.legend()
    plt.savefig("plots/omfr_investigate_firing_rate.svg")
    plt.show()


def omfr_investigate_number_of_buffers():
    num_buffers = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    average_times_per_buffer = []
    BOMFR10 = BenchmarkOnlineMeanFiringRate(10)
    for nb in num_buffers:
        average_times_per_buffer.append(BOMFR10.do_benchmark_omfr(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer count influence on runtime of online MFR")
    ax1.semilogx(num_buffers, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=num_buffers[0],
               xmax=num_buffers[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("number of buffers")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.0001,
                 max(average_times_per_buffer)+0.0001)
    ax1.legend()
    plt.savefig("plots/omfr_investigate_number_of_buffers.svg")
    plt.show()


class BenchmarkOnlineInterSpikeInterval:
    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def do_benchmark_oisi(self, buffer_size, num_buffers, firing_rate):
        repetition_runtimes = []
        for r in range(self.num_repetitions):
            # simulate buffered reading/transport of spiketrains,
            # i.e. create spiketrain in loop and call calculate_isi()
            oisi = OnlineInterSpikeInterval()
            buffer_runtimes = []
            for i in range(num_buffers):
                # create spiketrain
                st = homogeneous_poisson_process(
                    firing_rate*pq.Hz, t_start=buffer_size*i*pq.s,
                    t_stop=(buffer_size*i+buffer_size)*pq.s)
                # measure runtime for one buffer
                tic1 = perf_counter_ns()
                oisi.calculate_isi(spike_buffer=st, mode="histogram")
                toc1 = perf_counter_ns()
                buffer_runtimes.append((toc1-tic1)*1e-9)

            # add sum of buffer_runtimes to repetition_runtimes list
            repetition_runtimes.append(sum(buffer_runtimes))

        # calculate average runtime per buffer
        average_time_per_buffer = np.mean(repetition_runtimes) / num_buffers
        print(f"average runtime per buffer for online_isi: "
              f"{average_time_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return average_time_per_buffer


def oisi_investigate_buffer_size():
    buffer_sizes = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    average_times_per_buffer = []
    BOISI10 = BenchmarkOnlineInterSpikeInterval(10)
    for b in buffer_sizes:
        average_times_per_buffer.append(BOISI10.do_benchmark_oisi(
            buffer_size=b, num_buffers=100, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer size influence on runtime of online ISI")
    ax1.semilogx(buffer_sizes, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=buffer_sizes[0],
               xmax=buffer_sizes[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("buffer size in sec")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.0001,
                 max(average_times_per_buffer)+0.0001)
    ax1.legend()
    plt.savefig("plots/oisi_investigate_buffer_size.svg")
    plt.show()


def oisi_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    average_times_per_buffer = []
    BOISI10 = BenchmarkOnlineInterSpikeInterval(10)
    for f in firing_rates:
        average_times_per_buffer.append(BOISI10.do_benchmark_oisi(
            buffer_size=1, num_buffers=100, firing_rate=f))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Firing Rate influence on runtime of online ISI")
    ax1.semilogx(firing_rates, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=firing_rates[0],
               xmax=firing_rates[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("firing rate in Hz")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.0001,
                 max(average_times_per_buffer)+0.0001)
    ax1.legend()
    plt.savefig("plots/oisi_investigate_firing_rate.svg")
    plt.show()


def oisi_investigate_number_of_buffers():
    num_buffers = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    average_times_per_buffer = []
    BOISI10 = BenchmarkOnlineInterSpikeInterval(10)
    for nb in num_buffers:
        average_times_per_buffer.append(BOISI10.do_benchmark_oisi(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer count influence on runtime of online ISI")
    ax1.semilogx(num_buffers, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=num_buffers[0],
               xmax=num_buffers[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("number of buffers")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.0001,
                 max(average_times_per_buffer)+0.0001)
    ax1.legend()
    plt.savefig("plots/oisi_investigate_number_of_buffers.svg")
    plt.show()


class BenchmarkOnlinePearsonCorrelationCoefficient:
    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def do_benchmark_opcc(self, buffer_size, num_buffers, firing_rate):
        repetition_runtimes = []
        for r in range(self.num_repetitions):
            # simulate buffered reading/transport of spiketrains,
            # i.e. create binned spiketrain and call calculate_pcc()
            opcc = OnlinePearsonCorrelationCoefficient()
            buffer_runtimes = []
            for i in range(num_buffers):
                # create BinnedSpiketrain from single spiketrains
                st1 = homogeneous_poisson_process(
                    firing_rate * pq.Hz, t_start=buffer_size * i * pq.s,
                    t_stop=(buffer_size * i + buffer_size) * pq.s)
                st2 = homogeneous_poisson_process(
                    firing_rate * pq.Hz, t_start=buffer_size * i * pq.s,
                    t_stop=(buffer_size * i + buffer_size) * pq.s)
                binned_st = BinnedSpikeTrain([st1, st2], bin_size=5 * pq.ms)
                # measure runtime for one buffer
                tic1 = perf_counter_ns()
                opcc.calculate_pcc(binned_spike_buffer=binned_st)
                toc1 = perf_counter_ns()
                buffer_runtimes.append((toc1-tic1)*1e-9)

            # add sum of buffer_runtimes to repetition_runtimes list
            repetition_runtimes.append(sum(buffer_runtimes))

        # calculate average runtime per buffer
        average_time_per_buffer = np.mean(repetition_runtimes) / num_buffers
        print(f"average runtime per buffer for online_pcc: "
              f"{average_time_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return average_time_per_buffer


def opcc_investigate_buffer_size():
    buffer_sizes = [0.1, 0.5, 1, 5, 10, 50]
    #, 100, 500, 1000] other values take to long to compute
    average_times_per_buffer = []
    BOPCC10 = BenchmarkOnlinePearsonCorrelationCoefficient(10)
    for b in buffer_sizes:
        average_times_per_buffer.append(BOPCC10.do_benchmark_opcc(
            buffer_size=b, num_buffers=100, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer size influence on runtime of online PCC")
    ax1.semilogx(buffer_sizes, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=buffer_sizes[0],
               xmax=buffer_sizes[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("buffer size in sec")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.01,
                 max(average_times_per_buffer)+0.01)
    ax1.legend()
    plt.savefig("plots/opcc_investigate_buffer_size.svg")
    plt.show()


def opcc_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    average_times_per_buffer = []
    BOPCC10 = BenchmarkOnlinePearsonCorrelationCoefficient(10)
    for f in firing_rates:
        average_times_per_buffer.append(BOPCC10.do_benchmark_opcc(
            buffer_size=1, num_buffers=100, firing_rate=f))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Firing rate influence on runtime of online PCC")
    ax1.semilogx(firing_rates, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=firing_rates[0],
               xmax=firing_rates[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("firing rate in Hz")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.001,
                 max(average_times_per_buffer)+0.001)
    ax1.legend()
    plt.savefig("plots/opcc_investigate_firing_rate.svg")
    plt.show()


def opcc_investigate_number_of_buffers():
    num_buffers = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    average_times_per_buffer = []
    BOPCC10 = BenchmarkOnlinePearsonCorrelationCoefficient(10)
    for nb in num_buffers:
        average_times_per_buffer.append(BOPCC10.do_benchmark_opcc(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer count influence on runtime of online PCC")
    ax1.semilogx(num_buffers, average_times_per_buffer,
                 label="average times per buffer", marker="x",
                 markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=num_buffers[0],
               xmax=num_buffers[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("number of buffers")
    ax1.set_ylabel("average runtime in sec")
    ax1.set_ylim(min(average_times_per_buffer)-0.001,
                 max(average_times_per_buffer)+0.001)
    ax1.legend()
    plt.savefig("plots/opcc_investigate_number_of_buffers.svg")
    plt.show()


if __name__ == '__main__':
    # MFR benchmarks
    # omfr_investigate_buffer_size()
    # omfr_investigate_firing_rate()
    # omfr_investigate_number_of_buffers()

    # ISI benchmarks
    # oisi_investigate_buffer_size()
    # oisi_investigate_firing_rate()
    # oisi_investigate_number_of_buffers()

    # PCC benchmarks
    # opcc_investigate_buffer_size()
    # opcc_investigate_firing_rate()
    # opcc_investigate_number_of_buffers()
    pass
