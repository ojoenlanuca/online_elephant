from time import perf_counter_ns
from math import log10, floor
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_generation import homogeneous_poisson_process

from analysis import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient


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
                # create spiketrain as np.ndarray
                st = homogeneous_poisson_process(
                    firing_rate*pq.Hz, t_start=buffer_size*i*pq.s,
                    t_stop=(buffer_size*i+buffer_size)*pq.s).magnitude
                # measure runtime for one buffer
                tic1 = perf_counter_ns()
                omfr.calculate_mfr(spike_buffer=st)
                toc1 = perf_counter_ns()
                buffer_runtimes.append((toc1-tic1)*1e-9)

            # add sum of buffer_runtimes to repetition_runtime list
            repetition_runtimes.append(sum(buffer_runtimes))

        # calculate average runtime per buffer considering all repetitions
        average_time_per_buffer = np.mean(repetition_runtimes) / num_buffers
        print(f"average runtime per buffer for online_mfr: "
              f"{average_time_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return average_time_per_buffer


def omfr_investigate_buffer_size():
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]
    average_times_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for b in buffer_sizes:
        average_times_per_buffer.append(bomfr.do_benchmark_omfr(
            buffer_size=b, num_buffers=100, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer size influence on runtime of online MFR")
    ax1.plot(buffer_sizes, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=buffer_sizes[0],
               xmax=buffer_sizes[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("buffer size in sec")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/omfr_investigate_buffer_size.svg")
    plt.show()


def omfr_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for f in firing_rates:
        average_times_per_buffer.append(bomfr.do_benchmark_omfr(
            buffer_size=1, num_buffers=100, firing_rate=f))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Firing rate influence on runtime of online MFR")
    ax1.plot(firing_rates, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=firing_rates[0],
               xmax=firing_rates[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("firing rate in Hz")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/omfr_investigate_firing_rate.svg")
    plt.show()


def omfr_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for nb in num_buffers:
        average_times_per_buffer.append(bomfr.do_benchmark_omfr(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer count influence on runtime of online MFR")
    ax1.plot(num_buffers, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=num_buffers[0],
               xmax=num_buffers[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("number of buffers")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
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
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]
    average_times_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for b in buffer_sizes:
        average_times_per_buffer.append(boisi.do_benchmark_oisi(
            buffer_size=b, num_buffers=100, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer size influence on runtime of online ISI")
    ax1.plot(buffer_sizes, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=buffer_sizes[0],
               xmax=buffer_sizes[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("buffer size in sec")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/oisi_investigate_buffer_size.svg")
    plt.show()


def oisi_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for f in firing_rates:
        average_times_per_buffer.append(boisi.do_benchmark_oisi(
            buffer_size=1, num_buffers=100, firing_rate=f))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Firing Rate influence on runtime of online ISI")
    ax1.plot(firing_rates, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=firing_rates[0],
               xmax=firing_rates[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("firing rate in Hz")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/oisi_investigate_firing_rate.svg")
    plt.show()


def oisi_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for nb in num_buffers:
        average_times_per_buffer.append(boisi.do_benchmark_oisi(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer count influence on runtime of online ISI")
    ax1.plot(num_buffers, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=num_buffers[0],
               xmax=num_buffers[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("number of buffers")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
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
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]
    average_times_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=10)
    for b in buffer_sizes:
        average_times_per_buffer.append(bopcc.do_benchmark_opcc(
            buffer_size=b, num_buffers=100, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer size influence on runtime of online PCC")
    ax1.plot(buffer_sizes, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=buffer_sizes[0],
               xmax=buffer_sizes[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("buffer size in sec")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/opcc_investigate_buffer_size.svg")
    plt.show()


def opcc_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=10)
    for f in firing_rates:
        average_times_per_buffer.append(bopcc.do_benchmark_opcc(
            buffer_size=1, num_buffers=100, firing_rate=f))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Firing rate influence on runtime of online PCC")
    ax1.plot(firing_rates, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=firing_rates[0],
               xmax=firing_rates[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("firing rate in Hz")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/opcc_investigate_firing_rate.svg")
    plt.show()


def opcc_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=10)
    for nb in num_buffers:
        average_times_per_buffer.append(bopcc.do_benchmark_opcc(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Buffer count influence on runtime of online PCC")
    ax1.plot(num_buffers, average_times_per_buffer,
             label="average times per buffer", marker="x",
             markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(average_times_per_buffer), xmin=num_buffers[0],
               xmax=num_buffers[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.set_xlabel("number of buffers")
    ax1.set_ylabel("average runtime in sec")
    order_of_magnitude_of_ylim = \
        float(f"1e{floor(log10(max(average_times_per_buffer)))}")
    ax1.set_ylim(min(average_times_per_buffer)-order_of_magnitude_of_ylim,
                 max(average_times_per_buffer)+order_of_magnitude_of_ylim)
    ax1.legend()
    plt.savefig("plots/opcc_investigate_number_of_buffers.svg")
    plt.show()


if __name__ == '__main__':
    # MFR benchmarks
    # omfr_investigate_buffer_size()
    # omfr_investigate_firing_rate()
    # omfr_investigate_number_of_buffers()
    # omfr_investigate_number_of_neurons()

    # ISI benchmarks
    # oisi_investigate_buffer_size()
    # oisi_investigate_firing_rate()
    # oisi_investigate_number_of_buffers()

    # PCC benchmarks
    # opcc_investigate_buffer_size()
    # opcc_investigate_firing_rate()
    # opcc_investigate_number_of_buffers()
    pass
