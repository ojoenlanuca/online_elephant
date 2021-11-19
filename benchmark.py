from time import perf_counter_ns
from math import log10, floor
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_generation import homogeneous_poisson_process

from analysis import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient


def create_benchmark_plot(parameter_values, run_times, parameter_name,
                          method_name):
    """
    Creates the benchmark plots of a specified method (MFR, ISI, PCC) for
    an investigated parameter (buffer size, firing rate, buffer count).

    :param parameter_values: list
        values of the investigated parameter
    :param run_times: list
        run time values for the investigated parameter
    :param parameter_name: string
        name of the parameter (format: "Xxxx_Yyyy")
    :param method_name: string
        name of the method as abbreviation (format: "ABC")
    :return: None

    """
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(f"{parameter_name.replace('_', ' ')} Influence on Run Time of"
                 f" Online {method_name}", y=0.93, fontsize=18,
                 fontweight="bold")
    ax1.plot(parameter_values, run_times, label="average times per buffer",
             marker="x", markerfacecolor="red", markeredgecolor='red')
    ax1.hlines(np.mean(run_times), xmin=parameter_values[0],
               xmax=parameter_values[-1], label="mean across buffer sizes",
               colors="orange")
    ax1.tick_params(axis='both', labelsize=14)
    ax1.tick_params(axis='x', direction="in", pad=-22, )
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.yaxis.offsetText.set_visible(False)
    ax1.figure.canvas.draw()
    fmt = ax1.yaxis.get_major_formatter()
    scale_factor = fmt.get_offset()
    ax1.set_xlabel(f"{parameter_name.replace('_', ' ')} in sec", fontsize=16,
                   fontweight="bold")
    ax1.set_ylabel(f"Average Run Time in {scale_factor} sec", fontsize=16,
                   fontweight="bold")
    order_of_magnitude_of_ylim = float(f"1e{floor(log10(max(run_times)))}")
    ax1.set_ylim(min(run_times) - order_of_magnitude_of_ylim,
                 max(run_times) + order_of_magnitude_of_ylim)
    ax1.legend(fontsize=16)
    plt.savefig(f"plots/o{method_name.lower()}_investigate_"
                f"{parameter_name.lower()}.pdf")
    plt.show()


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
                omfr.update_mfr(spike_buffer=st)
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
    create_benchmark_plot(parameter_values=buffer_sizes,
                          run_times=average_times_per_buffer,
                          parameter_name="Buffer_Size", method_name="MFR")


def omfr_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for f in firing_rates:
        average_times_per_buffer.append(bomfr.do_benchmark_omfr(
            buffer_size=1, num_buffers=100, firing_rate=f))
    create_benchmark_plot(parameter_values=firing_rates,
                          run_times=average_times_per_buffer,
                          parameter_name="Firing_Rate", method_name="MFR")


def omfr_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for nb in num_buffers:
        average_times_per_buffer.append(bomfr.do_benchmark_omfr(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    create_benchmark_plot(parameter_values=num_buffers,
                          run_times=average_times_per_buffer,
                          parameter_name="Buffer_Count", method_name="MFR")


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
                oisi.update_isi(spike_buffer=st, mode="histogram")
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
    create_benchmark_plot(parameter_values=buffer_sizes,
                          run_times=average_times_per_buffer,
                          parameter_name="Buffer_Size", method_name="ISI")


def oisi_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for f in firing_rates:
        average_times_per_buffer.append(boisi.do_benchmark_oisi(
            buffer_size=1, num_buffers=100, firing_rate=f))
    create_benchmark_plot(parameter_values=firing_rates,
                          run_times=average_times_per_buffer,
                          parameter_name="Firing_Rate", method_name="ISI")


def oisi_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for nb in num_buffers:
        average_times_per_buffer.append(boisi.do_benchmark_oisi(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    create_benchmark_plot(parameter_values=num_buffers,
                          run_times=average_times_per_buffer,
                          parameter_name="Buffer_Count", method_name="ISI")


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
                opcc.update_pcc(binned_spike_buffer=binned_st)
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
    create_benchmark_plot(parameter_values=buffer_sizes,
                          run_times=average_times_per_buffer,
                          parameter_name="Buffer_Size", method_name="PCC")


def opcc_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=10)
    for f in firing_rates:
        average_times_per_buffer.append(bopcc.do_benchmark_opcc(
            buffer_size=1, num_buffers=100, firing_rate=f))
    create_benchmark_plot(parameter_values=firing_rates,
                          run_times=average_times_per_buffer,
                          parameter_name="Firing_Rate", method_name="PCC")


def opcc_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=10)
    for nb in num_buffers:
        average_times_per_buffer.append(bopcc.do_benchmark_opcc(
            buffer_size=1, num_buffers=nb, firing_rate=50))
    create_benchmark_plot(parameter_values=num_buffers,
                          run_times=average_times_per_buffer,
                          parameter_name="Buffer_Count", method_name="PCC")


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
