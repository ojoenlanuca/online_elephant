from math import log10, floor
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
from elephant.spike_train_generation import homogeneous_poisson_process

from online_statistics import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient


def create_benchmark_plot(parameter_values, run_times, parameter_name,
                          method_name, panel_label, xaxis_unit,
                          std_error, buffer_size):
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
    :param xaxis_unit: string
        SI-Unit of the x-axis (e.g. 's', 'Hz')
    :param panel_label: string
        label of the panel (e.g. 'A', 'B', 'C')
    :param std_error: list of floats
        list of standard error values of the runtimes per buffer
    :param buffer_size: float or list of floats
        size(s) of the buffer
    :return: None

    """
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"{parameter_name.replace('_', ' ')} Influence on Run Time of"
                 f" Online {method_name}", y=0.93, fontsize=22)
    # rescale runtimes to μs and accordingly rescale the respective std_error
    run_times = np.multiply(run_times, 1e6)
    std_error = np.multiply(std_error, 1e6)
    ax1.errorbar(parameter_values, run_times, yerr=std_error, ecolor='red',
                 marker="o", markerfacecolor="blue", markeredgecolor='black',
                 color="blue", label="average times per buffer")
    ax1.text(-0.07, 1.07, panel_label, transform=ax1.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='right')
    ax1.tick_params(axis='both', labelsize=18)
    ax1.tick_params(axis='y',  labelcolor='blue', labelsize=18)
    ax1.set_xlabel(f"{parameter_name.replace('_', ' ')} {xaxis_unit}",
                   fontsize=22)
    ax1.set_ylabel(f"Average Run Time [μs]", color='blue', fontsize=22)
    order_of_magnitude_of_y1_lim = float(f"1e{floor(log10(max(run_times)))}")
    ax1.set_ylim(min(run_times)-order_of_magnitude_of_y1_lim,
                 max(run_times)+order_of_magnitude_of_y1_lim)
    # display estimated number of neurons, which can be processed sequentially
    # based on the average run times per buffer
    ax2 = ax1.twinx()
    n_neurons_per_buffer = np.divide(buffer_size, np.multiply(run_times, 1e-6))
    ax2.plot(parameter_values, n_neurons_per_buffer, color='green',
             marker="o", markerfacecolor="green", markeredgecolor='black',
             label="number of neurons")
    ax2.tick_params(axis='y', labelcolor='green', labelsize=18)
    ax2.set_ylim(0, max(n_neurons_per_buffer)*1.25)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.yaxis.offsetText.set_visible(False)
    ax2.figure.canvas.draw()
    fmt2 = ax2.yaxis.get_major_formatter()
    scale_factor2 = fmt2.get_offset()
    ax2.set_ylabel(f'Estimated Number of Neurons [{scale_factor2}]',
                   color='green', fontsize=22)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, fontsize=18)
    plt.savefig(f"plots/o{method_name.lower()}_investigate_"
                f"{parameter_name.lower()}.pdf")
    plt.show()


class BenchmarkOnlineMeanFiringRate:
    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def do_benchmark_omfr(self, buffer_size, num_buffers, firing_rate):
        repetition_runtimes_per_buffer = []
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
            repetition_runtimes_per_buffer.append(
                sum(buffer_runtimes) / num_buffers)

        # calculate average runtime per buffer and standard error
        mean_runtime_per_buffer = np.mean(repetition_runtimes_per_buffer)
        std_error_of_runtimes_per_buffer = \
            np.std(repetition_runtimes_per_buffer)/np.sqrt(self.num_repetitions)
        print(f"average runtime per buffer for online_mfr: "
              f"{mean_runtime_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return mean_runtime_per_buffer, std_error_of_runtimes_per_buffer


def omfr_investigate_buffer_size():
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for b in buffer_sizes:
        mean, std_error = bomfr.do_benchmark_omfr(
            buffer_size=b, num_buffers=100, firing_rate=50)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=buffer_sizes, run_times=average_times_per_buffer,
        parameter_name="Buffer_Duration", method_name="MFR", panel_label="A",
        xaxis_unit="[s]", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=buffer_sizes)


def omfr_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for f in firing_rates:
        mean, std_error = bomfr.do_benchmark_omfr(
            buffer_size=1, num_buffers=100, firing_rate=f)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=firing_rates, run_times=average_times_per_buffer,
        parameter_name="Firing_Rate", method_name="MFR", panel_label="B",
        xaxis_unit="[Hz]", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=1)


def omfr_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    bomfr = BenchmarkOnlineMeanFiringRate(num_repetitions=100)
    for nb in num_buffers:
        mean, std_error = bomfr.do_benchmark_omfr(
            buffer_size=1, num_buffers=nb, firing_rate=50)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=num_buffers, run_times=average_times_per_buffer,
        parameter_name="Buffer_Count", method_name="MFR", panel_label="C",
        xaxis_unit="", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=1)


class BenchmarkOnlineInterSpikeInterval:
    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def do_benchmark_oisi(self, buffer_size, num_buffers, firing_rate):
        repetition_runtimes_per_buffer = []
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
                oisi.update_isi(spike_buffer=st)
                toc1 = perf_counter_ns()
                buffer_runtimes.append((toc1-tic1)*1e-9)

            # add sum of buffer_runtimes to repetition_runtimes list
            repetition_runtimes_per_buffer.append(
                sum(buffer_runtimes) / num_buffers)

        # calculate average runtime per buffer and standard error
        mean_runtime_per_buffer = np.mean(repetition_runtimes_per_buffer)
        std_error_of_runtimes_per_buffer = \
            np.std(repetition_runtimes_per_buffer)/np.sqrt(self.num_repetitions)
        print(f"average runtime per buffer for online_isi: "
              f"{mean_runtime_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return mean_runtime_per_buffer, std_error_of_runtimes_per_buffer


def oisi_investigate_buffer_size():
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for b in buffer_sizes:
        mean, std_error = boisi.do_benchmark_oisi(
            buffer_size=b, num_buffers=100, firing_rate=50)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=buffer_sizes, run_times=average_times_per_buffer,
        parameter_name="Buffer_Duration", method_name="ISI", panel_label="A",
        xaxis_unit="[s]", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=buffer_sizes)


def oisi_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for f in firing_rates:
        mean, std_error = boisi.do_benchmark_oisi(
            buffer_size=1, num_buffers=100, firing_rate=f)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=firing_rates, run_times=average_times_per_buffer,
        parameter_name="Firing_Rate", method_name="ISI", panel_label="B",
        xaxis_unit="[Hz]", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=1)


def oisi_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    boisi = BenchmarkOnlineInterSpikeInterval(num_repetitions=100)
    for nb in num_buffers:
        mean, std_error = boisi.do_benchmark_oisi(
            buffer_size=1, num_buffers=nb, firing_rate=50)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=num_buffers, run_times=average_times_per_buffer,
        parameter_name="Buffer_Count", method_name="ISI", panel_label="C",
        xaxis_unit="", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=1)


class BenchmarkOnlinePearsonCorrelationCoefficient:
    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def do_benchmark_opcc(self, buffer_size, num_buffers, firing_rate):
        repetition_runtimes_per_buffer = []
        for r in range(self.num_repetitions):
            # simulate buffered reading/transport of spiketrains,
            # i.e. create binned spiketrain and call calculate_pcc()
            opcc = OnlinePearsonCorrelationCoefficient(buffer_size=buffer_size)
            buffer_runtimes = []
            for i in range(num_buffers):
                # create BinnedSpiketrain from single spiketrains
                st1 = homogeneous_poisson_process(
                    firing_rate * pq.Hz, t_start=buffer_size * i * pq.s,
                    t_stop=(buffer_size * i + buffer_size) * pq.s)
                st2 = homogeneous_poisson_process(
                    firing_rate * pq.Hz, t_start=buffer_size * i * pq.s,
                    t_stop=(buffer_size * i + buffer_size) * pq.s)
                # measure runtime for one buffer
                tic1 = perf_counter_ns()
                opcc.update_pcc(spike_buffer1=st1, spike_buffer2=st2)
                toc1 = perf_counter_ns()
                buffer_runtimes.append((toc1-tic1)*1e-9)

            # add sum of buffer_runtimes to repetition_runtimes list
            repetition_runtimes_per_buffer.append(
                sum(buffer_runtimes) / num_buffers)

        # calculate average runtime per buffer and standard error
        mean_runtime_per_buffer = np.mean(repetition_runtimes_per_buffer)
        std_error_of_runtimes_per_buffer = \
            np.std(repetition_runtimes_per_buffer)/np.sqrt(self.num_repetitions)
        print(f"average runtime per buffer for online_pcc: "
              f"{mean_runtime_per_buffer}sec\n"
              f"-> with buffer_size={buffer_size}, firing_rate={firing_rate}, "
              f"number of buffers={num_buffers}")
        return mean_runtime_per_buffer, std_error_of_runtimes_per_buffer


def opcc_investigate_buffer_size():
    buffer_sizes = [0.1, 0.25, 0.5, 0.75, 1, 2, 4, 6, 8, 10]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=100)
    for b in buffer_sizes:
        mean, std_error = bopcc.do_benchmark_opcc(
            buffer_size=b, num_buffers=100, firing_rate=50)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=buffer_sizes, run_times=average_times_per_buffer,
        parameter_name="Buffer_Duration", method_name="PCC", panel_label="A",
        xaxis_unit="[s]", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=buffer_sizes)


def opcc_investigate_firing_rate():
    firing_rates = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=100)
    for f in firing_rates:
        mean, std_error = bopcc.do_benchmark_opcc(
            buffer_size=1, num_buffers=100, firing_rate=f)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=firing_rates, run_times=average_times_per_buffer,
        parameter_name="Firing_Rate", method_name="PCC", panel_label="B",
        xaxis_unit="[Hz]", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=1)


def opcc_investigate_number_of_buffers():
    num_buffers = [2, 5, 10, 50, 100, 250, 500, 750, 1000]
    average_times_per_buffer = []
    std_error_of_runtimes_per_buffer = []
    bopcc = BenchmarkOnlinePearsonCorrelationCoefficient(num_repetitions=100)
    for nb in num_buffers:
        mean, std_error = bopcc.do_benchmark_opcc(
            buffer_size=1, num_buffers=nb, firing_rate=50)
        average_times_per_buffer.append(mean)
        std_error_of_runtimes_per_buffer.append(std_error)
    create_benchmark_plot(
        parameter_values=num_buffers, run_times=average_times_per_buffer,
        parameter_name="Buffer_Count", method_name="PCC", panel_label="C",
        xaxis_unit="", std_error=std_error_of_runtimes_per_buffer,
        buffer_size=1)


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
