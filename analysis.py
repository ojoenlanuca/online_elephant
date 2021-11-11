import numpy as np
import quantities as pq
from elephant.statistics import mean_firing_rate, isi


class OnlineMeanFiringRate:
    def __init__(self):
        self.current_mfr = 0*pq.Hz
        self.buffer_counter = 0

    def calculate_mfr(self, spike_buffer):
        """
        Calculates the mean firing rate of a single neuron.

        :param spike_buffer: neo.Spiketrain
            contains the spikes of the neuron

        :return self.current_mfr: float
            mean firing rate of a neuron

        Notes:
        recurrent formula for sample mean:
        x'_bar = x_bar + (x_n+1 - x_bar)/(n+1)
        """
        buffer_mfr = mean_firing_rate(spike_buffer).rescale(pq.Hz)
        self.current_mfr += (buffer_mfr-self.current_mfr)/(self.buffer_counter+1)
        self.buffer_counter += 1
        return self.current_mfr


class OnlineInterSpikeInterval:
    def __init__(self):
        self.last_spike_time_of_previous_buffer = None
        self.current_isi = [] # in sec
        self.bin_size = 0.0005 # in sec
        self.bins_edges = np.asarray([self.bin_size * i for i in range(int(1 / self.bin_size))])
        self.current_isi_histogram = np.empty(shape=(self.bins_edges.shape[0]-1))

    def calculate_isi(self, spike_buffer, mode="raw"):
        """
        Calculates the inter-spike interval of one single neuron.

        :param spike_buffer: neo.Spiketrain
            contains the spikes of the neuron
        :param mode: "raw" or "histogram"
            ISI values are either returned as list or histogram with bin_edges
        :return
            - self.current_isi: list, if mode="raw"
            list of all ISI values of one neuron, which were calculated
            until now (unit: sec)
            - self.current_isi_histogram, self.bin_edges: both numpy.ndarray,
                if mode="histogram"
            histogram of the current ISI values according to the defined
            bin_edges

        Notes:
        if spike_buffer is empty, nothing will change and therefore nothing will
        be returned

        """
        if spike_buffer.size:  # case1: spike_buffer not empty
            if self.last_spike_time_of_previous_buffer:  # for second till last buffer
                buffer_isi = isi(np.append(self.last_spike_time_of_previous_buffer, spike_buffer))
                self.last_spike_time_of_previous_buffer = spike_buffer[-1]
            else:  # for first buffer
                buffer_isi = isi(spike_buffer)
                self.last_spike_time_of_previous_buffer = spike_buffer[-1]
            if mode == "raw":
                self.current_isi.extend(buffer_isi.magnitude.tolist())
            elif mode == "histogram":
                buffer_hist, _ = np.histogram(buffer_isi, bins=self.bins_edges)
                self.current_isi_histogram += buffer_hist
        else:  # case2: spike_buffer is empty
            pass
        if mode == "raw":
            return self.current_isi
        elif mode == "histogram":
            return self.current_isi_histogram


class OnlinePearsonCorrelationCoefficient:
    def __init__(self):
        self.buffer_counter = 0
        self.x_bar = 0  # mean spikes per bin
        self.y_bar = 0  # mean spikes per bin
        self.M_x = 0  # sum_i=1_to_n (x_i - x_bar)^2
        self.M_y = 0  # sum_i=1_to_n (y_i - y_bar)^2
        self.C_s = 0  # sum_i=1_to_n (x_i - x_bar)(y_i - y_bar)
        self.R_xy = 0  # C_s / (sqrt(M_x) * sqrt(M_y))

    def calculate_pcc(self, binned_spike_buffer):
        """
        Calculates Pearson's Correlation Coefficient between two neurons.

        :param spike_buffer: elephant.conversion.BinnedSpikeTrain
            contains one binned spiketrain for each neuron of user defined
            binsize
        :return self.R_xy: float
            Pearson's correlation coefficient of two neurons, range: [-1, 1]

        Notes: update formulas for M_x, M_y and C_s, also x_bar and y_bar:
        x'_bar = x_bar + (x_n+1 - x_bar)/(n+1)
        y'_bar = y_bar + (y_n+1 - y_bar)/(n+1)
        M_x' = M_x + (x_n+1 - x_bar)(x_n+1 - x'_bar)
        M_y' = M_y + (y_n+1 - y_bar)(y_n+1 - y'_bar)
        C_s' = C_s + n/(n+1) * (x_n+1 - x_bar)(y_n+1 - y_bar)

        Reference:
        this implementation of Pearson's correlation coefficient is based on
        the 'incremental one pass approach' of:

        Socha, Petr, Vojtěch Miškovský, Hana Kubátová, und Martin Novotný.
        „Optimization of Pearson correlation coefficient calculation for DPA
        and comparison of different approaches“. In 2017 IEEE 20th
        International Symposium on Design and Diagnostics of
        Electronic Circuits Systems (DDECS), 184–89, 2017.
        https://doi.org/10.1109/DDECS.2017.7934563.
        """
        spike_count_array = binned_spike_buffer.to_array() # spike count per bin per neuron
        for b in range(binned_spike_buffer.num_bins):
            x_n_plus_1 = spike_count_array[0][b]
            y_n_plus_1 = spike_count_array[1][b]
            n = self.buffer_counter * binned_spike_buffer.num_bins + b  # number of bins in total
            self.C_s += n / (n + 1) * (x_n_plus_1 - self.x_bar) * (y_n_plus_1 -self.y_bar)
            x_bar_new = self.x_bar + (x_n_plus_1 - self.x_bar) / (n + 1)
            y_bar_new = self.y_bar + (y_n_plus_1 - self.y_bar) / (n + 1)
            self.M_x += (x_n_plus_1 - self.x_bar) * (x_n_plus_1 - x_bar_new)
            self.M_y += (y_n_plus_1 - self.y_bar) * (y_n_plus_1 - y_bar_new)
            self.x_bar = x_bar_new
            self.y_bar = y_bar_new
        self.R_xy = self.C_s / (np.sqrt(self.M_x * self.M_y))
        self.buffer_counter += 1
        return self.R_xy