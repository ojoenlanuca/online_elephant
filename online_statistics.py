from math import ceil

import neo
import numpy as np
import quantities as pq
from elephant.statistics import mean_firing_rate, isi


class OnlineMeanFiringRate:
    def __init__(self, buffer_size=1):
        self.unit = 1 * pq.Hz
        self._current_mfr = 0  # in  Hz
        self.buffer_counter = 0
        self.buffer_size = buffer_size   # in sec

    @property
    def current_mfr(self):
        return self._current_mfr * self.unit

    def update_mfr(self, spike_buffer):
        """
        Calculates the mean firing rate of a single neuron.

        :param spike_buffer: neo.Spiketrain or np.ndarray
            contains the spiketimes of one neuron
        :return self.current_mfr: pq.Quantity
            mean firing rate of a neuron

        Notes:
        recurrent formula for sample mean:
        x'_bar = x_bar + (x_n+1 - x_bar)/(n+1)
        """
        if len(spike_buffer) == 0:
            buffer_mfr = 0
        else:
            if isinstance(spike_buffer, neo.SpikeTrain):
                buffer_mfr = mean_firing_rate(spike_buffer).rescale(
                    self.unit).magnitude
            if isinstance(spike_buffer, np.ndarray) and \
                    not isinstance(spike_buffer, neo.SpikeTrain):
                buffer_mfr = mean_firing_rate(
                    spike_buffer, t_start=self.buffer_counter *
                                          self.buffer_size,
                    t_stop=self.buffer_counter * self.buffer_size +
                           self.buffer_size)
        self.buffer_counter += 1
        self._current_mfr += \
            (buffer_mfr - self._current_mfr) / self.buffer_counter
        return self._current_mfr * self.unit


class OnlineInterSpikeInterval:
    def __init__(self, bin_size=0.0005, max_isi_value=1):
        self.max_isi_value = max_isi_value  # in sec
        self.last_spike_time = None
        self.bin_size = bin_size  # in sec
        self.num_bins = int(self.max_isi_value / self.bin_size)
        self.bin_edges = np.linspace(start=0, stop=self.max_isi_value,
                                     num=self.num_bins+1)
        self.current_isi_histogram = np.empty(shape=(len(self.bin_edges) - 1))

    def update_isi(self, spike_buffer):
        """
        Calculates the inter-spike interval of a single neuron.

        :param spike_buffer: neo.Spiketrain or np.ndarray
            contains the spiketimes of one neuron
        :return
            self.current_isi_histogram: numpy.ndarray,
                histogram of the current ISI values according to the defined
                bin_edges

        Notes:
        if spike_buffer is empty, an unchanged histogram will be returned

        """
        if spike_buffer.size is not 0:  # case1: spike_buffer not empty
            if self.last_spike_time is not None:
                # from second to last buffer
                buffer_isi = isi(np.append(
                    self.last_spike_time, spike_buffer))
            else:  # for first buffer
                buffer_isi = isi(spike_buffer)
            self.last_spike_time = spike_buffer[-1]
            buffer_hist, _ = np.histogram(buffer_isi, bins=self.bin_edges)
            self.current_isi_histogram += buffer_hist
            return self.current_isi_histogram
        else:  # case2: spike_buffer is empty
            return self.current_isi_histogram


class OnlinePearsonCorrelationCoefficient:
    def __init__(self, buffer_size=1, bin_size=0.005):
        self.bin_size = bin_size  # in sec
        self.buffer_size = buffer_size  # in sec
        self.buffer_counter = 0
        self.x_bar = 0  # mean spikes per bin
        self.y_bar = 0  # mean spikes per bin
        self.M_x = 0  # sum_i=1_to_n (x_i - x_bar)^2
        self.M_y = 0  # sum_i=1_to_n (y_i - y_bar)^2
        self.C_s = 0  # sum_i=1_to_n (x_i - x_bar)(y_i - y_bar)
        self.R_xy = 0  # C_s / (sqrt(M_x) * sqrt(M_y))

    def update_pcc(self, spike_buffer1, spike_buffer2):
        """
        Calculates Pearson's Correlation Coefficient between two neurons.

        :param spike_buffer1: numpy.ndarray
            contains spike times for first neuron
        :param spike_buffer2: numpy.ndarray
            contains spike times for second neuron
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
        # create binned spike trains according to bin edges
        num_bins = ceil(int(self.buffer_size/self.bin_size))
        bin_edges = np.linspace(start=self.buffer_counter * self.buffer_size,
                                stop=self.buffer_counter * self.buffer_size +
                                     self.buffer_size, num=num_bins+1)
        binned_st1, _ = np.histogram(a=spike_buffer1, bins=bin_edges)
        binned_st2, _ = np.histogram(a=spike_buffer2, bins=bin_edges)
        # loop over bins and calculate update
        for b in range(num_bins):
            x_n_plus_1 = binned_st1[b]
            y_n_plus_1 = binned_st2[b]
            # n = number of bins in total considering all buffers
            n = self.buffer_counter * num_bins + b
            self.C_s += n / (n + 1) * (x_n_plus_1 - self.x_bar) * \
                            (y_n_plus_1 - self.y_bar)
            x_bar_new = self.x_bar + (x_n_plus_1 - self.x_bar) / (n + 1)
            y_bar_new = self.y_bar + (y_n_plus_1 - self.y_bar) / (n + 1)
            self.M_x += (x_n_plus_1 - self.x_bar) * (x_n_plus_1 - x_bar_new)
            self.M_y += (y_n_plus_1 - self.y_bar) * (y_n_plus_1 - y_bar_new)
            self.x_bar = x_bar_new
            self.y_bar = y_bar_new
        self.R_xy = self.C_s / (np.sqrt(self.M_x * self.M_y))
        self.buffer_counter += 1
        return self.R_xy
