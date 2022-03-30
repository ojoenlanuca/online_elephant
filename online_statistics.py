from math import ceil
import warnings
import neo
import numpy as np
import quantities as pq
import scipy.special as sc
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
        self.current_isi_histogram = np.zeros(shape=(len(self.bin_edges) - 1))

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


class OnlineUnitaryEventAnalysis:
    def __init__(self, bw_size=0.005 * pq.s, ew_pre_size=0.5 * pq.s, ew_post_size=0.5 * pq.s,
                 idw_size=1 * pq.s, saw_size=0.1 * pq.s,
                 saw_step=0.005*pq.s, mw_size=3,
                 significance_level_alpha=0.05, target_event=None):
        """
        Abbreviations:
        bw = bin window
        ew = event window
        tw = trial window
        saw = sliding analysis window
        idw = incoming data window
        mw = memory window
        """
        self.num_neurons = 2
        self.tw_size = ew_pre_size + ew_post_size
        self.tw = [[] for i in range(self.num_neurons)]  # pointer to slice of mw
        self.tw_counter = 0
        self.ew_pre_size = ew_pre_size
        self.ew_post_size = ew_post_size
        self.idw_size = idw_size  # TODO: unused now, but needed for further usage?
        self.saw_size = saw_size  # multiple of bw_size
        self.saw_step = saw_step  # multiple of bw_size
        self.saw_pos = None
        self.num_saw_pos = int(self.tw_size/self.saw_step + 1)
        self.saw_pos_counter = 0
        self.bw_size = bw_size
        self.num_bins = ceil(int((self.tw_size + self.saw_size) / self.bw_size))
        self.bw = np.zeros([self.num_neurons, self.num_bins])  # binned copy of tw
        self.bin_edges = None
        self.mw_size = mw_size
        self.mw = [[] for i in range(self.num_neurons)]  # array of all saved spiketimes
        self.memory_counter = 0  # TODO: unused now, but needed for further usage?
        self.significance_level_alpha = significance_level_alpha
        # for the moment it is assumed, that the target events are known
        # in advance of the simulation/experiment
        self.target_event = target_event  # list of target events
        self.waiting_for_new_target = True
        self.target_events_left_over = True
        self.result_dict = {
            "Js": np.zeros([self.num_saw_pos, 1]),
            "times": [[] for i in range(self.num_neurons)],
            "indices": {f"trial{i}": [] for i in range(len(self.target_event))},
            "n_emp": np.zeros([self.num_saw_pos, 1]),
            "n_exp": np.zeros([self.num_saw_pos, 1]),
            "rate_avg": np.zeros([self.num_saw_pos, 1, 2]) * pq.Hz,
            "input_parameters": {
                "pattern_hash": [3],  # defaults to 3 if only 2 neurons are used, i.e. (the first and second neuron) # TODO: calculate 'patter_hash' in  a later step
                "bin_size": self.bw_size.rescale('ms'),
                "win_size": self.saw_size.rescale('ms'),
                "win_step": self.saw_step.rescale('ms'),
                "method": None,  # None, because different methods (for calculating UEs) aren't available
                "t_start": 0 * pq.s,  # all trials are aligned to 0s
                "t_stop": self.tw_size,  # all trials are aligned to 0s
                "n_surrogates": None  # None, because different methods (for calculating UEs) aren't available, e.g. surrogate approach
            }
        }

    def get_results(self):
        self.result_dict["rate_avg"] /= self.tw_counter
        for i in range(len(self.target_event)):
            self.result_dict["indices"][f"trial{i}"] = (np.array(self.result_dict["indices"][f"trial{i}"]).flatten() - 10)   # -10 (num bins in -saw_size/2) needed to compensate different win_pos
        return self.result_dict

    def _save_idw_into_mw(self, idw):
        for i in range(self.num_neurons):
            self.mw[i] += idw[i].tolist()

    def _move_mw(self, overlap, t_stop):
        for i in range(self.num_neurons):
            idx = np.where((t_stop - overlap <= self.mw[i]) & (self.mw[i] <= t_stop))[0]
            if not len(idx) == 0:  # move mv
                self.mw[i] = self.mw[i][idx[0]:idx[-1]+1]
            else:  # keep mv
                pass
            # Debug Info: in step 33, idx is empty list => nothing to move => leads to index error

    def _define_tw(self, target_event):  # TODO: debug info -> tw was empty because target event wasn't updated / counter increased
        for i in range(self.num_neurons):
            self.tw[i] = [t for t in self.mw[i] if
                          (target_event - self.ew_pre_size - self.saw_size/2 <= t) &
                          (t <= target_event + self.ew_post_size + self.saw_size/2)]  # TODO: use a slicing view of mw instead of creating a new list

    def _check_tw_overlap(self, current_target_event, next_target_event):
        if current_target_event + self.ew_post_size + self.saw_size/2 > \
                next_target_event - self.ew_pre_size - self.saw_size/2:
            return True
        else:
            return False

    def _apply_bw_to_tw(self):
        c = self.tw_counter
        if c <= len(self.target_event)-1:
            trial_start = self.target_event[c] - self.ew_pre_size - self.saw_size/2
            trial_stop = self.target_event[c] + self.ew_post_size + self.saw_size/2
            self.bin_edges = np.linspace(start=trial_start, stop=trial_stop,
                                         num=self.num_bins + 1)
            for i in range(self.num_neurons):
                histo_i, _ = np.histogram(a=self.tw[i], bins=self.bin_edges)
                # clip histogramm
                histo_i = np.clip(histo_i, 0, 1)
                if histo_i.max() > 1:
                    print(f"histo_i.max{histo_i.max()}")
                self.bw[i] += histo_i
        else:
            # raise UserWarning("No further target events available!")
            self.target_events_left_over = False

    def _set_saw_positions(self):
        c = self.tw_counter
        if c <= len(self.target_event) - 1:
            t_start = self.target_event[c] - self.ew_pre_size
            t_stop = self.target_event[c] + self.ew_post_size
            self.saw_pos = np.linspace(start=t_start, stop=t_stop, num=self.num_saw_pos)
        else:
            # raise UserWarning("No further target events available!")
            self.target_events_left_over = False

    def _move_saw_over_tw(self, t_stop_idw):
        # define saw positions
        self._set_saw_positions()

        # iterate over saw positions
        for i in range(self.saw_pos_counter, len(self.saw_pos)):
            p = self.saw_pos[i]
            # check if saw filled with data? yes: -> a) & b);  no: -> pause
            if p + self.saw_size/2 <= t_stop_idw:  # saw is filled  # TODO: maybe check for lower boundery is also needed
                # at each saw position calculate rate_avg, n_exp & n_emp
                rate_avg = self._calculate_rate_avg(pos=p)
                n_emp_at_p, idx_coincidences = self._count_n_emp(pos=p)
                n_exp_at_p = self._calculate_n_exp(pos=p)
                # add rate_avg, n_emp & n_exp to result_dict (summed across trials)
                self.result_dict["rate_avg"][i] += rate_avg
                self.result_dict["n_emp"][i] += n_emp_at_p
                self.result_dict["n_exp"][i] += n_exp_at_p
                if len(idx_coincidences) > 0:
                    for idx in idx_coincidences:
                        self.result_dict["indices"][f"trial{self.tw_counter}"].append(idx)
                # evaluate significance of n_emp based on n_exp
                jp_value_at_p = self._evaluate_significance(self.result_dict["n_emp"][i]/self.tw_counter, self.result_dict["n_exp"][i]/self.tw_counter)
                # check if jp_values are significant
                if jp_value_at_p <= self.significance_level_alpha:
                    self._mark_unitary_events(p)
                    js = self._evaluate_surprise(jp_value_at_p)
                    self.result_dict["Js"][i] = js
                else:
                    pass

            else:  # saw is empty / half-filled -> pause iteration
                self.saw_pos_counter = i
                break
            if i == len(self.saw_pos)-1:  # last SAW position finished
                self.saw_pos_counter = 0
                if self.tw_counter < len(self.target_event)-1:
                    self.tw_counter += 1
                print(f"tw_counter = {self.tw_counter}")
                #  move MV after SAW is finished with analysis of one trial
                self._move_mw(overlap=self.ew_pre_size + self.saw_size/2, t_stop=t_stop_idw)
                # reset bw
                self.bw = np.zeros_like(self.bw)

    def _count_n_emp(self, pos):
        bin_idx_of_pos = np.where(np.histogram(pos, self.bin_edges)[0])[0][0]
        n_bins_in_saw_half = int(self.saw_size/(2*self.bw_size))
        pos_idx_minus_saw_half = bin_idx_of_pos - int(np.floor(n_bins_in_saw_half))
        pos_idx_plus_saw_half = bin_idx_of_pos + int(np.ceil(n_bins_in_saw_half))
        # TODO: expand it to more than 2 neurons
        idx_coincidences = np.where(
            np.sum(self.bw[:, pos_idx_minus_saw_half:pos_idx_plus_saw_half], axis=0)==2)[0]
        idx_coincidences = [j + pos_idx_minus_saw_half for j in idx_coincidences]
        n_emp = len(idx_coincidences)
        return n_emp, idx_coincidences

    def _calculate_n_exp(self, pos):
        bin_idx_of_pos = np.where(np.histogram(pos, self.bin_edges)[0])[0][0]
        n_bins_in_saw_half = int(self.saw_size / (2 * self.bw_size))
        pos_idx_minus_saw_half = bin_idx_of_pos - int(np.floor(n_bins_in_saw_half))
        pos_idx_plus_saw_half = bin_idx_of_pos + int(np.ceil(n_bins_in_saw_half))
        # TODO: expand it to more than 2 neurons
        c = np.sum(self.bw[:, pos_idx_minus_saw_half:pos_idx_plus_saw_half], axis=1)
        p = c / (2*n_bins_in_saw_half)
        n_exp = np.prod(p)*(2*n_bins_in_saw_half)
        return n_exp

    def _calculate_rate_avg(self, pos):
        bin_idx_of_pos = np.where(np.histogram(pos, self.bin_edges)[0])[0][0]
        n_bins_in_saw_half = int(self.saw_size / (2 * self.bw_size))
        pos_idx_minus_saw_half = bin_idx_of_pos - int(np.floor(n_bins_in_saw_half))
        pos_idx_plus_saw_half = bin_idx_of_pos + int(np.ceil(n_bins_in_saw_half))
        # TODO: expand it to more than 2 neurons
        c = np.sum(self.bw[:, pos_idx_minus_saw_half:pos_idx_plus_saw_half], axis=1)
        rate_avg = c / self.saw_size
        return rate_avg

    def _evaluate_significance(self, n_emp, n_exp):
        jp_value = 1.-sc.gammaincc(n_emp, n_exp)
        return jp_value

    def _evaluate_surprise(self, jp_value):
        with np.errstate(divide='ignore'):
            js = np.log((1-jp_value)/jp_value)
        return js

    def _mark_unitary_events(self, pos):
        saw_half = self.saw_size / 2
        for i in range(self.num_neurons):
            ue_times = [t for t in self.mw[i] if (t <= pos + saw_half) & (t >= pos - saw_half)]
            self.result_dict["times"][i] += ue_times
            # ToDo: add also "indices" to result_dict
            # histo_idx, _ = np.histogram(pos, self.bin_edges)
            # bin_idx = np.where(histo_idx)
            # self.result_dict["indices"][f"trial{self.tw_counter}"] += bin_idx

    def update_uea(self, spiketrains):
        # save incoming spikes (IDW) into memory (MW)
        self._save_idw_into_mw(spiketrains)
        # extract relevant time informations
        idw_t_start = spiketrains[0].t_start
        idw_t_stop = spiketrains[0].t_stop
        current_target_event = self.target_event[self.tw_counter]
        if self.tw_counter < len(self.target_event)-1:
            next_target_event = self.target_event[self.tw_counter+1]  # Todo: what happens for last target?
        else:
            next_target_event = np.inf * pq.s

        # # case 1: pre/post trial analysis, i.e. waiting for IDW  with new target event
        if self.waiting_for_new_target:
            # # subcase 1: IDW contains target event
            if (idw_t_start <= current_target_event) & (current_target_event <= idw_t_stop):
                self.waiting_for_new_target = False
                if self.target_events_left_over:
                    # define trial (TW) around target event, i.e. trial interval ranges from:
                    # [targetEvent - preEvent -SAW/w, targetEvent + postEvent + SAW/2]
                    # -> TW is pointer to a slice of MW
                    self._define_tw(target_event=current_target_event)
                    # apply BW to available data in TW; -> BW is a binned copy of TW
                    self._apply_bw_to_tw()
                    # move SAW over available data in TW
                    self._move_saw_over_tw(t_stop_idw=idw_t_stop)
                else:
                    pass
            # # subcase 2: IDW does not contain target event
            else:
                self._move_mw(overlap=self.ew_pre_size + self.saw_size / 2, t_stop=idw_t_stop)

        # # Case 2: within trial analysis, i.e. waiting for new IDW with spikes of current trial
        else:
            # # Subcase 3: IDW contains new target event
            if (idw_t_start <= next_target_event) & (next_target_event <= idw_t_stop):
                # check if an overlap between the current and the upcoming trial ranges exists
                if self._check_tw_overlap(current_target_event=current_target_event,
                                          next_target_event=next_target_event):
                    warnings.warn(f"Data in trial {self.tw_counter} will be analysed twice! "
                                      "Adjust the target events and/or the trial window size.", UserWarning)
                else:  # no overlap exists
                    pass
            # # Subcase 4: IDW does not contain target event, i.e. just new spikes of the current trial
            else:
                pass
            if self.target_events_left_over:
                # define trial (TW) around target event, i.e. trial interval ranges from:
                # [targetEvent - preEvent -SAW/w, targetEvent + postEvent + SAW/2]
                # -> TW is pointer to a slice of MW
                self._define_tw(target_event=current_target_event)
                # apply BW to available data in TW; -> BW is a binned copy of TW
                self._apply_bw_to_tw()
                # move SAW over available data in TW
                self._move_saw_over_tw(t_stop_idw=idw_t_stop)
            else:
                pass
