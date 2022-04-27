import warnings
from collections import defaultdict
from math import ceil

import elephant.conversion as conv
import neo
import numpy as np
import quantities as pq
import scipy.special as sc
from elephant.statistics import mean_firing_rate, isi
from elephant.unitary_event_analysis import *
from elephant.unitary_event_analysis import _winpos, _bintime, _UE

from utils import round_to_nearest_fraction_multiple


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
    def __init__(self, bw_size=0.005 * pq.s, trigger_pre_size=0.5 * pq.s,
                 trigger_post_size=0.5 * pq.s, idw_size=1 * pq.s,
                 saw_size=0.1 * pq.s, saw_step=0.005*pq.s, mw_size=3,
                 trigger_events=None, n_neurons=2, pattern_hash=None):
        """
        Abbreviations:
        bw = bin window
        ew = event window
        tw = trial window
        saw = sliding analysis window
        idw = incoming data window
        mw = memory window
        """
        self.data_available_in_mv = None
        self.time_unit = 1 * pq.s
        self.n_neurons = n_neurons
        self.tw_size = trigger_pre_size + trigger_post_size
        self.tw = [[] for _ in range(self.n_neurons)]  # pointer to slice of mw
        self.tw_counter = 0
        self.trigger_pre_size = trigger_pre_size
        self.trigger_post_size = trigger_post_size
        # self.idw_size = idw_size  # TODO: unused now, but needed in future?
        self.saw_size = saw_size  # multiple of bw_size
        self.saw_step = saw_step  # multiple of bw_size
        self.saw_pos_counter = 0
        self.bw_size = bw_size
        self.n_bins = None
        self.bw = None  # binned copy of tw
        # self.mw_size = mw_size  # TODO: unused now, but needed in future?
        self.mw = [[] for i in range(self.n_neurons)]  # array of all spiketimes
        # self.memory_counter = 0  # TODO: unused now, but needed in future?
        # for the moment it is assumed, that the trigger events are known
        # in advance of the simulation/experiment
        self.trigger_events = trigger_events.tolist()  # list of trigger events
        self.n_trials = len(trigger_events)
        self.waiting_for_new_trigger = True
        self.trigger_events_left_over = True
        if pattern_hash is None:
            pattern = [1] * n_neurons
            self.pattern_hash = hash_from_pattern(pattern)
        if np.issubdtype(type(self.pattern_hash), np.integer):
            self.pattern_hash = [int(self.pattern_hash)]
        self.n_hashes = len(self.pattern_hash)
        self.method = 'analytic_TrialByTrial'
        self.n_surrogates = 100
        self.input_parameters = dict(pattern_hash=self.pattern_hash,
                                     bin_size=self.bw_size.rescale(pq.ms),
                                     win_size=self.saw_size.rescale(pq.ms),
                                     win_step=self.saw_step.rescale(pq.ms),
                                     method=self.method,
                                     t_start=0*pq.s,
                                     t_stop=self.tw_size,
                                     n_surrogates=self.n_surrogates)
        self.n_windows = int(np.round(
            (self.tw_size-self.saw_size+self.saw_step) / self.saw_step))
        self.Js_win, self.n_exp_win, self.n_emp_win = np.zeros(
            (3, self.n_windows, self.n_hashes), dtype=np.float64)
        self.rate_avg = np.zeros(
            (self.n_windows, self.n_hashes, self.n_neurons), dtype=np.float64)
        self.indices_win = defaultdict(list)

    def get_results(self):
        """Return result dictionary."""
        for key in self.indices_win.keys():
            self.indices_win[key] = np.hstack(self.indices_win[key]).flatten()
        self.n_exp_win /= (self.saw_size / self.bw_size)
        p = self._pval(self.n_emp_win.astype(np.float64),
                       self.n_exp_win.astype(np.float64)).flatten()
        self.Js_win = jointJ(p)
        self.rate_avg = (self.rate_avg * (self.saw_size / self.bw_size)) / \
                        (self.saw_size.rescale(pq.ms) * self.n_trials)
        return {
            'Js': self.Js_win.reshape((len(self.Js_win), 1)).astype(np.float32),
            'indices': self.indices_win,
            'n_emp': self.n_emp_win.reshape((len(self.n_emp_win), 1)).astype(np.float32),
            'n_exp': self.n_exp_win.reshape((len(self.n_exp_win), 1)).astype(np.float32),
            'rate_avg': self.rate_avg.reshape((len(self.rate_avg), 1, 2)).astype(np.float32),
            'input_parameters': self.input_parameters}

    def _pval(self, n_emp, n_exp):
        """Calculate p-value of detecting 'n_emp' or more coincidences based
        on a distribution with sole parameter 'n_exp'."""
        p = 1. - sc.gammaincc(n_emp, n_exp)
        return p

    def _save_idw_into_mw(self, idw):
        """Save in-incoming data window (IDW) into memory window (MW)."""
        for i in range(self.n_neurons):
            self.mw[i] += idw[i].tolist()

    def _move_mw(self, new_t_start):
        """Move memory window."""
        # TODO: too small overlap leads to too fast moving of mv,
        #  i.e. spikes of current trial are discarded
        for i in range(self.n_neurons):
            idx = np.where(new_t_start <= self.mw[i])[0]
            if not len(idx) == 0:  # move mv
                self.mw[i] = self.mw[i][idx[0]:idx[-1]+1]
            else:  # keep mv
                self.data_available_in_mv = False
                pass

    def _define_tw(self, trigger_event):
        """Define trial window (TW) based on a trigger event."""
        self.trial_start = trigger_event - self.trigger_pre_size
        self.trial_stop = trigger_event + self.trigger_post_size
        for i in range(self.n_neurons):
            # TODO: use a slicing view of mw instead of creating a new list
            self.tw[i] = [t for t in self.mw[i]
                          if (self.trial_start <= t) & (t <= self.trial_stop)]

    def _check_tw_overlap(self, current_trigger_event, next_trigger_event):
        """Check if successive trials do overlap each other."""
        if current_trigger_event + self.trigger_post_size > \
                next_trigger_event - self.trigger_pre_size:
            return True
        else:
            return False

    def _apply_bw_to_tw(self, spiketrains, bin_size, t_start, t_stop,
                        n_neurons):
        """Apply bin window (BW) to trial window (TW)."""
        self.n_bins = int(((t_stop - t_start) / bin_size).simplified.item())
        self.bw = np.zeros((1, n_neurons, self.n_bins), dtype=np.int32)
        spiketrains = [neo.SpikeTrain(np.array(st)*self.time_unit,
                                      t_start=t_start, t_stop=t_stop)
                       for st in spiketrains]
        bs = conv.BinnedSpikeTrain(spiketrains, t_start=t_start,
                                   t_stop=t_stop, bin_size=bin_size)
        self.bw = bs.to_bool_array()

    def _set_saw_positions(self, t_start, t_stop, win_size, win_step, bin_size):
        """Set positions of the sliding analysis window (SAW)."""
        self.t_winpos = _winpos(t_start, t_stop, win_size, win_step,
                                position='left-edge')
        while len(self.t_winpos) != self.n_windows:
            # print(f"n_winpos = {self.n_windows} | "             # DEBUG-aid
            #       f"len(t_winpos) = {len(self.t_winpos)}")      # DEBUG-aid
            if len(self.t_winpos) > self.n_windows:
                self.t_winpos = _winpos(t_start, t_stop - win_step/2, win_size,
                                        win_step, position='left-edge')
            else:
                self.t_winpos = _winpos(t_start, t_stop + win_step/2, win_size,
                                        win_step, position='left-edge')
        self.t_winpos_bintime = _bintime(self.t_winpos, bin_size)
        self.winsize_bintime = _bintime(win_size, bin_size)
        self.winstep_bintime = _bintime(win_step, bin_size)
        if self.winsize_bintime * bin_size != win_size:
            warnings.warn(f"The ratio between the win_size ({win_size}) and the"
                          f" bin_size ({bin_size}) is not an integer")
        if self.winstep_bintime * bin_size != win_step:
            warnings.warn(f"The ratio between the win_step ({win_step}) and the"
                          f" bin_size ({bin_size}) is not an integer")

    def _move_saw_over_tw(self, t_stop_idw):
        """Move sliding analysis window (SAW) over trial window (TW)."""
        # define saw positions
        self._set_saw_positions(
            t_start=self.trial_start, t_stop=self.trial_stop,
            win_size=self.saw_size, win_step=self.saw_step,
            bin_size=self.bw_size)

        # iterate over saw positions
        for i in range(self.saw_pos_counter, self.n_windows):
            p_realtime = self.t_winpos[i]
            p_bintime = self.t_winpos_bintime[i] - self.t_winpos_bintime[0]
            # check if saw filled with data? yes: -> a) & b);  no: -> pause
            # TODO: maybe check for lower boundery is also needed
            if p_realtime + self.saw_size <= t_stop_idw:  # saw is filled
                mat_win = np.zeros((1, self.n_neurons, self.winsize_bintime))
                # if i == 0:                                # DEBUG-aid
                #     print(f"debug_entry = {i}")           # DEBUG-aid
                n_bins_in_current_saw = self.bw[
                    :, p_bintime:p_bintime + self.winsize_bintime].shape[1]
                if n_bins_in_current_saw < self.winsize_bintime:
                    mat_win[0] += np.pad(
                        self.bw[:, p_bintime:p_bintime+self.winsize_bintime],
                        (0, self.winsize_bintime-n_bins_in_current_saw),
                        "minimum")[0:2]
                else:
                    mat_win[0] += \
                        self.bw[:, p_bintime:p_bintime+self.winsize_bintime]
                Js_win, rate_avg, n_exp_win, n_emp_win, indices_lst = _UE(
                    mat_win, pattern_hash=self.pattern_hash,
                    method=self.method, n_surrogates=self.n_surrogates)
                # if i == 0:                                # DEBUG-aid
                #     print(f"trial = {self.tw_counter}     # DEBUG-aid
                #     sum = {rate_avg * (self.saw_size/     # DEBUG-aid
                #     self.bw_size )}")                     # DEBUG-aid
                self.rate_avg[i] += rate_avg
                self.n_exp_win[i] += (np.round(n_exp_win * (self.saw_size / self.bw_size))).astype(int)
                self.n_emp_win[i] += n_emp_win
                self.indices_lst = indices_lst
                if len(self.indices_lst[0]) > 0:
                    self.indices_win[f"trial{self.tw_counter}"].append(
                        self.indices_lst[0] + p_bintime)
            else:  # saw is empty / half-filled -> pause iteration
                self.saw_pos_counter = i
                self.data_available_in_mv = False
                break
            if i == self.n_windows-1:  # last SAW position finished
                self.saw_pos_counter = 0
                #  move MV after SAW is finished with analysis of one trial
                self._move_mw(new_t_start=self.trigger_events[self.tw_counter] +
                                          self.tw_size)
                # reset bw
                self.bw = np.zeros_like(self.bw)
                if self.tw_counter <= self.n_trials - 1:
                    self.tw_counter += 1
                else:
                    self.waiting_for_new_trigger = True
                    self.trigger_events_left_over = False
                    self.data_available_in_mv = False
                print(f"tw_counter = {self.tw_counter}")        # DEBUG-aid

    def update_uea(self, spiketrains, events=None):
        """Update unitary events analysis UEA with new arriving spike data."""
        if events is None:
            events = np.array([])
        if len(events) > 0:
            for event in events:
                if event not in self.trigger_events:
                    self.trigger_events.append(event)
            self.trigger_events.sort()
            self.n_trials = len(self.trigger_events)
        # save incoming spikes (IDW) into memory (MW)
        self._save_idw_into_mw(spiketrains)
        # extract relevant time informations
        idw_t_start = spiketrains[0].t_start
        idw_t_stop = spiketrains[0].t_stop
        
        # analyse all trials which are available in the memory
        self.data_available_in_mv = True
        while self.data_available_in_mv:
            if self.tw_counter == self.n_trials:
                break
            if self.n_trials == 0:
                current_trigger_event = np.inf * pq.s
                next_trigger_event = np.inf * pq.s
            else:
                current_trigger_event = self.trigger_events[self.tw_counter]
                if self.tw_counter <= self.n_trials - 2:
                    next_trigger_event = self.trigger_events[self.tw_counter + 1]
                else:
                    next_trigger_event = np.inf * pq.s
    
            # # case 1: pre/post trial analysis,
            # i.e. waiting for IDW  with new trigger event
            if self.waiting_for_new_trigger:
                # # subcase 1: IDW contains trigger event
                if (idw_t_start <= current_trigger_event) & \
                        (current_trigger_event <= idw_t_stop):
                    self.waiting_for_new_trigger = False
                    if self.trigger_events_left_over:
                        # define trial (TW) around trigger event,
                        # i.e. trial interval ranges from:
                        # [trigger - preEvent -SAW/2, trigger + postEvent + SAW/2]
                        # -> TW is pointer to a slice of MW
                        self._define_tw(trigger_event=current_trigger_event)
                        # apply BW to available data in TW
                        # -> BW is a binned copy of TW
                        self._apply_bw_to_tw(
                            spiketrains=self.tw, bin_size=self.bw_size,
                            t_start=self.trial_start, t_stop=self.trial_stop,
                            n_neurons=self.n_neurons)
                        # move SAW over available data in TW
                        self._move_saw_over_tw(t_stop_idw=idw_t_stop)
                    else:
                        pass
                # # subcase 2: IDW does not contain trigger event
                else:
                    self._move_mw(new_t_start=idw_t_stop-self.trigger_pre_size)
    
            # # Case 2: within trial analysis,
            # i.e. waiting for new IDW with spikes of current trial
            else:
                # # Subcase 3: IDW contains new trigger event
                if (idw_t_start <= next_trigger_event) & \
                        (next_trigger_event <= idw_t_stop):
                    # check if an overlap between current / next trial range exists
                    if self._check_tw_overlap(
                            current_trigger_event=current_trigger_event,
                            next_trigger_event=next_trigger_event):
                        warnings.warn(
                            f"Data in trial {self.tw_counter} will be analysed "
                            f"twice! Adjust the trigger events and/or "
                            f"the trial window size.", UserWarning)
                    else:  # no overlap exists
                        pass
                # # Subcase 4: IDW does not contain trigger event,
                # i.e. just new spikes of the current trial
                else:
                    pass
                if self.trigger_events_left_over:
                    # define trial (TW) around trigger event,
                    # i.e. trial interval ranges from:
                    # [trigger - preEvent -SAW/w, trigger + postEvent + SAW/2]
                    # -> TW is pointer to a slice of MW
                    self._define_tw(trigger_event=current_trigger_event)
                    # apply BW to available data in TW; -> BW is a binned copy of TW
                    self._apply_bw_to_tw(
                        spiketrains=self.tw, bin_size=self.bw_size,
                        t_start=self.trial_start, t_stop=self.trial_stop,
                        n_neurons=self.n_neurons)
                    # move SAW over available data in TW
                    self._move_saw_over_tw(t_stop_idw=idw_t_stop)
                else:
                    pass
