import random
import unittest
from collections import defaultdict
from time import perf_counter_ns

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
import viziphant
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.statistics import mean_firing_rate, isi
from elephant.unitary_event_analysis import jointJ_window_analysis

from online_statistics import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient, OnlineUnitaryEventAnalysis


class TestOnlineMeanFiringRate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rtol = 1e-15
        cls.atol = 1e-15
        cls.num_neurons = 10
        cls.num_buffers = 100
        cls.buff_size = 1  # in sec

    def test_correctness_multiple_neurons(self):
        """Test, if results of online and normal mean firing rate function
        are identical for multiple neurons."""
        # create list of spiketrain lists
        list_of_st_lists = [[homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)] for _ in range(self.num_neurons)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call calculate_mfr()
        # for neo.Spiketrain input
        omfr_all_neurons_neo = [OnlineMeanFiringRate()
                                for _ in range(self.num_neurons)]
        tic1_neo = perf_counter_ns()
        for j, st_list in enumerate(list_of_st_lists):
            for st in st_list:
                omfr_all_neurons_neo[j].update_mfr(spike_buffer=st)
        final_online_mfr_neo = [omfr_all_neurons_neo[j].current_mfr
                                for j in range(self.num_neurons)]
        toc1_neo = perf_counter_ns()
        # for numpy.ndarray input
        omfr_all_neurons_np = [OnlineMeanFiringRate()
                               for _ in range(self.num_neurons)]
        tic1_np = perf_counter_ns()
        for j, st_list in enumerate(list_of_st_lists):
            for st in st_list:
                omfr_all_neurons_np[j].update_mfr(spike_buffer=st.magnitude)
        final_online_mfr_np = [omfr_all_neurons_np[j].current_mfr
                               for j in range(self.num_neurons)]
        toc1_np = perf_counter_ns()

        # concatenate each list of spiketrains to one spiketrain per neuron
        # and call 'offline' mean_firing_rate()
        t_start_first_st = list_of_st_lists[0][0].t_start
        t_stop_last_st = list_of_st_lists[0][-1].t_stop
        concatenated_st = [neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st_list])*pq.s,
            t_start=t_start_first_st, t_stop=t_stop_last_st)
            for st_list in list_of_st_lists]
        tic2 = perf_counter_ns()
        normal_mfr = [mean_firing_rate(concatenated_st[j]).rescale(pq.Hz)
                      for j in range(self.num_neurons)]
        toc2 = perf_counter_ns()

        # compare results of normal mfr and online mfr
        with self.subTest(msg="neo.Spiketrain input"):
            print(f"neo.Spiketrain input\n"
                  f"online_mfr:  | run-time: {(toc1_neo-tic1_neo)*1e-9}sec\n"
                  f"normal_mfr:  | run-time: {(toc2-tic2)*1e-9}sec\n"
                  f"(t_online_neo/t_normal)={(toc1_neo-tic1_neo)/(toc2-tic2)}")
            for j in range(self.num_neurons):
                np.testing.assert_allclose(
                    final_online_mfr_neo[j].magnitude, normal_mfr[j].magnitude,
                    rtol=self.rtol, atol=self.atol)
        with self.subTest(msg="numpy.ndarray input"):
            print(f"numpy.ndarray input\n"
                  f"online_mfr:  | run-time: {(toc1_np-tic1_np)*1e-9}sec\n"
                  f"normal_mfr:  | run-time: {(toc2-tic2)*1e-9}sec\n"
                  f"(t_online_np/t_normal)={(toc1_np-tic1_np)/(toc2-tic2)}")
            for j in range(self.num_neurons):
                np.testing.assert_allclose(
                    final_online_mfr_np[j].magnitude, normal_mfr[j].magnitude,
                    rtol=self.rtol, atol=self.atol)


class TestOnlineInterSpikeInterval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rtol = 1e-15
        cls.atol = 1e-15
        cls.num_neurons = 10
        cls.num_buffers = 100
        cls.buff_size = 1  # in sec

    def test_correctness_multiple_neurons(self):
        """Test, if results of online and normal inter-spike interval
        function are identical for multiple neurons."""
        # create list of spiketrain lists
        list_of_st_lists = [[homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)] for _ in range(self.num_neurons)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call calculate_isi()
        # for neo.Spiketrain input
        oisi_all_neurons_neo = [OnlineInterSpikeInterval()
                                for _ in range(self.num_neurons)]
        tic1_neo = perf_counter_ns()
        for j, st_list in enumerate(list_of_st_lists):
            for st in st_list:
                oisi_all_neurons_neo[j].update_isi(spike_buffer=st)
        final_online_isi_histogram_neo = [oisi_all_neurons_neo[j].current_isi_histogram
                                for j in range(self.num_neurons)]
        toc1_neo = perf_counter_ns()
        # for np.ndarray input
        oisi_all_neurons_np = [OnlineInterSpikeInterval()
                               for _ in range(self.num_neurons)]
        tic1_np = perf_counter_ns()
        for j, st_list in enumerate(list_of_st_lists):
            for st in st_list:
                oisi_all_neurons_np[j].update_isi(spike_buffer=st.magnitude)
        final_online_isi_histogram_np = [oisi_all_neurons_np[j].current_isi_histogram
                               for j in range(self.num_neurons)]
        toc1_np = perf_counter_ns()

        # concatenate list of spiketrains to one single spiketrain
        # and call 'offline' isi()
        t_start_first_st = list_of_st_lists[0][0].t_start
        t_stop_last_st = list_of_st_lists[0][-1].t_stop
        concatenated_st = [neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st_list]) * pq.s,
            t_start=t_start_first_st, t_stop=t_stop_last_st)
            for st_list in list_of_st_lists]
        tic2 = perf_counter_ns()
        normal_isi = [isi(concatenated_st[j]).magnitude.tolist()
                      for j in range(self.num_neurons)]
        # create histogram of ISI values
        bin_size = 0.0005  # in sec
        max_isi_value = 1  # in sec
        num_bins = int(max_isi_value / bin_size)
        bin_edges = np.linspace(start=0, stop=max_isi_value, num=num_bins+1)
        normal_isi_histogram = []
        for j in range(self.num_neurons):
            histogram, _ = np.histogram(normal_isi[j], bins=bin_edges)
            normal_isi_histogram.append(histogram)
        toc2 = perf_counter_ns()

        # compare results of normal isi and online isi
        with self.subTest(msg="neo.Spiketrain input"):
            print(f"online_isi:  | run-time: {(toc1_neo-tic1_neo)*1e-9}sec\n"
                  f"normal_isi:  | run-time: {(toc2-tic2)*1e-9}sec\n"
                  f"(t_online_neo/t_normal)={(toc1_neo-tic1_neo)/(toc2-tic2)}")
            for j in range(self.num_neurons):
                np.testing.assert_allclose(
                    final_online_isi_histogram_neo[j], normal_isi_histogram[j],
                    rtol=self.rtol, atol=self.atol)
        with self.subTest(msg="np.ndarray input"):
            print(f"online_isi:  | run-time: {(toc1_np-tic1_np)*1e-9}sec\n"
                  f"normal_isi:  | run-time: {(toc2-tic2)*1e-9}sec\n"
                  f"(t_online_np/t_normal)={(toc1_np-tic1_np)/(toc2-tic2)}")
            for j in range(self.num_neurons):
                np.testing.assert_allclose(
                    final_online_isi_histogram_np[j], normal_isi_histogram[j],
                    rtol=self.rtol, atol=self.atol)


class TestOnlinePearsonCorrelationCoefficient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rtol = 1e-15
        cls.atol = 1e-15
        cls.num_neurons = 10
        cls.num_buffers = 100
        cls.buff_size = 1  # in sec

    def test_correctness_with_two_neurons_and_single_loop(self):
        """Test, if results of online and normal function of
        Pearson's correlation coefficient are identical for two neurons and
        a single loop / buffer call."""
        # create spiketrains
        t_length = 1*pq.s
        st1 = homogeneous_poisson_process(rate=50*pq.Hz, t_stop=t_length)
        st2 = homogeneous_poisson_process(rate=50*pq.Hz, t_stop=t_length)
        binned_st = BinnedSpikeTrain([st1, st2], bin_size=5*pq.ms)

        # call calculate_pcc()
        opcc_neuron_pair1 = OnlinePearsonCorrelationCoefficient()
        tic1 = perf_counter_ns()
        opcc_neuron_pair1.update_pcc(spike_buffer1=st1, spike_buffer2=st2)
        toc1 = perf_counter_ns()
        final_online_pcc = opcc_neuron_pair1.R_xy

        # call normal correlation_coefficient()
        tic2 = perf_counter_ns()
        normal_pcc = correlation_coefficient(binned_spiketrain=binned_st)
        toc2 = perf_counter_ns()

        # compare online and normal pcc results
        print(f"online pcc:{final_online_pcc}|run-time:{(toc1-tic1)*1e-9}sec\n"
              f"normal pcc:{normal_pcc[0][1]}|run-time:{(toc2-tic2)*1e-9}sec\n"
              f"PCC1: (t_online / t_normal)={(toc1-tic1)/(toc2-tic2)}")
        np.testing.assert_allclose(final_online_pcc, normal_pcc[0][1],
                                   rtol=self.rtol, atol=self.atol)

    def test_correctness_with_two_neurons_and_multiple_loops(self):
        """Test, if results of online and normal function of
        Pearson's correlation coefficient are identical for two neurons and
        multiple loops / buffer calls."""
        # create list of spiketrains
        st1_list = [homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)]
        st2_list = [homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over binned spiketrain list and call calculate_pcc()
        opcc_neuron_pair1 = OnlinePearsonCorrelationCoefficient()
        tic1 = perf_counter_ns()
        for i in range(self.num_buffers):
            opcc_neuron_pair1.update_pcc(spike_buffer1=st1_list[i],
                                         spike_buffer2=st2_list[i])
        final_online_pcc = opcc_neuron_pair1.R_xy
        toc1 = perf_counter_ns()

        # concatenate each list of spiketrains to one spiketrain
        t_start_first_st1 = st1_list[0].t_start
        t_stop_last_st1 = st1_list[-1].t_stop
        concatenated_st1 = neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st1_list])*pq.s,
            t_start=t_start_first_st1, t_stop=t_stop_last_st1)
        t_start_first_st2 = st2_list[0].t_start
        t_stop_last_st2 = st2_list[-1].t_stop
        concatenated_st2 = neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st2_list]) * pq.s,
            t_start=t_start_first_st2, t_stop=t_stop_last_st2)
        # then create one BinnedSpikeTrain of the concatenated spiketrains
        concatenated_binned_st = BinnedSpikeTrain(
            [concatenated_st1, concatenated_st2], bin_size=5 * pq.ms)
        # call normal correlation_coefficient()
        tic2 = perf_counter_ns()
        normal_pcc = correlation_coefficient(
            binned_spiketrain=concatenated_binned_st)
        toc2 = perf_counter_ns()

        # compare online and normal pcc results
        print(f"online pcc:{final_online_pcc}|run-time:{(toc1-tic1)*1e-9}sec\n"
              f"normal pcc:{normal_pcc[0][1]}|run-time:{(toc2-tic2)*1e-9}sec\n"
              f"PCC2: (t_online / t_normal)={(toc1-tic1)/(toc2-tic2)}\n")
        np.testing.assert_allclose(final_online_pcc, normal_pcc[0][1],
                                   rtol=self.rtol, atol=self.atol)


def _generate_spiketrains(freq, length, trigger_events, injection_pos,
                          trigger_pre_size, trigger_post_size,
                          time_unit=1*pq.s):
    """
    Generate two spiketrains from an homogeneous Poisson process with
    injected coincideces.
    """
    st1 = homogeneous_poisson_process(rate=freq,
                                      t_start=(0*pq.s).rescale(time_unit),
                                      t_stop=length.rescale(time_unit))
    st2 = homogeneous_poisson_process(rate=freq,
                                      t_start=(0*pq.s.rescale(time_unit)),
                                      t_stop=length.rescale(time_unit))
    # inject 10 coincidences within a 0.1s interval for each trial
    injection = (np.linspace(0, 0.1, 10)*pq.s).rescale(time_unit)
    all_injections = np.array([])
    for i in trigger_events:
        all_injections = np.concatenate(
            (all_injections, (i+injection_pos)+injection), axis=0) * time_unit
    st1 = st1.duplicate_with_new_data(
        np.sort(np.concatenate((st1.times, all_injections)))*time_unit)
    st2 = st2.duplicate_with_new_data(
        np.sort(np.concatenate((st2.times, all_injections)))*time_unit)

    # stack spiketrains by trial
    st1_stacked = [st1.time_slice(
        t_start=i - trigger_pre_size,
        t_stop=i + trigger_post_size).time_shift(-i + trigger_pre_size)
                   for i in trigger_events]
    st2_stacked = [st2.time_slice(
        t_start=i - trigger_pre_size,
        t_stop=i + trigger_post_size).time_shift(-i + trigger_pre_size)
                   for i in trigger_events]
    spiketrains = np.stack((st1_stacked, st2_stacked), axis=1)

    return spiketrains, st1, st2


def _visualize_results_of_offline_and_online_uea(
        spiketrains, ue_dict_offline, ue_dict_online, alpha):
    viziphant.unitary_event_analysis.plot_ue(
        spiketrains, Js_dict=ue_dict_offline, significance_level=alpha,
        unit_real_ids=['1', '2'])
    plt.show()
    viziphant.unitary_event_analysis.plot_ue(
        spiketrains, Js_dict=ue_dict_online, significance_level=alpha,
        unit_real_ids=['1', '2'])
    plt.show()


def _simulate_buffered_reading(n_buffers, ouea, st1, st2, IDW_length,
                               length_remainder, events=None):
    if events is None:
        events = np.array([])
    for i in range(n_buffers):
        buff_t_start = i * IDW_length

        if length_remainder > 1e-7 and i == n_buffers - 1:
            buff_t_stop = i * IDW_length + length_remainder
        else:
            buff_t_stop = i * IDW_length + IDW_length

        events_in_buffer = np.array([])
        if len(events) > 0:
            idx_events_in_buffer = (events >= buff_t_start) & \
                                   (events <= buff_t_stop)
            events_in_buffer = events[idx_events_in_buffer].tolist()
            events = events[np.logical_not(idx_events_in_buffer)]

        ouea.update_uea(
            spiketrains=[
                st1.time_slice(t_start=buff_t_start, t_stop=buff_t_stop),
                st2.time_slice(t_start=buff_t_start, t_stop=buff_t_stop)],
            events=events_in_buffer)
        print(f"#buffer = {i}")  # DEBUG-aid
        # # aid to create timelapses
        # result_dict = ouea.get_results()
        # viziphant.unitary_event_analysis.plot_ue(
        #     spiketrains[:i+1], Js_dict=result_dict, significance_level=0.05,
        #     unit_real_ids=['1', '2'])
        # plt.savefig(f"plots/timelapse_UE/ue_real_data_buff_{i}.pdf")


def _load_real_data(n_trials, trial_length, time_unit):
    # load data and extract spiketrains
    io = neo.io.NixIO("data/dataset-1.nix", 'ro')
    block = io.read_block()
    spiketrains = []
    # each segment contains a single trial
    for ind in range(len(block.segments)):
        spiketrains.append(block.segments[ind].spiketrains)

    # for each neuron: concatenate all trials to one long neo.Spiketrain
    st1_long = [spiketrains[i].multiplexed[1][
                    np.where(spiketrains[i].multiplexed[0] == 0)]
                + i * (trial_length)
                for i in range(len(spiketrains))]
    st2_long = [spiketrains[i].multiplexed[1][
                    np.where(spiketrains[i].multiplexed[0] == 1)]
                + i * (trial_length)
                for i in range(len(spiketrains))]
    st1_concat = st1_long[0]
    st2_concat = st2_long[0]
    for i in range(1, len(st1_long)):
        st1_concat = np.concatenate((st1_concat, st1_long[i]))
        st2_concat = np.concatenate((st2_concat, st2_long[i]))
    neo_st1 = neo.SpikeTrain((st1_concat / 1000) * pq.s, t_start=0 * pq.s,
                             t_stop=n_trials * trial_length).rescale(time_unit)
    neo_st2 = neo.SpikeTrain((st2_concat / 1000) * pq.s, t_start=0 * pq.s,
                             t_stop=n_trials * trial_length).rescale(time_unit)
    return spiketrains, neo_st1, neo_st2


def _calculate_n_buffers(n_trials, tw_length, noise_length, idw_length):
    _n_buffers_float = n_trials * (tw_length + noise_length) / idw_length
    _n_buffers_int = int(_n_buffers_float)
    _n_buffers_fraction = _n_buffers_float - _n_buffers_int
    n_buffers = _n_buffers_int + 1 if _n_buffers_fraction > 1e-7 else \
        _n_buffers_int
    length_remainder = idw_length * _n_buffers_fraction
    return n_buffers, length_remainder


class TestOnlineUnitaryEventAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(73)
        cls.time_unit = 1 * pq.s

    def setUp(self):
        pass

    def _assert_equality_of_result_dicts(self, ue_dict_offline, ue_dict_online,
                                         tol_dict_user):
        eps_float64 = np.finfo(np.float64).eps
        eps_float32 = np.finfo(np.float32).eps
        tol_dict = {"atol_Js": eps_float64, "rtol_Js": eps_float64,
                    "atol_indices": eps_float64, "rtol_indices": eps_float64,
                    "atol_n_emp": eps_float64, "rtol_n_emp": eps_float64,
                    "atol_n_exp": eps_float64, "rtol_n_exp": eps_float64,
                    "atol_rate_avg": eps_float32, "rtol_rate_avg": eps_float32}
        tol_dict.update(tol_dict_user)

        with self.subTest("test 'Js' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["Js"], desired=ue_dict_offline["Js"],
                atol=tol_dict["atol_Js"],
                rtol=tol_dict["rtol_Js"])
        with self.subTest("test 'indices' equality"):
            for key in ue_dict_offline["indices"].keys():
                np.testing.assert_allclose(
                    actual=ue_dict_online["indices"][key],
                    desired=ue_dict_offline["indices"][key],
                    atol=tol_dict["atol_indices"],
                    rtol=tol_dict["rtol_indices"])
        with self.subTest("test 'n_emp' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["n_emp"],
                desired=ue_dict_offline["n_emp"],
                atol=tol_dict["atol_n_emp"], rtol=tol_dict["rtol_n_emp"])
        with self.subTest("test 'n_exp' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["n_exp"],
                desired=ue_dict_offline["n_exp"],
                atol=tol_dict["atol_n_exp"],
                rtol=tol_dict["rtol_n_exp"])
        with self.subTest("test 'rate_avg' equality"):
            np.testing.assert_allclose(
                actual=ue_dict_online["rate_avg"].magnitude,
                desired=ue_dict_offline["rate_avg"].magnitude,
                atol=tol_dict["atol_rate_avg"], rtol=tol_dict["rtol_rate_avg"])
        with self.subTest("test 'input_parameters' equality"):
            for key in ue_dict_offline["input_parameters"].keys():
                np.testing.assert_equal(
                    actual=ue_dict_online["input_parameters"][key],
                    desired=ue_dict_offline["input_parameters"][key])

    def _test_unitary_events_analysis_with_real_data(
            self, idw_length, method="pass_events_at_initialization",
            time_unit=1 * pq.s):
        # Fix random seed to guarantee fixed output
        random.seed(1224)

        # set relevant variables of this TestCase
        n_trials = 36  # determined by real data
        TW_length = (2.1 * pq.s).rescale(time_unit)  # determined by real data
        IDW_length = idw_length.rescale(time_unit)
        noise_length = (0. * pq.s).rescale(time_unit)
        trigger_events = (np.arange(0., n_trials * 2.1, 2.1) * pq.s).rescale(
            time_unit)
        n_buffers, length_remainder = _calculate_n_buffers(
            n_trials=n_trials, tw_length=TW_length,
            noise_length=noise_length, idw_length=IDW_length)

        # load data and extract spiketrains
        # 36 trials with 2.1s length and 0s background noise in between trials
        spiketrains, neo_st1, neo_st2 = _load_real_data(n_trials=n_trials,
                                                        trial_length=TW_length,
                                                        time_unit=time_unit)

        # perform standard unitary events analysis
        ue_dict = jointJ_window_analysis(
            spiketrains, bin_size=(0.005 * pq.s).rescale(time_unit),
            winsize=(0.1 * pq.s).rescale(time_unit),
            winstep=(0.005 * pq.s).rescale(time_unit), pattern_hash=[3])

        if method == "pass_events_at_initialization":
            init_events = trigger_events
            reading_events = np.array([]) * time_unit
        elif method == "pass_events_while_buffered_reading":
            init_events = np.array([]) * time_unit
            reading_events = trigger_events
        else:
            raise ValueError("Illeagal method to pass events!")

        # create instance of OnlineUnitaryEventAnalysis
        # TODO: use one pyhsical unit as standard and rescale others accordingly
        ouea = OnlineUnitaryEventAnalysis(
            bw_size=(0.005 * pq.s).rescale(time_unit),
            trigger_pre_size=(0. * pq.s).rescale(time_unit),
            trigger_post_size=(2.1 * pq.s).rescale(time_unit),
            saw_size=(0.1 * pq.s).rescale(time_unit),
            saw_step=(0.005 * pq.s).rescale(time_unit),
            trigger_events=init_events,
            time_unit=time_unit)
        # perform online unitary events analysis
        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call update_ue()
        _simulate_buffered_reading(n_buffers=n_buffers, ouea=ouea, st1=neo_st1,
                                   st2=neo_st2, IDW_length=IDW_length,
                                   length_remainder=length_remainder,
                                   events=reading_events)
        ue_dict_online = ouea.get_results()

        # assert equality between result dicts of standard and online ue version
        self._assert_equality_of_result_dicts(
            ue_dict_offline=ue_dict, ue_dict_online=ue_dict_online,
            tol_dict_user={})
        # fixme: larger atol & rtol ok?

        # visualize results of online and standard UEA for real data
        # _visualize_results_of_offline_and_online_uea(
        #     spiketrains=spiketrains, ue_dict_offline=ue_dict,
        #     ue_dict_online=ue_dict_online, alpha=0.05)

        return ouea

    def _test_unitary_events_analysis_with_artificial_data(
            self, idw_length, method="pass_events_at_initialization",
            time_unit=1 * pq.s):
        # Fix random seed to guarantee fixed output
        random.seed(1224)

        # set relevant variables of this TestCase
        n_trials = 40
        TW_length = (1 * pq.s).rescale(time_unit)
        noise_length = (1.5 * pq.s).rescale(time_unit)
        IDW_length = idw_length.rescale(time_unit)
        trigger_events = (np.arange(0., n_trials*2.5, 2.5) * pq.s).rescale(
            time_unit)
        trigger_pre_size = (0. * pq.s).rescale(time_unit)
        trigger_post_size = (1. * pq.s).rescale(time_unit)
        n_buffers, length_remainder = _calculate_n_buffers(
            n_trials=n_trials, tw_length=TW_length,
            noise_length=noise_length, idw_length=IDW_length)

        # create two long random homogeneous poisson spiketrains which represent
        # 40 trials with 1s length and 1.5s background noise in between trials
        spiketrains, st1_long, st2_long = _generate_spiketrains(
            freq=5*pq.Hz, length=(TW_length+noise_length)*n_trials,
            trigger_events=trigger_events,
            injection_pos=(0.6 * pq.s).rescale(time_unit),
            trigger_pre_size=trigger_pre_size,
            trigger_post_size=trigger_post_size,
            time_unit=time_unit)

        # perform standard unitary event analysis
        ue_dict = jointJ_window_analysis(
            spiketrains, bin_size=(0.005 * pq.s).rescale(time_unit),
            win_size=(0.1 * pq.s).rescale(time_unit),
            win_step=(0.005 * pq.s).rescale(time_unit), pattern_hash=[3])

        if method == "pass_events_at_initialization":
            init_events = trigger_events
            reading_events = np.array([]) * time_unit
        elif method == "pass_events_while_buffered_reading":
            init_events = np.array([]) * time_unit
            reading_events = trigger_events
        else:
            raise ValueError("Illeagal method to pass events!")

        # create instance of OnlineUnitaryEventAnalysis
        ouea = OnlineUnitaryEventAnalysis(
            bw_size=(0.005 * pq.s).rescale(time_unit),
            trigger_pre_size=trigger_pre_size,
            trigger_post_size=trigger_post_size,
            saw_size=(0.1 * pq.s).rescale(time_unit),
            saw_step=(0.005 * pq.s).rescale(time_unit),
            trigger_events=init_events,
            time_unit=time_unit)
        # perform online unitary event analysis
        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call update_ue()
        _simulate_buffered_reading(n_buffers=n_buffers, ouea=ouea, st1=st1_long,
                                   st2=st2_long, IDW_length=IDW_length,
                                   length_remainder=length_remainder,
                                   events=reading_events)
        ue_dict_online = ouea.get_results()

        # assert equality between result dicts of standard and online ue version
        self._assert_equality_of_result_dicts(
            ue_dict_offline=ue_dict, ue_dict_online=ue_dict_online,
            tol_dict_user={})
        # fixme: larger atol ok?

        # visualize results of online and standard UEA for artifical data
        # _visualize_results_of_offline_and_online_uea(
        #     spiketrains=spiketrains, ue_dict_offline=ue_dict,
        #     ue_dict_online=ue_dict_online, alpha=0.01)

        return ouea

    # test: trial window > in-coming data window    (TW > IDW)
    def test_TW_larger_IDW_artificial_data(self):
        """Test, if online UE analysis is correct when the trial window is
        larger than the in-coming data window with artificial data."""
        idw_length = ([0.995, 0.8, 0.6, 0.3, 0.1, 0.05]*pq.s).rescale(
            self.time_unit)
        for idw in idw_length:
            with self.subTest(f"IDW = {idw}"):
                self._test_unitary_events_analysis_with_artificial_data(
                    idw_length=idw, time_unit=self.time_unit)
                self.doCleanups()

    def test_TW_larger_IDW_real_data(self):
        """Test, if online UE analysis is correct when the trial window is
                larger than the in-coming data window with real data."""
        idw_length = ([2.05, 2., 1.1, 0.8, 0.1, 0.05]*pq.s).rescale(
            self.time_unit)
        for idw in idw_length:
            with self.subTest(f"IDW = {idw}"):
                self._test_unitary_events_analysis_with_real_data(
                    idw_length=idw, time_unit=self.time_unit)
                self.doCleanups()

    # test: trial window = in-coming data window    (TW = IDW)
    def test_TW_as_large_as_IDW_real_data(self):
        """Test, if online UE analysis is correct when the trial window is
                as large as the in-coming data window with real data."""
        idw_length = (2.1*pq.s).rescale(self.time_unit)
        with self.subTest(f"IDW = {idw_length}"):
            self._test_unitary_events_analysis_with_real_data(
                idw_length=idw_length, time_unit=self.time_unit)
            self.doCleanups()

    def test_TW_as_large_as_IDW_artificial_data(self):
        """Test, if online UE analysis is correct when the trial window is
                as large as the in-coming data window with artificial data."""
        idw_length = (1*pq.s).rescale(self.time_unit)
        with self.subTest(f"IDW = {idw_length}"):
            self._test_unitary_events_analysis_with_artificial_data(
                idw_length=idw_length, time_unit=self.time_unit)
            self.doCleanups()

    # test: trial window < in-coming data window    (TW < IDW)
    def test_TW_smaller_IDW_artificial_data(self):
        """Test, if online UE analysis is correct when the trial window is
        smaller than the in-coming data window with artificial data."""
        idw_length = ([1.05, 1.1, 2, 10, 50, 100]*pq.s).rescale(self.time_unit)
        for idw in idw_length:
            with self.subTest(f"IDW = {idw}"):
                self._test_unitary_events_analysis_with_artificial_data(
                    idw_length=idw, time_unit=self.time_unit)
                self.doCleanups()

    def test_TW_smaller_IDW_real_data(self):
        """Test, if online UE analysis is correct when the trial window is
                smaller than the in-coming data window with real data."""
        idw_length = ([2.15, 2.2, 3, 10, 50, 75.6]*pq.s).rescale(self.time_unit)
        for idw in idw_length:
            with self.subTest(f"IDW = {idw}"):
                self._test_unitary_events_analysis_with_real_data(
                    idw_length=idw, time_unit=self.time_unit)
                self.doCleanups()

    def test_pass_trigger_events_while_buffered_reading_real_data(self):
        idw_length = (2.1*pq.s).rescale(self.time_unit)
        with self.subTest(f"IDW = {idw_length}"):
            self._test_unitary_events_analysis_with_real_data(
                idw_length=idw_length,
                method="pass_events_while_buffered_reading",
                time_unit=self.time_unit)
            self.doCleanups()

    def test_pass_trigger_events_while_buffered_reading_artificial_data(self):
        idw_length = (1*pq.s).rescale(self.time_unit)
        with self.subTest(f"IDW = {idw_length}"):
            self._test_unitary_events_analysis_with_artificial_data(
                idw_length=idw_length,
                method="pass_events_while_buffered_reading",
                time_unit=self.time_unit)
            self.doCleanups()

    def test_reset(self):
        idw_length = (2.1*pq.s).rescale(self.time_unit)
        with self.subTest(f"IDW = {idw_length}"):
            ouea = self._test_unitary_events_analysis_with_real_data(
                idw_length=idw_length, time_unit=self.time_unit)
            self.doCleanups()
        # do reset with default parameters
        ouea.reset()
        # check all class attributes
        with self.subTest(f"check 'time_unit'"):
            self.assertEqual(ouea.time_unit, 1*pq.s)
        with self.subTest(f"check 'data_available_in_mv'"):
            self.assertEqual(ouea.data_available_in_mv, None)
        with self.subTest(f"check 'waiting_for_new_trigger'"):
            self.assertEqual(ouea.waiting_for_new_trigger, True)
        with self.subTest(f"check 'trigger_events_left_over'"):
            self.assertEqual(ouea.trigger_events_left_over, True)
        with self.subTest(f"check 'bw_size'"):
            self.assertEqual(ouea.bw_size, 0.005 * pq.s)
        with self.subTest(f"check 'trigger_events'"):
            self.assertEqual(ouea.trigger_events, [])
        with self.subTest(f"check 'trigger_pre_size'"):
            self.assertEqual(ouea.trigger_pre_size, 0.5 * pq.s)
        with self.subTest(f"check 'trigger_post_size'"):
            self.assertEqual(ouea.trigger_post_size, 0.5 * pq.s)
        with self.subTest(f"check 'saw_size'"):
            self.assertEqual(ouea.saw_size, 0.1 * pq.s)
        with self.subTest(f"check 'saw_step'"):
            self.assertEqual(ouea.saw_step, 0.005 * pq.s)
        with self.subTest(f"check 'n_neurons'"):
            self.assertEqual(ouea.n_neurons, 2)
        with self.subTest(f"check 'pattern_hash'"):
            self.assertEqual(ouea.pattern_hash, [3])
        with self.subTest(f"check 'mw'"):
            np.testing.assert_equal(ouea.mw, [[] for _ in range(2)])
        with self.subTest(f"check 'tw_size'"):
            self.assertEqual(ouea.tw_size, 1 * pq.s)
        with self.subTest(f"check 'tw'"):
            np.testing.assert_equal(ouea.tw, [[] for _ in range(2)])
        with self.subTest(f"check 'tw_counter'"):
            self.assertEqual(ouea.tw_counter, 0)
        with self.subTest(f"check 'n_bins'"):
            self.assertEqual(ouea.n_bins, None)
        with self.subTest(f"check 'bw'"):
            self.assertEqual(ouea.bw, None)
        with self.subTest(f"check 'saw_pos_counter'"):
            self.assertEqual(ouea.saw_pos_counter, 0)
        with self.subTest(f"check 'n_windows'"):
            self.assertEqual(ouea.n_windows, 181)
        with self.subTest(f"check 'n_trials'"):
            self.assertEqual(ouea.n_trials, 0)
        with self.subTest(f"check 'n_hashes'"):
            self.assertEqual(ouea.n_hashes, 1)
        with self.subTest(f"check 'method'"):
            self.assertEqual(ouea.method, 'analytic_TrialByTrial')
        with self.subTest(f"check 'n_surrogates'"):
            self.assertEqual(ouea.n_surrogates, 100)
        with self.subTest(f"check 'input_parameters'"):
            self.assertEqual(ouea.input_parameters["pattern_hash"], [3])
            self.assertEqual(ouea.input_parameters["bin_size"], 5 * pq.ms)
            self.assertEqual(ouea.input_parameters["win_size"], 100 * pq.ms)
            self.assertEqual(ouea.input_parameters["win_step"], 5 * pq.ms)
            self.assertEqual(ouea.input_parameters["method"],
                             'analytic_TrialByTrial')
            self.assertEqual(ouea.input_parameters["t_start"], 0 * pq.s)
            self.assertEqual(ouea.input_parameters["t_stop"], 1 * pq.s)
            self.assertEqual(ouea.input_parameters["n_surrogates"], 100)
        with self.subTest(f"check 'Js'"):
            np.testing.assert_equal(ouea.Js, np.zeros((181, 1),
                                                      dtype=np.float64))
        with self.subTest(f"check 'n_exp'"):
            np.testing.assert_equal(ouea.n_exp, np.zeros((181, 1),
                                                         dtype=np.float64))
        with self.subTest(f"check 'n_emp'"):
            np.testing.assert_equal(ouea.n_emp, np.zeros((181, 1),
                                                         dtype=np.float64))
        with self.subTest(f"check 'rate_avg'"):
            np.testing.assert_equal(ouea.rate_avg, np.zeros((181, 1, 2),
                                                            dtype=np.float64))
        with self.subTest(f"check 'indices'"):
            np.testing.assert_equal(ouea.indices, defaultdict(list))


if __name__ == '__main__':
    unittest.main()
