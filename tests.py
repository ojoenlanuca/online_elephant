import unittest
from time import perf_counter_ns, sleep

import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.unitary_event_analysis import jointJ_window_analysis
from elephant.statistics import mean_firing_rate, isi
import viziphant
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


def _generate_spiketrains(freq, length, ts_events, injection_pos):
    """
    Generate two spiketrains from an homogeneous Poisson process with
    injected coincideces.
    """
    st1 = homogeneous_poisson_process(
        rate=freq, t_start=0*pq.s, t_stop=length)
    st2 = homogeneous_poisson_process(
        rate=freq, t_start=0*pq.s, t_stop=length)
    # inject 10 coincidences within a 0.1s interval for each trial
    injection = np.linspace(0, 0.1, 10)*pq.s
    all_injections = np.array([])
    for i in ts_events:
        all_injections = np.concatenate((all_injections, (i+injection_pos)+injection), axis=0)*pq.s
    st1 = st1.duplicate_with_new_data(np.sort(np.concatenate((st1.times, all_injections)))*pq.s)
    st2 = st2.duplicate_with_new_data(np.sort(np.concatenate((st2.times, all_injections)))*pq.s)

    return st1, st2


class TestOnlineUnitaryEventAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rtol = 1e-15
        cls.atol = 1e-15
        cls.num_neurons = 10
        cls.num_buffers = 100
        cls.buff_size = 1  # in sec
        np.random.seed(73)

    def setUp(self):
        pass

    # test sensitivity (low FN rate) with injected coincidences
    def test_sensitivity_with_injected_coincidences(self):
        pass

    # test selectivity (low FP rate) with injected coincidences
    def test_selectivity_with_injected_coincidences(self):
        pass

    # test: trial window > in-coming data window    (TW > IDW)
    def test_TW_larger_IDW(self):
        """Test, if online UE analysis is correct when the trial window is
        larger than the in-coming data window."""
        n_trials = 40
        TW_length = 1 * pq.s  # sec
        IDW_length = 1 * pq.s  # sec
        TS_events = np.arange(0.5, n_trials*2.5, 2.5)*pq.s  # 40 trials with 1s length and 1.5s background noise in between trials
        # create two long random homogeneous poisson spiketrains
        st1_long, st2_long = _generate_spiketrains(
            freq=5*pq.Hz, length=IDW_length * self.num_buffers, ts_events=TS_events,
            injection_pos=0.1*pq.s)
        # stack spiketrains by trial
        st1_stacked = [st1_long.time_slice(t_start=i-TW_length/2, t_stop=i+TW_length/2).time_shift(-i+TW_length/2) for i in TS_events]
        st2_stacked = [st2_long.time_slice(t_start=i-TW_length/2, t_stop=i+TW_length/2).time_shift(-i+TW_length/2) for i in TS_events]

        spiketrains = np.stack((st1_stacked, st2_stacked), axis=1)
        ue_dict = jointJ_window_analysis(spiketrains, bin_size=5*pq.ms, win_size=100*pq.ms, win_step=5*pq.ms)
        viziphant.unitary_event_analysis.plot_ue(spiketrains, Js_dict=ue_dict, significance_level=0.01, unit_real_ids=['1', '2'])
        plt.show()

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call update_ue()
        ouea = OnlineUnitaryEventAnalysis(
            bw_size=0.005*pq.s, ew_pre_size=0.5*pq.s,
            ew_post_size=0.5*pq.s, idw_size=IDW_length,
            saw_size=0.1*pq.s, saw_step=0.005*pq.s,
            mw_size=2.5*IDW_length, significance_level_alpha=0.05,
            target_event=TS_events) # TODO: use one pyhsical unit as standard and rescale others accordingly
        for i in range(self.num_buffers):
            # sleep(1)
            if i == 75:
                print(f"step {i}")  # needed for debugging
            ouea.update_uea(
                spiketrains=[st1_long.time_slice(t_start=i*IDW_length,
                                            t_stop=i*IDW_length+IDW_length),
                             st2_long.time_slice(t_start=i*IDW_length,
                                            t_stop=i*IDW_length+IDW_length)])
            print(f"#buffer = {i}")
        ue_dict_online = ouea.get_results()
        ue_dict_online["input_parameters"]["t_stop"] += ue_dict_online["input_parameters"]["win_size"]  # compensates different win_pos definitions: center(online version) vs. left-edge(standard version)
        viziphant.unitary_event_analysis.plot_ue(spiketrains, Js_dict=ue_dict_online, significance_level=0.01, unit_real_ids=['1', '2'])  # different num of win_pos in plot_ue causes error
        plt.show()

    # test: trial window = in-coming data window    (TW = IDW)
    # test: trial window < in-coming data window    (TW < IDW)





if __name__ == '__main__':
    unittest.main()
