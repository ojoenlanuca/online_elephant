import unittest
from time import perf_counter_ns

import neo
import numpy as np
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.statistics import mean_firing_rate, isi

from analysis import OnlineMeanFiringRate, OnlineInterSpikeInterval, \
    OnlinePearsonCorrelationCoefficient


class TestOnlineMeanFiringRate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(73)
        cls.rtol = 1e-15
        cls.atol = 1e-15
        cls.num_neurons = 10
        cls.num_buffers = 100
        cls. buff_size = 1  # in sec

    def test_correctness_single_neuron(self):
        """Test, if results of online and normal mean firing rate function
        are identical for a single neuron."""
        # create list of spiketrains
        st_list = [homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call calculate_mfr()
        omfr_neuron1 = OnlineMeanFiringRate()
        tic1 = perf_counter_ns()
        for st in st_list:
            omfr_neuron1.calculate_mfr(spike_buffer=st)
        final_online_mfr = omfr_neuron1.current_mfr
        toc1 = perf_counter_ns()

        # concatenate list of spiketrains to one single spiketrain
        # and call 'offline' mean_firing_rate()
        t_start_first_st = st_list[0].t_start
        t_stop_last_st = st_list[-1].t_stop
        concatenated_st = neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st_list])*pq.s,
            t_start=t_start_first_st, t_stop=t_stop_last_st)
        tic2 = perf_counter_ns()
        normal_mfr = mean_firing_rate(concatenated_st).rescale(pq.Hz)
        toc2 = perf_counter_ns()

        # compare results of normal mfr and online mfr
        print(f"online_mfr = {final_online_mfr} | run-time: {(toc1-tic1)*1e-9}sec\n"
              f"normal_mfr = {normal_mfr} | run-time: {(toc2-tic2)*1e-9}sec\n"
              f"MFR1 Speed factor (t_online / t_normal) {(toc1-tic1)/(toc2-tic2)}")
        np.testing.assert_allclose(
            final_online_mfr.magnitude, normal_mfr.magnitude,
            rtol=self.rtol, atol=self.atol)

    def test_correctness_multiple_neurons(self):
        """Test, if results of online and normal mean firing rate function
        are identical for multiple neurons."""
        # create list of spiketrain lists
        list_of_st_lists = [[homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)] for j in range(self.num_neurons)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call calculate_mfr()
        omfr_all_neurons = [OnlineMeanFiringRate()
                            for j in range(self.num_neurons)]
        tic1 = perf_counter_ns()
        for j, st_list in enumerate(list_of_st_lists):
            for st in st_list:
                omfr_all_neurons[j].calculate_mfr(spike_buffer=st)
        final_online_mfr = [omfr_all_neurons[j].current_mfr
                            for j in range(self.num_neurons)]
        toc1 = perf_counter_ns()

        # concatenate each list of spiketrains to one spiketrain per neuron
        # and call 'offline' mean_firing_rate()
        t_start_first_st = list_of_st_lists[0][0].t_start
        t_stop_last_st = list_of_st_lists[0][-1].t_stop
        concatenated_st = [
            neo.SpikeTrain(np.concatenate([np.asarray(st) for st in st_list])*pq.s,
                           t_start=t_start_first_st, t_stop=t_stop_last_st)
            for st_list in list_of_st_lists]
        tic2 = perf_counter_ns()
        normal_mfr = [mean_firing_rate(concatenated_st[j]).rescale(pq.Hz)
                      for j in range(self.num_neurons)]
        toc2 = perf_counter_ns()

        # compare results of normal mfr and online mfr
        print(f"online_mfr:  | run-time: {(toc1-tic1)*1e-9}sec\n"
              f"normal_mfr:  | run-time: {(toc2-tic2)*1e-9}sec\n"
              f"MFR2 Speed factor (t_online / t_normal): {(toc1-tic1)/(toc2-tic2)}")
        for j in range(self.num_neurons):
            np.testing.assert_allclose(
                final_online_mfr[j].magnitude, normal_mfr[j].magnitude,
                rtol=self.rtol, atol=self.atol)


class TestOnlineInterSpikeInterval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(73)
        cls.rtol = 1e-15
        cls.atol = 1e-15
        cls.num_neurons = 10
        cls.num_buffers = 100
        cls.buff_size = 1  # in sec

    def test_correctness_single_neuron(self):
        """Test, if results of online and normal inter-spike interval
        function are identical for single neuron."""
        # create list of spiketrains
        st_list = [homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s,
            t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call online_isi()
        oisi_neuron1 = OnlineInterSpikeInterval()
        tic1 = perf_counter_ns()
        for st in st_list:
            oisi_neuron1.calculate_isi(spike_buffer=st)
        final_online_isi = oisi_neuron1.current_isi
        toc1 = perf_counter_ns()

        # concatenate list of spiketrains to one single spiketrain
        # and call 'offline' isi()
        t_start_first_st = st_list[0].t_start
        t_stop_last_st = st_list[-1].t_stop
        concatenated_st = neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st_list]) * pq.s,
            t_start=t_start_first_st, t_stop=t_stop_last_st)
        tic2 = perf_counter_ns()
        normal_isi = isi(concatenated_st).magnitude.tolist()
        toc2 = perf_counter_ns()

        # compare results of normal isi and online isi
        print(f"online_isi:  | run-time: {(toc1-tic1)*1e-9}sec\n"
              f"normal_isi:  | run-time: {(toc2-tic2)*1e-9}sec\n"
              f"ISI1 Speed factor (t_online / t_normal): {(toc1-tic1)/(toc2-tic2)}")
        np.testing.assert_allclose(
            final_online_isi, normal_isi,
            rtol=self.rtol, atol=self.atol)

    def test_correctness_multiple_neurons(self):
        """Test, if results of online and normal inter-spike interval
        function are identical multiple neurons."""
        # create list of spiketrain lists
        list_of_st_lists = [[homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s, t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)] for j in range(self.num_neurons)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over spiketrain list and call calculate_isi()
        oisi_all_neurons = [OnlineInterSpikeInterval() for j in range(self.num_neurons)]
        tic1 = perf_counter_ns()
        for j, st_list in enumerate(list_of_st_lists):
            for st in st_list:
                oisi_all_neurons[j].calculate_isi(spike_buffer=st)
        final_online_isi = [oisi_all_neurons[j].current_isi for j in range(self.num_neurons)]
        toc1 = perf_counter_ns()

        # concatenate list of spiketrains to one single spiketrain
        # and call 'offline' isi()
        t_start_first_st = list_of_st_lists[0][0].t_start
        t_stop_last_st = list_of_st_lists[0][-1].t_stop
        concatenated_st = [neo.SpikeTrain(
            np.concatenate([np.asarray(st) for st in st_list]) * pq.s,
            t_start=t_start_first_st, t_stop=t_stop_last_st) for st_list in list_of_st_lists]
        tic2 = perf_counter_ns()
        normal_isi = [isi(concatenated_st[j]).magnitude.tolist() for j in range(self.num_neurons)]
        toc2 = perf_counter_ns()

        # compare results of normal isi and online isi
        print(f"online_isi:  | run-time: {(toc1-tic1)*1e-9}sec\n"
              f"normal_isi:  | run-time: {(toc2-tic2)*1e-9}sec\n"
              f"ISI2 Speed factor (t_online / t_normal): {(toc1-tic1)/(toc2-tic2)}")
        for j in range(self.num_neurons):
            np.testing.assert_allclose(
                final_online_isi[j], normal_isi[j],
                rtol=self.rtol, atol=self.atol)


class TestOnlinePearsonCorrelationCoefficient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # np.random.seed(73)
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
        Opcc = OnlinePearsonCorrelationCoefficient()
        tic1 = perf_counter_ns()
        online_pcc = Opcc.calculate_pcc(binned_spike_buffer=binned_st)
        toc1 = perf_counter_ns()

        # call normal correlation_coefficient()
        tic2 = perf_counter_ns()
        normal_pcc = correlation_coefficient(binned_spiketrain=binned_st)
        toc2 = perf_counter_ns()

        # compare online and normal pcc results
        print(f"online pcc: {online_pcc} | run-time: {(toc1-tic1)*1e-9}sec\n"
              f"normal pcc: {normal_pcc[0][1]} | run-time: {(toc2-tic2)*1e-9}sec\n"
              f"PCC1 Speed factor (t_online / t_normal): {(toc1-tic1)/(toc2-tic2)}")
        np.testing.assert_allclose(online_pcc, normal_pcc[0][1],
                                   rtol=self.rtol, atol=self.atol)

    def test_correctness_with_two_neurons_and_multiple_loops(self):
        """Test, if results of online and normal function of
        Pearson's correlation coefficient are identical for two neurons and
        multiple loops / buffer calls."""
        # create list of spiketrains
        st1_list = [homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s, t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)]
        st2_list = [homogeneous_poisson_process(
            50*pq.Hz, t_start=self.buff_size*i*pq.s, t_stop=(self.buff_size*i+self.buff_size)*pq.s)
            for i in range(self.num_buffers)]
        # create list of binned spiketrains
        binned_st_list = [BinnedSpikeTrain([st1, st2], bin_size=5*pq.ms)
                     for st1, st2 in zip(st1_list, st2_list)]

        # simulate buffered reading/transport of spiketrains,
        # i.e. loop over binned spiketrain list and call calculate_pcc()
        Opcc = OnlinePearsonCorrelationCoefficient()
        tic1 = perf_counter_ns()
        for binned_st in binned_st_list:
            Opcc.calculate_pcc(binned_spike_buffer=binned_st)
        online_pcc = Opcc.R_xy
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
        normal_pcc = correlation_coefficient(binned_spiketrain=concatenated_binned_st)
        toc2 = perf_counter_ns()

        # compare online and normal pcc results
        print(f"online pcc: {online_pcc} | run-time: {(toc1-tic1)*1e-9}sec\n"
              f"normal pcc: {normal_pcc[0][1]} | run-time: {(toc2-tic2)*1e-9}sec\n"
              f"PCC2 Speed factor (t_online / t_normal): {(toc1-tic1)/(toc2-tic2)}\n")
        np.testing.assert_allclose(online_pcc, normal_pcc[0][1],
                                   rtol=self.rtol, atol=self.atol)


if __name__ == '__main__':
    unittest.main()