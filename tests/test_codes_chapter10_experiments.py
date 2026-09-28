"""Independent small answers for E10-18..27, including actual PCM readback."""
import math
import unittest
import numpy as np
from codes.examples.chapter10_experiments import run_experiments
from codes.array_tutorial.audio_samples import agc_blocks_case, prepare_exports, read_pcm16


class Chapter10ExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.answers = run_experiments()

    def test_identifiers(self):
        self.assertEqual(list(self.answers), [f'E10-{i}' for i in range(18, 28)])

    def test_rtf_uses_hop_and_duration_weights(self):
        x = self.answers['E10-18']
        self.assertEqual(x['hop_rtf'], .5)
        self.assertEqual(x['wrong_window_rtf'], .125)
        self.assertAlmostEqual(x['aggregate_rtf'], 91/110)
        self.assertEqual(x['unweighted_mean_rtf'], .5)

    def test_subtraction_expectation_then_clipping(self):
        x = self.answers['E10-19']
        self.assertEqual(x['unclipped_power'], [3., -1.])
        self.assertEqual(x['mean_cross_term'], 0)
        self.assertAlmostEqual(x['clipped_mean'], 1.5)

    def test_q15_rounding_and_wide_sum(self):
        x = self.answers['E10-20']
        self.assertEqual((x['round_once'], x['round_each']), (1, 0))
        self.assertEqual(x['two_worst_products_sum'], 2147483648)
        self.assertEqual(x['saturated_result'], 32767)
        self.assertEqual(x['safe_int64_length_bound'], 8589934591)

    def test_dag_parallel_and_serial(self):
        x = self.answers['E10-21']
        self.assertEqual(x['parallel_finish_c_ms'], 42)
        self.assertEqual(x['single_worker_finish_c_ms'], 46)

    def test_phase_exact_clock_and_complex_sum(self):
        x = self.answers['E10-22']
        self.assertAlmostEqual(x['exact_delay_s'], 1/10001)
        for f in (1000, 4000):
            gain = abs((1 + np.exp(-2j*np.pi*f/10001))/2)
            self.assertAlmostEqual(x['frequencies_hz'][str(f)]['exact_amplitude'], gain)
        self.assertAlmostEqual(x['frequencies_hz']['4000']['first_order_db'], -10.2003527182792)

    def test_memory_lifetimes(self):
        x = self.answers['E10-23']
        self.assertEqual(x['in_flight_kib'], 7.5)
        self.assertEqual(x['serial_with_buffers_kib'], 647.5)
        self.assertEqual(x['parallel_with_buffers_kib'], 743.5)

    def test_streaming_gap_recovers_at_valid_support(self):
        x = self.answers['E10-24']
        self.assertEqual(x['complete']['outputs'], [(0, 0.), (1, .5), (2, 1.), (3, 1.5)])
        self.assertEqual(x['missing_index_2']['outputs'], [(0, 0.), (1, None), (2, 1.), (3, 1.5)])

    def test_alias_is_indistinguishable_at_output_samples(self):
        x = self.answers['E10-25']
        np.testing.assert_allclose(x['output'], [1, 0, -1, 0, 1, 0, -1, 0, 1], atol=1e-14)
        self.assertFalse(x['anti_alias_filter_run'])

    def test_telemetry_units(self):
        x = self.answers['E10-26']
        self.assertEqual(x['missing_time_samples'], 160)
        self.assertEqual(x['missing_scalar_values'], 640)
        self.assertEqual(x['waiting_blocks'], 2)
        self.assertEqual(x['expected_first_sample_timestamp_ns'], [0, 10_000_000, 30_000_000])

    def test_agc_independent_block_recurrence_and_pcm(self):
        case = agc_blocks_case()
        files, groups = prepare_exports({'agc_blocks': case})
        source = case['signals']['agc_blocks_input'][0]
        for name, hop, alpha_hop in [('10ms', 160, 160), ('100ms', 1600, 1600), ('100ms_wrong_alpha', 1600, 160)]:
            gain = 1.
            expected = np.empty(32000)
            for start in range(0, 32000, hop):
                peak = max(abs(source[start:start+hop]))
                desired = min(8., .8/peak)
                tau = .02 if desired < gain else .2
                retention = math.exp(-alpha_hop/16000/tau)
                gain = min(retention*gain + (1-retention)*desired, 1/peak, 8.)
                expected[start:start+hop] = source[start:start+hop] * gain
            np.testing.assert_allclose(case['signals']['agc_blocks_'+name][0], expected, atol=2e-15)
            sample_rate, pcm = read_pcm16(files['agc_blocks_'+name+'.wav'][0])
            self.assertEqual((sample_rate, pcm.shape), (16000, (1, 32000)))
            np.testing.assert_array_equal(pcm[0], np.rint(expected*.7*32768)/32768)
            _, reference = read_pcm16(files['agc_blocks_input.wav'][0])
            for window, (start, stop) in case['parameters']['score_windows_samples_half_open'].items():
                ratio = np.linalg.norm(pcm[0,start:stop]) / np.linalg.norm(reference[0,start:stop])
                self.assertAlmostEqual(case['pcm_analysis']['agc_blocks_'+name]['windows'][window]['rms_ratio'], ratio)
        self.assertEqual(groups['agc_blocks']['common_export_gain'], .7)

    def test_agc_block_time_and_preburst_difference(self):
        x = self.answers['E10-27']
        self.assertEqual(x['availability']['10ms']['first_block_end_s'], .01)
        self.assertEqual(x['availability']['100ms']['first_block_end_s'], .1)
        a = x['float_analysis']
        self.assertAlmostEqual(a['agc_blocks_10ms']['windows']['before_burst']['rms_ratio'], 7.4667685408814)
        self.assertAlmostEqual(a['agc_blocks_100ms']['windows']['before_burst']['rms_ratio'], .93293158806913)
