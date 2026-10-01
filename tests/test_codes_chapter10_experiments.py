"""Independent answers for E10-18..33 and read-only PCM contract failures."""
import math
import json
import shutil
import struct
import tempfile
import wave
from fractions import Fraction
from pathlib import Path
import unittest
import numpy as np
from codes.chapters.ch10.chapter10_experiments import run_experiments, agc_anchor, noise_mismatch_anchor, ROOT
from codes.chapters.ch10.core.noise_mismatch import build_fixture
from codes.chapters.ch10.examples.generate_noise_mismatch import OUTPUT as NOISE_OUTPUT
from codes.chapters.ch00.core.audio_samples import agc_blocks_case, prepare_exports, read_pcm16


class Chapter10ExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.answers = run_experiments()

    def test_identifiers(self):
        self.assertEqual(list(self.answers), [f'E10-{i}' for i in range(18, 34)])

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

    def test_agc_published_integer_energy_independent(self):
        answer = self.answers['E10-27']
        self.assertTrue(answer['pcm_validation']['scored_from_stored_integer_pcm'])
        root = ROOT / 'codes/chapters/ch10/audio'
        def integers(stem):
            with wave.open(str(root/(stem+'.wav')), 'rb') as reader:
                return struct.unpack('<32000h', reader.readframes(32000))
        reference = integers('agc_blocks_input')
        for stem, score in answer['pcm_analysis'].items():
            samples = integers(stem)
            for key, (start, stop) in answer['parameters']['score_windows_samples_half_open'].items():
                denominator = sum(int(v)**2 for v in reference[start:stop])
                numerator = sum(int(v)**2 for v in samples[start:stop])
                self.assertEqual(score['windows'][key]['input_integer_squared_sum'], denominator)
                self.assertEqual(score['windows'][key]['output_integer_squared_sum'], numerator)
                self.assertAlmostEqual(score['windows'][key]['rms_ratio'], math.sqrt(numerator/denominator))

    def _temporary_agc_tree(self, directory):
        destination = Path(directory)
        manifest = destination / 'ch00/audio/MANIFEST.json'
        manifest.parent.mkdir(parents=True)
        shutil.copyfile(ROOT/'codes/chapters/ch00/audio/MANIFEST.json', manifest)
        audio = destination / 'ch10/audio'
        audio.mkdir(parents=True)
        for path in (ROOT/'codes/chapters/ch10/audio').glob('agc_blocks*.wav'):
            shutil.copyfile(path, audio/path.name)
        return destination, manifest

    def test_agc_corrupt_file_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self._temporary_agc_tree(directory)
            self.assertTrue(agc_anchor(root)['pcm_validation']['checked_read_only'])
            path = root/'ch10/audio/agc_blocks_10ms.wav'
            corrupted = path.read_bytes()[:-2]+b'\x01\x00'
            path.write_bytes(corrupted)
            with self.assertRaises(ValueError):
                agc_anchor(root)
            self.assertEqual(path.read_bytes(), corrupted)

    def test_agc_stale_nan_bool_and_duplicate_manifest_rejected(self):
        for mutation in ('source', 'nan', 'bool', 'duplicate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, path = self._temporary_agc_tree(directory)
                manifest = json.loads(path.read_text())
                if mutation == 'source':
                    manifest['generator_inputs'][next(iter(manifest['generator_inputs']))] = '0'*64
                elif mutation == 'nan':
                    manifest['groups']['agc_blocks']['common_export_gain'] = float('nan')
                elif mutation == 'bool':
                    next(r for r in manifest['files'] if r['group']=='agc_blocks')['channels'] = True
                blob = json.dumps(manifest)
                if mutation == 'duplicate':
                    blob = '{"schema_version": 2,'+blob[1:]
                path.write_text(blob)
                with self.assertRaises(ValueError):
                    agc_anchor(root)
                self.assertEqual(path.read_text(), blob)

    def test_soft_noise_update_exact_fractions(self):
        rows = self.answers['E10-28']['rows']
        p, noise = Fraction(0), Fraction(1)
        for row, indicator, expected in zip(rows, (1, 1, 0),
                                            (Fraction(9,5), Fraction(54,25), Fraction(603,200))):
            p = (p+indicator)/2
            # Average the H1 hold and H0 recursive update directly, rather
            # than reproducing the implementation's combined retention.
            noise = p*noise+(1-p)*(Fraction(4,5)*noise+Fraction(9,5))
            self.assertEqual(noise, expected)
            self.assertEqual(Fraction(row['noise_power_fraction']), noise)
        self.assertFalse(self.answers['E10-28']['complete_mcra_run'])

    def test_finite_tail_independent_reverse_sum(self):
        x = self.answers['E10-29']
        squared = [Fraction(1), Fraction(1,2), Fraction(1,4), Fraction(1,8)]
        running, reverse = Fraction(0), []
        for value in reversed(squared):
            running += value
            reverse.insert(0, running)
        self.assertEqual([Fraction(v) for v in x['retained_tail_energy_fractions']], reverse)
        full_slope = 10*math.log10(.5)/.1
        self.assertAlmostEqual(x['infinite_tail_t60_s'], -60/full_slope)
        finite_slope = 10*math.log10(float(reverse[-1]/reverse[0]))/.3
        self.assertAlmostEqual(x['finite_endpoint_extrapolated_t60_s'], -60/finite_slope)
        self.assertLess(x['finite_endpoint_extrapolated_t60_s'], x['infinite_tail_t60_s'])
        self.assertFalse(x['rt20_measurement'])

    def test_nonpreemptive_timeline_and_bound_are_distinct(self):
        x = self.answers['E10-30']
        job = x['nonpreemptive']
        self.assertEqual(job['B_start_ms']+x['tasks']['B']['service_ms'], job['A_start_ms'])
        self.assertEqual(job['A_finish_ms']-job['A_release_ms'], 16)
        self.assertGreater(job['response_ms'], x['tasks']['A']['deadline_ms'])
        self.assertEqual(x['response_upper_bound_ms'], 17)

    def test_external_clock_ambiguity_exact_and_common_scale_cancellation(self):
        x = self.answers['E10-31']
        device2 = Fraction(100020001, 100000000)
        timer2 = Fraction(10001,10000)
        self.assertEqual(device2/timer2, Fraction(10001,10000))
        self.assertEqual([r['observed_rate_ratio_fraction'] for r in x['scenarios']], ['10001/10000']*2)
        self.assertEqual(x['timestamp_slope_fraction'], '1/10001')
        for timer in (Fraction(1), Fraction(17,13)):
            self.assertEqual((device2/timer)/(timer2/timer), device2/timer2)

    def test_original_wrapper_padding_is_not_source_time(self):
        x = self.answers['E10-32']
        self.assertEqual(x['call_start_samples'], [0,256,512,768,1024])
        self.assertEqual(x['call_count'], 5)
        self.assertEqual(x['printed_denominator_samples'], 1280)
        self.assertEqual(x['cropped_output_samples'], 1000)
        self.assertEqual(x['printed_rtf'], .125)
        self.assertEqual(x['source_duration_rtf'], .16)

    def test_noise_expected_power_integrates_entire_window(self):
        fixture = build_fixture()
        centers = fixture['spectral']['frame_centers_samples']
        power = fixture['spectral']['known_variance_power']
        self.assertAlmostEqual(power[list(centers).index(10000//128*128)], .1728)
        self.assertAlmostEqual(power[list(centers).index(23040)], 2.7648)
        # Independent trigonometric window, with variance changed at the
        # middle sample; no call to periodic_hann or the spectrum kernel.
        total = sum((.03**2 if i < 256 else .12**2)
                    *(.5-.5*math.cos(2*math.pi*i/512))**2 for i in range(512))
        self.assertAlmostEqual(total, 1.47555)
        self.assertAlmostEqual(power[list(centers).index(19200)], total)

    def test_noise_conditioned_components_keep_finite_cross_term(self):
        fixture = build_fixture()
        answer = self.answers['E10-33']
        for window, (start, stop) in fixture['parameters']['score_windows'].items():
            reference = fixture['signals']['noise_reference'][0, start:stop]
            for stem, parts in fixture['components'].items():
                distortion = parts['target'][0, start:stop]-reference
                residual = parts['noise'][0, start:stop]
                # Direct sample-by-sample identity, not a sum of report fields.
                mse = sum(float(d+n)**2 for d, n in zip(distortion, residual))/(stop-start)
                measured = answer['floating_point']['score_windows'][window]['scores'][stem+'.wav']
                self.assertAlmostEqual(mse, measured['mse'], places=14)
                cross = 2*sum(float(d)*float(n) for d,n in zip(distortion,residual))/(stop-start)
                self.assertAlmostEqual(cross, measured['twice_cross_term'], places=14)
        self.assertLess(answer['floating_point']['score_windows']['before_step']['scores']
                        ['noise_polluted.wav']['twice_cross_term'], 0)

    def test_noise_published_pcm_integer_scores(self):
        answer = self.answers['E10-33']
        self.assertTrue(answer['pcm_validation']['scored_from_stored_integer_pcm'])
        integers = {}
        for path in NOISE_OUTPUT.glob('*.wav'):
            with wave.open(str(path), 'rb') as reader:
                self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getnframes()),
                                 (16000,1,32000))
                integers[path.name] = struct.unpack('<32000h',reader.readframes(32000))
        for window, (start,stop) in answer['parameters']['score_windows'].items():
            measured = answer['pcm_analysis']['score_windows'][window]
            reference = integers['noise_reference.wav'][start:stop]
            denominator = sum(v*v for v in reference)
            self.assertEqual(denominator, 71466999200)
            self.assertEqual(measured['reference_squared_sum_pcm_integer'], denominator)
            for filename, row in measured['scores'].items():
                numerator = sum((v-r)**2 for v,r in zip(integers[filename][start:stop],reference))
                self.assertEqual(row['error_squared_sum_pcm_integer'], numerator)
                self.assertEqual(row['mse'], numerator/((stop-start)*32768**2))
                self.assertEqual(row['nmse'], numerator/denominator)

    def test_noise_missing_assets_are_not_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory)/'not-generated'
            with self.assertRaises(ValueError):
                noise_mismatch_anchor(missing)
            self.assertFalse(missing.exists())

    def test_noise_corrupt_pcm_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'assets'
            shutil.copytree(NOISE_OUTPUT,destination)
            self.assertTrue(noise_mismatch_anchor(destination)['pcm_validation']['checked_read_only'])
            path = destination/'noise_fixed.wav'
            changed = bytearray(path.read_bytes())
            changed[44+2*10000] ^= 1
            path.write_bytes(changed)
            with self.assertRaises(ValueError):
                noise_mismatch_anchor(destination)
            self.assertEqual(path.read_bytes(), changed)
