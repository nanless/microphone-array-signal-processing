"""Independent geometry, PCM, causality and asset checks for moving tracking."""
import hashlib
import io
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np

from codes.chapters.ch09.core.tracking_audio import build_fixture, analyze_array, read_pcm16, frame_truth
from codes.chapters.ch09.examples.chapter09_tracking_audio import generate, SOURCE_PATHS, ROOT


class TrackingAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.buffers, cls.report = build_fixture()

    def test_pcm_layout_duration_and_common_gain(self):
        self.assertEqual(set(self.buffers), {'source.wav', 'array_noisy.wav'})
        for name, channels in [('source.wav', 1), ('array_noisy.wav', 2)]:
            with wave.open(io.BytesIO(self.buffers[name]), 'rb') as reader:
                self.assertEqual((reader.getnchannels(), reader.getnframes(), reader.getframerate(), reader.getsampwidth()),
                                 (channels, 32000, 16000, 2))
            waveform = read_pcm16(self.buffers[name])
            self.assertLessEqual(float(abs(waveform).max()), .7+1/65536)
        self.assertGreater(self.report['common_export_gain'], 0)

    def test_quadratic_retardation_and_trigonometric_sum_rebuild_pcm_samples(self):
        # Independent retarded-flight quadratic, not Newton or frame_truth.
        indices = np.arange(0, 32000, 317); t = indices/16000
        microphones = np.array([[-.05, 0.], [.05, 0.]])
        p0, velocity = np.array([-.8, 1.5]), np.array([.8, 0.])
        q = p0+velocity*t[None, :, None]-microphones[:, None, :]
        dot = q@velocity; a = 343.**2-.8**2
        flight = (np.sqrt(dot**2+a*np.sum(q*q, axis=2))-dot)/a
        u = t-flight
        distance = 343*flight
        rng = np.random.default_rng(9001); phases = rng.uniform(-np.pi, np.pi, 32)
        np.testing.assert_array_equal(phases, self.report['phases_rad'])
        noise = rng.standard_normal((2, 32000))[:, indices]*.01
        # sin(ωu+φ) = sin(ωu)cosφ + cos(ωu)sinφ.
        result = np.zeros_like(u)
        for frequency, phase in zip(range(300, 3401, 100), phases):
            phase_u = 2*math.pi*frequency*u
            result += np.sin(phase_u)*math.cos(phase)+np.cos(phase_u)*math.sin(phase)
        envelope = np.minimum(np.clip(u/.02, 0, 1), np.clip((2-u)/.02, 0, 1))
        envelope *= 1-np.minimum(np.clip((u-.8)/.02, 0, 1), np.clip((1.1-u)/.02, 0, 1))
        expected = (result/math.sqrt(32)*envelope/distance+noise)*self.report['common_export_gain']
        actual = read_pcm16(self.buffers['array_noisy.wav'])[:, indices]
        self.assertLessEqual(float(np.max(abs(expected-actual))), .500001/32768)

    def test_emission_silence_is_not_zero_receiver_noise(self):
        source = read_pcm16(self.buffers['source.wav'])[0]
        self.assertTrue(np.all(source[int(.82*16000):int(1.08*16000)] == 0))
        array = read_pcm16(self.buffers['array_noisy.wav'])
        self.assertGreater(float(np.sum(array[:, 14000:16000]**2)), 0)
        fields = self.report['pcm_analysis']['frames']
        missing_times = [t for t, valid in zip(fields['state_time_s'], fields['observation_valid']) if not valid]
        self.assertEqual(len(missing_times), 24)
        self.assertAlmostEqual(missing_times[0], .83596875)
        self.assertAlmostEqual(missing_times[-1], 1.06596875)
        self.assertTrue(all(reason == 'below_rms_gate' for reason, valid in zip(fields['reason'], fields['observation_valid']) if not valid))

    def test_scores_recomputed_with_independent_center_retarded_geometry(self):
        fields = self.report['pcm_analysis']['frames']; t = np.array(fields['state_time_s'])
        qx = -.8+.8*t; qy = 1.5; a = 343.**2-.8**2
        flight = (np.sqrt((.8*qx)**2+a*(qx*qx+qy*qy))-.8*qx)/a
        bearing = np.rad2deg(np.arctan((-.8+.8*(t-flight))/1.5))
        np.testing.assert_allclose(fields['truth_angle_deg'], bearing, atol=1e-13)
        valid = np.array(fields['observation_valid']); raw = np.array(fields['observation_angle_deg'], float); filtered = np.array(fields['filtered_angle_deg'])
        scores = self.report['pcm_analysis']['scores']
        for field, values, selected in [('raw_valid_rmse_deg', raw, valid), ('filtered_valid_rmse_deg', filtered, valid),
                                         ('filtered_missing_rmse_deg', filtered, ~valid)]:
            numerator = math.fsum(float((values[i]-bearing[i])**2) for i in np.flatnonzero(selected))
            expected = math.sqrt(numerator/int(selected.sum()))
            self.assertAlmostEqual(scores[field], expected, places=12)
        self.assertEqual((scores['valid_observation_count'], scores['missing_observation_count']), (173, 24))

    def test_state_and_availability_times_and_delay_sign(self):
        frames = self.report['pcm_analysis']['frames']
        np.testing.assert_allclose(np.array(frames['available_time_s'])-frames['state_time_s'], 256.5/16000, atol=2e-16)
        self.assertGreater(frames['truth_tau10_samples'][0], 0)
        self.assertLess(frames['truth_tau10_samples'][-1], 0)
        # Far-field arcsine approximates but is not identical to near-field bearing.
        plane = np.rad2deg(np.arcsin(-343*np.array(frames['truth_tau10_samples'])/(16000*.1)))
        error = abs(plane-np.array(frames['truth_angle_deg']))
        self.assertGreater(float(error.max()), 0)
        self.assertLess(float(error.max()), .03)

    def test_measurement_time_tracks_only_observations_and_last_valid_is_held(self):
        for analysis_name in ('float_analysis', 'pcm_analysis'):
            frames = self.report[analysis_name]['frames']
            starts = frames['start_sample']
            valid = frames['observation_valid']
            measured = frames['measurement_time_s']
            last_valid = frames['last_valid_measurement_time_s']
            self.assertEqual(len(starts), len(measured))
            self.assertEqual(len(starts), len(last_valid))
            previous_measurement = None
            for index, start in enumerate(starts):
                state_time = (start + 255.5) / 16000
                available_time = (start + 512) / 16000
                self.assertEqual(frames['state_time_s'][index], state_time)
                self.assertEqual(frames['available_time_s'][index], available_time)
                self.assertGreater(available_time, state_time)
                if valid[index]:
                    self.assertEqual(measured[index], state_time)
                    previous_measurement = state_time
                else:
                    self.assertIsNone(measured[index])
                self.assertEqual(last_valid[index], previous_measurement)

        pcm = self.report['pcm_analysis']['frames']
        missing = [index for index, valid in enumerate(pcm['observation_valid']) if not valid]
        self.assertEqual(missing, list(range(82, 106)))
        self.assertEqual(pcm['last_valid_measurement_time_s'][81], .82596875)
        self.assertEqual([pcm['last_valid_measurement_time_s'][index] for index in missing],
                         [.82596875] * 24)
        self.assertEqual(pcm['state_time_s'][105], 1.06596875)
        self.assertEqual(pcm['last_valid_measurement_time_s'][106], pcm['state_time_s'][106])

    def test_no_observation_has_no_measurement_history(self):
        frames = analyze_array(np.zeros((2, 32000)))['frames']
        self.assertEqual(frames['state_phase'], ['uninitialized'] * 197)
        self.assertEqual(frames['measurement_time_s'], [None] * 197)
        self.assertEqual(frames['last_valid_measurement_time_s'], [None] * 197)
        self.assertEqual(frames['state_time_s'][0], 255.5/16000)
        self.assertEqual(frames['available_time_s'][0], 512/16000)

    def test_estimates_cannot_use_samples_after_window_end(self):
        array = read_pcm16(self.buffers['array_noisy.wav'])
        modified = array.copy(); boundary = 100*160+512; modified[:, boundary:] = 0
        alternate = analyze_array(modified, export_gain=self.report['common_export_gain'])
        for field in ('observation_angle_deg', 'filtered_angle_deg', 'angle_variance_deg2', 'state_phase',
                      'measurement_time_s', 'last_valid_measurement_time_s'):
            self.assertEqual(alternate['frames'][field][:101], self.report['pcm_analysis']['frames'][field][:101])

    def test_gcc_sign_uses_channel1_minus_channel0(self):
        rng = np.random.default_rng(999); reference = .1*rng.normal(size=32000)
        delayed = np.r_[np.zeros(2), reference[:-2]]
        report = analyze_array(np.array([reference, delayed]))
        values = report['frames']['observation_tau10_samples']
        self.assertAlmostEqual(float(np.median(values)), 2., delta=.02)
        self.assertTrue(all(angle < 0 for angle in report['frames']['observation_angle_deg']))

    def test_check_is_read_only_and_binds_sources_and_pcm(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            metadata = generate(directory)
            self.assertEqual(set(p.name for p in directory.iterdir()), {'source.wav', 'array_noisy.wav', 'MANIFEST.json'})
            saved = json.loads((directory/'MANIFEST.json').read_text())
            self.assertEqual(saved['pcm_analysis']['frames']['last_valid_measurement_time_s'][105], .82596875)
            for path in SOURCE_PATHS:
                self.assertEqual(metadata['source_sha256'][path], hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
            generate(directory, check=True)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()})
            manifest = directory/'MANIFEST.json'
            changed = json.loads(manifest.read_text()); changed['source_sha256'][SOURCE_PATHS[0]] = '0'*64
            manifest.write_text(json.dumps(changed))
            previous = manifest.read_bytes()
            with self.assertRaises(ValueError):
                generate(directory, check=True)
            self.assertEqual(previous, manifest.read_bytes())

    def test_check_rejects_missing_audio_and_extras(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name); generate(directory)
            (directory/'extra.wav').write_bytes(b'not an allowed asset')
            with self.assertRaises(ValueError):
                generate(directory, check=True)
            (directory/'extra.wav').unlink(); (directory/'source.wav').unlink()
            with self.assertRaises(ValueError):
                generate(directory, check=True)


if __name__ == '__main__':
    unittest.main()
