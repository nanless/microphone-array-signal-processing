"""Fixed internal tone-model drift must fail before any asset write."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from codes.chapters.ch11.examples import generate_selection_audio as generator


def snapshot(directory):
    if not directory.exists():
        return None
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}


class SelectionFixturePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = generator.build_fixture()
        cls.buffers, _ = generator.expected_assets()

    def assert_rejected_before_write(self, fixture):
        # Both a new parent chain and an existing valid member set remain
        # unchanged. The negative control replaces a trusted internal model,
        # not an externally configurable fixture or a malicious callback API.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            new = root/'new'/'deep'/'audio'
            existing = root/'existing'
            existing.mkdir()
            for name, data in self.buffers.items():
                (existing/name).write_bytes(data)
            for destination in (new, existing):
                before = snapshot(destination)
                with mock.patch.object(generator, 'build_fixture', return_value=fixture), \
                     mock.patch.object(generator, 'pcm16_bytes') as encode:
                    with self.assertRaises(ValueError):
                        generator.generate(destination)
                    encode.assert_not_called()
                self.assertEqual(snapshot(destination), before)
            self.assertFalse((root/'new').exists())

    def test_every_fixed_parameter_rejects_value_or_type_drift(self):
        leaves = []
        def visit(value, path=()):
            if type(value) is dict:
                for key, child in value.items():
                    visit(child, path+(key,))
            elif type(value) is list:
                for index, child in enumerate(value):
                    visit(child, path+(index,))
            else:
                leaves.append((path, value))
        visit(self.fixture['parameters'])
        for path, value in leaves:
            # Equal-valued int/float and bool controls deliberately test real
            # types rather than relying on Python's numeric equality.
            wrong_values = ((float(value), True, value+1) if type(value) is int else
                            (int(value), True, value+.001) if type(value) is float else
                            (value+' changed', 0))
            for invalid in wrong_values:
                with self.subTest(path=path, invalid=invalid):
                    fixture = copy.deepcopy(self.fixture)
                    owner = fixture['parameters']
                    for part in path[:-1]:
                        owner = owner[part]
                    owner[path[-1]] = invalid
                    self.assert_rejected_before_write(fixture)

    def test_parameter_structure_and_unstated_claims_rejected(self):
        cases = [lambda f: f['parameters'].pop('score'),
                 lambda f: f['parameters'].__setitem__('extra', 'unknown'),
                 lambda f: f['parameters'].__setitem__('source_score', (1600, 30400)),
                 lambda f: f['parameters']['filters']['fir3'].__setitem__('taps', np.array([1., 1., 1.])),
                 lambda f: f['parameters'].__setitem__('component_amplitude', float('nan')),
                 lambda f: f['parameters'].__setitem__('source_duration_s', float('inf')),
                 lambda f: f['parameters']['scene_target_frequencies_hz']['dual'].reverse()]
        for index, change in enumerate(cases):
            with self.subTest(index=index):
                fixture = copy.deepcopy(self.fixture)
                change(fixture)
                self.assert_rejected_before_write(fixture)

    def test_all_signal_and_component_shapes_types_finiteness_and_truth(self):
        for collection, names in [('signals', self.fixture['signals']),
                                  ('components', self.fixture['components'])]:
            for name in names:
                slots = (None,) if collection == 'signals' else ('clean', 'noise')
                for slot in slots:
                    original = self.fixture[collection][name] if slot is None else self.fixture[collection][name][slot]
                    invalid_values = [original.astype(bool), original.astype('float32'),
                                      original.astype('complex128'), original[:-1], original[None, :],
                                      original.tolist()]
                    for index, replacement in enumerate(invalid_values):
                        with self.subTest(collection=collection, name=name, slot=slot, type_case=index):
                            fixture = copy.deepcopy(self.fixture)
                            if slot is None:
                                fixture[collection][name] = replacement
                            else:
                                fixture[collection][name][slot] = replacement
                            self.assert_rejected_before_write(fixture)
                    for location, invalid in [(0, float('nan')), (32007, float('inf')),
                                              (1800, 2.), (31999, float(original[31999])+.0001)]:
                        with self.subTest(collection=collection, name=name, slot=slot, location=location):
                            fixture = copy.deepcopy(self.fixture)
                            array = fixture[collection][name] if slot is None else fixture[collection][name][slot]
                            array[location] = invalid
                            self.assert_rejected_before_write(fixture)

    def test_member_structure_is_fixed_before_encoding(self):
        cases = [lambda f: f.__setitem__('extra', {}),
                 lambda f: f['signals'].pop('selection_single_target'),
                 lambda f: f['signals'].__setitem__('extra', np.zeros(32008)),
                 lambda f: f['components'].pop('selection_dual_fir9'),
                 lambda f: f['components']['selection_single_fir3'].pop('noise'),
                 lambda f: f['components']['selection_single_fir3'].__setitem__('extra', np.zeros(32008))]
        for index, change in enumerate(cases):
            with self.subTest(index=index):
                fixture = copy.deepcopy(self.fixture)
                change(fixture)
                self.assert_rejected_before_write(fixture)

    def test_normal_replay_keeps_all_published_wav_bytes(self):
        directory = generator.ROOT/'codes/chapters/ch11/scenario_audio'
        for name, data in self.buffers.items():
            if name.endswith('.wav'):
                self.assertEqual(data, (directory/name).read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'audio'
            generator.generate(destination)
            before = snapshot(destination)
            generator.generate(destination, check=True)
            self.assertEqual(snapshot(destination), before)


if __name__ == '__main__':
    unittest.main()
