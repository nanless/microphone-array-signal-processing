"""Trusted internal model drift must fail before any asset write.

These controls are monkeypatches of internal trusted model functions, not claims
that callers can supply arbitrary generation metadata through the CLI.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from codes.chapters.ch08.examples import mask_representation_demo as mask, gss_teaching_demo as gss


class GenerationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask_result = mask.run_experiment()
        cls.gss_result = gss.run_experiment()

    def leaves(self, value, path=()):
        if isinstance(value, dict):
            for key, item in value.items(): yield from self.leaves(item, path+(key,))
        elif isinstance(value, list):
            for key, item in enumerate(value): yield from self.leaves(item, path+(key,))
        else: yield path, value

    def change(self, original, path, value):
        data = copy.deepcopy(original); destination = data
        for key in path[:-1]: destination = destination[key]
        destination[path[-1]] = value
        return data

    def assert_prewrite_rejected(self, module, result):
        generate = module.generate_assets if module is mask else module.generate
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'assets'
            with patch.object(module, 'run_experiment', return_value=result), self.assertRaises(ValueError):
                generate(out)
            self.assertFalse(out.exists())
        # Existing nonempty directory must also stay byte/mtime-identical.
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'assets'; generate(out)
            before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
            with patch.object(module,'run_experiment',return_value=result), self.assertRaises(ValueError): generate(out)
            self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()})

    def test_every_mask_parameter_true_type_or_value_drift_prewrite(self):
        report, arrays = self.mask_result
        for path,value in self.leaves(report['parameters']):
            replacement=(not value if type(value) is bool else True if type(value) in (int,float) else value+' altered')
            bad=copy.deepcopy(report); bad['parameters']=self.change(report['parameters'],path,replacement)
            with self.subTest(path=path): self.assert_prewrite_rejected(mask,(bad,arrays))
        bad=copy.deepcopy(report); bad['limits']='false source scope'
        self.assert_prewrite_rejected(mask,(bad,arrays))

    def test_every_gss_fixed_declaration_true_type_or_value_drift_prewrite(self):
        report,arrays=self.gss_result
        for path,value in self.leaves(gss.REQUIRED_MODEL):
            replacement=(not value if type(value) is bool else True if type(value) in (int,float) else value+' altered')
            bad=self.change(report,path,replacement)
            with self.subTest(path=path): self.assert_prewrite_rejected(gss,(bad,arrays))

    def test_mask_false_float_score_or_array_contract_prewrite(self):
        report,arrays=self.mask_result
        bad=copy.deepcopy(report); bad['float_measurements']['mixture']['total_reference_mse']=.1
        self.assert_prewrite_rejected(mask,(bad,arrays))
        for value in (arrays['mixture'].astype(complex), np.full((1,32000),np.nan),
                      arrays['mixture'][:,:-1], np.full((1,32000),1.1)):
            bad=dict(arrays); bad['mixture']=value
            self.assert_prewrite_rejected(mask,(report,bad))

    def test_gss_false_float_score_and_invalid_arrays_prewrite(self):
        report,arrays=self.gss_result
        bad=copy.deepcopy(report); bad['si_sdr_db']['correct_activity_output']=.1
        self.assert_prewrite_rejected(gss,(bad,arrays))
        bad=copy.deepcopy(report); bad['activity_error']['missed_output_scored_si_sdr_db']=.1
        self.assert_prewrite_rejected(gss,(bad,arrays))
        for value in (arrays['mixture'].astype(complex),np.full((2,32000),np.nan),arrays['mixture'][:,:-1]):
            bad=dict(arrays); bad['mixture']=value
            self.assert_prewrite_rejected(gss,(report,bad))

    def test_joint_mask_runtime_declaration_drift_does_not_define_truth(self):
        report,arrays=self.mask_result
        bad=copy.deepcopy(report); bad['parameters']['common_export_gain']=True
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'assets'
            with patch.object(mask,'parameters',return_value=bad['parameters']), \
                    patch.object(mask,'run_experiment',return_value=(bad,arrays)), self.assertRaises(ValueError):
                mask.generate_assets(out)
            self.assertFalse(out.exists())


if __name__ == '__main__': unittest.main()
