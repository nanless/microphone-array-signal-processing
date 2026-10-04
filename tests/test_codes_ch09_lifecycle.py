"""Independent integer boundary and actual-PCM lifecycle evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from codes.chapters.ch09.core.lifecycle import teaching_lifecycle
from codes.chapters.ch09.examples.chapter09_tracking_audio import generate
from codes.chapters.ch09.examples.tracking_lifecycle_demo import run_demo


def frames(starts, valid, window=512):
    return {'start_sample': starts, 'observation_valid': valid,
        'state_time_s':[(2*s+window-1)/32000 for s in starts],
        'available_time_s':[2*(s+window)/32000 for s in starts]}


class LifecycleTests(unittest.TestCase):
    def test_real_pcm_events_confirmation_delay_and_no_id_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)/'audio';generate(directory)
            saved={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()}
            result=run_demo(directory)
            self.assertEqual(saved,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()})
        events=result['lifecycle']['events']
        self.assertEqual([(e['frame'],e['event'],e['track_id']) for e in events],
            [(0,'candidate',1),(2,'confirmed',1),(100,'expired',1),(106,'candidate',2),(108,'confirmed',2)])
        self.assertEqual(events[1]['state_tick'],1151)
        self.assertEqual(events[1]['available_tick'],1664)
        self.assertEqual(events[1]['publication_time_s'],.052)
        self.assertEqual(events[2]['age_ticks'],6593)
        self.assertEqual(events[2]['publication_time_s'],1.032)
        self.assertEqual(events[-1]['publication_time_s'],1.112)
        rows=result['lifecycle']['rows']
        self.assertFalse(rows[0]['publish_confirmed']);self.assertFalse(rows[1]['publish_confirmed'])
        self.assertTrue(rows[2]['publish_confirmed']);self.assertEqual(rows[81]['last_valid_measurement_tick'],26431)
        self.assertEqual(rows[99]['age_ticks'],6273);self.assertIsNone(rows[100]['track_id'])
        self.assertEqual(result['analysis_scores']['valid_observation_count'],173)
        self.assertEqual(result['analysis_scores']['missing_observation_count'],24)
        comparison=result['state_age_comparison']
        self.assertEqual(comparison['rows'][101]['age_ticks'],6400)
        self.assertEqual(comparison['rows'][101]['phase'],'confirmed')
        self.assertEqual(comparison['events'][2]['frame'],102)
        self.assertEqual(comparison['events'][2]['age_ticks'],6720)

    def test_equal_boundary_retained_and_one_tick_after_expires(self):
        # window1 centre is an exact sample; availability remains distinct.
        item=teaching_lifecycle(frames([0,1,2,3202,3203],[True,True,True,False,False],window=1),
                                window_samples=1,age_clock='state')
        self.assertEqual(item['rows'][3]['age_ticks'],6400)
        self.assertEqual(item['rows'][3]['phase'],'confirmed')
        self.assertEqual(item['events'][-1]['age_ticks'],6402)
        self.assertEqual(item['events'][-1]['frame'],4)

    def test_candidate_miss_resets_confirmation_and_allocates_fresh_id(self):
        item=teaching_lifecycle(frames([0,160,320,480,640,800],[True,True,False,True,True,True]))
        self.assertEqual([(e['event'],e['track_id']) for e in item['events']],
                         [('candidate',1),('candidate_lost',1),('candidate',2),('confirmed',2)])
        self.assertEqual(item['events'][-1]['frame'],5)

    def test_expiry_precedes_a_valid_reacquisition_after_a_long_jump(self):
        item=teaching_lifecycle(frames([0,160,320,10000],[True]*4))
        self.assertEqual(item['rows'][-1]['events'],['expired','candidate'])
        self.assertEqual(item['rows'][-1]['phase'],'tentative')
        self.assertEqual(item['rows'][-1]['track_id'],2)

    def test_uninitialized_missing_has_no_measurement_or_identity(self):
        item=teaching_lifecycle(frames([0,160],[False,False]))
        self.assertEqual(item['events'],[])
        self.assertTrue(all(row['last_valid_measurement_tick'] is None for row in item['rows']))
        self.assertEqual(item['counts']['allocated_local_ids'],0)

    def test_invalid_types_or_false_clocks_are_rejected_without_mutation(self):
        baseline=frames([0,160],[True,False])
        for field,value in [('start_sample',[0,True]),('observation_valid',[True,1]),
                            ('state_time_s',[False,.02596875]),('available_time_s',[.0,.042])]:
            item=copy.deepcopy(baseline);item[field]=value;saved=copy.deepcopy(item)
            with self.assertRaises(ValueError):teaching_lifecycle(item)
            self.assertEqual(item,saved)
        for kwargs in ({'sample_rate_hz':True},{'confirmation_frames':True},{'maximum_age_ticks':.2},
                       {'maximum_age_ticks':-1},{'age_clock':'publish_late'},{'window_samples':0},
                       {'maximum_age_ticks':512}):
            with self.assertRaises(ValueError):teaching_lifecycle(baseline,**kwargs)

    def test_unresolvable_float_display_clock_is_not_reported_as_equal_epochs(self):
        # Integer ages are exact, but a public seconds field must not falsely
        # display availability as the same epoch as its earlier state centre.
        with self.assertRaises(ValueError):
            teaching_lifecycle(frames([2**62],[True]))
        with self.assertRaises(ValueError):
            teaching_lifecycle(frames([0],[True]),sample_rate_hz=10**400)
