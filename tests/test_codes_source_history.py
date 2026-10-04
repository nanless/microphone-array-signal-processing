"""Independent byte provenance and policy-change controls for historical reports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codes.chapters.ch00.core import source_history as history

OLD_LOCK_SHA = '55ab323ba665633141c4864763095046f9c6161ce2d88ca2aa9332dde7ec23f0'
OLD_STATUS_SHA = 'e3b3176d835837441224e4906b7c2befadcdc4fe2ce6163245b7d9a9ad0d9229'
HARMONY_LOCK_SHA = 'e3478006c7dbc6cec442bf6bccc4df9eca946d7d353b661e947596dd8b87608a'
HARMONY_STATUS_SHA = 'b113b63c97767d19b76ceb44677303961ff310ce8d96f4e9e777944b44916d3c'


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SourceHistoryFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.snapshots=self.root/'snapshots';self.snapshots.mkdir()
        self.lock=self.root/'SOURCES.lock.json';self.status=self.root/'SOURCE_STATUS.json'
        self.registry={'SOURCES':set(),'SOURCE_STATUS':set()}
        registration=patch.object(history,'_HISTORICAL_SNAPSHOTS',self.registry)
        registration.start();self.addCleanup(registration.stop)
        self.entry={'id':'used','url':'https://example.org/official.git','revision':'a'*40,'license':'MIT',
                    'fetch_enabled':True,'entrypoints':['module.py','LICENSE'],
                    'source_paths':['LICENSE','module.py'],'policy':{'weights':False,'limit':1}}
        self.state={'id':'used','revision':'a'*40,'status':'source_selection_mismatch',
                    'source_selection_verified':False,'execution':'not_run',
                    'requested_source_paths':['module.py','LICENSE'],'observed_sparse_patterns':['/module.py'],
                    'missing_entrypoints':[]}
        self.old_lock={'schema_version':1,'verified_at':'2026-10-02','projects':[copy.deepcopy(self.entry)]}
        self.old_lock_sha=self.snapshot('SOURCES',self.old_lock)
        self.old_status={'schema_version':1,'lock_sha256':self.old_lock_sha,'projects':[copy.deepcopy(self.state)]}
        self.old_status_sha=self.snapshot('SOURCE_STATUS',self.old_status)
        self.new_lock=copy.deepcopy(self.old_lock);extra=copy.deepcopy(self.entry);extra['id']='unused-new'
        self.new_lock['projects'].append(extra);self.new_lock['verified_at']='2026-10-04'
        self.lock.write_bytes(encoded(self.new_lock))
        self.new_status=copy.deepcopy(self.old_status);extra=copy.deepcopy(self.state);extra['id']='unused-new'
        self.new_status['projects'].append(extra);self.new_status['lock_sha256']=sha(self.lock.read_bytes())
        self.status.write_bytes(encoded(self.new_status))

    def snapshot(self,prefix,value):
        raw=encoded(value);digest=sha(raw);(self.snapshots/f'{prefix}.{digest}.json').write_bytes(raw)
        self.registry[prefix].add(digest)
        return digest

    def verify_lock(self,report_sha=None,ids=('used',),**kwargs):
        return history.verify_lock_binding(self.old_lock_sha if report_sha is None else report_sha,ids,
                                           current_lock=self.lock,snapshots=self.snapshots,**kwargs)

    def verify_status(self,status_sha=None,lock_sha=None,ids=('used',)):
        return history.verify_status_binding(self.old_status_sha if status_sha is None else status_sha,
                    self.old_lock_sha if lock_sha is None else lock_sha,ids,current_lock=self.lock,
                    current_status=self.status,snapshots=self.snapshots)

    def test_append_only_preserves_whole_records_and_returns_real_byte_evidence(self):
        result=self.verify_lock()
        self.assertTrue(result['historical']);self.assertEqual(result['records'],{'used':self.entry})
        self.assertEqual(result['report_sha256'],self.old_lock_sha)
        self.assertEqual(result['current_sha256'],sha(self.lock.read_bytes()))
        self.assertEqual(Path(result['snapshot_path']).read_bytes(),encoded(self.old_lock))
        result=self.verify_status()
        self.assertIs(result['records']['used']['source_selection_verified'],False)
        self.assertEqual(result['records']['used']['status'],'source_selection_mismatch')
        self.assertEqual(result['report_lock_sha256'],self.old_lock_sha)
        self.assertEqual(result['current_lock_sha256'],sha(self.lock.read_bytes()))

    def test_current_byte_binding_needs_no_historical_snapshot(self):
        result=self.verify_lock(sha(self.lock.read_bytes()))
        self.assertFalse(result['historical']);self.assertIsNone(result['snapshot_path'])
        result=self.verify_status(sha(self.status.read_bytes()),sha(self.lock.read_bytes()))
        self.assertFalse(result['historical'])

    def test_every_used_entry_field_and_nested_type_is_part_of_contract(self):
        variants={'url':'https://example.org/impostor.git','revision':'b'*40,'license':'GPL-3.0',
                  'fetch_enabled':False,'entrypoints':['other.py'], 'source_paths':['repository'],
                  'policy':{'weights':False,'limit':True}}
        for field,value in variants.items():
            with self.subTest(field=field):
                changed=copy.deepcopy(self.new_lock);changed['projects'][0][field]=value
                self.lock.write_bytes(encoded(changed))
                with self.assertRaises(ValueError):self.verify_lock()
        self.lock.write_bytes(encoded(self.new_lock))

    def test_unknown_short_uppercase_nonstring_and_traversal_sha_are_rejected(self):
        for digest in ('0'*64,'a'*63,'A'*64,'../'+self.old_lock_sha,None,True):
            with self.subTest(digest=digest),self.assertRaises(ValueError):
                history.verify_lock_binding(digest,['used'],current_lock=self.lock,snapshots=self.snapshots)

    def test_ids_must_be_explicit_nonempty_unique_and_known(self):
        for ids in ([],['used','used'],['missing'],['../used'],['/used'],['used/name'],[True],
                    'used',{'used'},None):
            with self.subTest(ids=ids),self.assertRaises(ValueError):self.verify_lock(ids=ids)

    def test_snapshot_name_cannot_launder_modified_bytes(self):
        for prefix,digest,verify in (('SOURCES',self.old_lock_sha,self.verify_lock),
                                     ('SOURCE_STATUS',self.old_status_sha,self.verify_status)):
            with self.subTest(prefix=prefix):
                path=self.snapshots/f'{prefix}.{digest}.json';raw=path.read_bytes()
                try:
                    path.write_bytes(raw+b' ')
                    with self.assertRaisesRegex(ValueError,'snapshot bytes'):verify()
                finally:path.write_bytes(raw)

    def test_unregistered_digest_is_refused_even_with_matching_valid_snapshot_bytes(self):
        for prefix,document in (('SOURCES',self.old_lock),('SOURCE_STATUS',self.old_status)):
            raw=encoded(document)+b' ';digest=sha(raw)
            (self.snapshots/f'{prefix}.{digest}.json').write_bytes(raw)
            with self.subTest(prefix=prefix),self.assertRaisesRegex(ValueError,'registered historical'):
                self.verify_lock(digest) if prefix=='SOURCES' else self.verify_status(digest)

    def test_duplicate_project_ids_rejected_in_current_and_historical_files(self):
        for historical in (False,True):
            for prefix in ('SOURCES','SOURCE_STATUS'):
                with self.subTest(historical=historical,prefix=prefix):
                    document=copy.deepcopy(self.old_lock if prefix=='SOURCES' else self.old_status)
                    document['projects'].append(copy.deepcopy(document['projects'][0]))
                    if historical:
                        digest=self.snapshot(prefix,document)
                        with self.assertRaises(ValueError):
                            self.verify_lock(digest) if prefix=='SOURCES' else self.verify_status(digest)
                    else:
                        path=self.lock if prefix=='SOURCES' else self.status
                        original=path.read_bytes();path.write_bytes(encoded(document))
                        try:
                            with self.assertRaises(ValueError):
                                self.verify_lock() if prefix=='SOURCES' else self.verify_status()
                        finally:path.write_bytes(original)

    def test_duplicate_json_keys_nonfinite_and_wrong_schema_rejected(self):
        bad=(b'{"schema_version":1,"schema_version":1,"projects":[]}',
             b'{"schema_version":1,"projects":[],"invalid":NaN}',
             b'{"schema_version":1,"projects":[],"invalid":1e999}', b'[]',
             encoded({'schema_version':True,'projects':[self.entry]}))
        for raw in bad:
            with self.subTest(raw=raw):
                self.lock.write_bytes(raw)
                with self.assertRaises(ValueError):self.verify_lock()
        self.lock.write_bytes(encoded(self.new_lock))

    def test_status_must_bind_the_reports_exact_historical_lock(self):
        changed=copy.deepcopy(self.old_status);changed['lock_sha256']=sha(self.lock.read_bytes())
        digest=self.snapshot('SOURCE_STATUS',changed)
        with self.assertRaises(ValueError):self.verify_status(digest)
        with self.assertRaises(ValueError):self.verify_status(lock_sha='0'*64)

    def test_current_status_stale_lock_and_false_upgrades_are_rejected(self):
        variants=[copy.deepcopy(self.new_status) for _ in range(3)]
        variants[0]['lock_sha256']=self.old_lock_sha
        variants[1]['projects'][0]['source_selection_verified']=True
        variants[2]['projects'][0]['status']='source_verified'
        for document in variants:
            with self.subTest(document=document):
                self.status.write_bytes(encoded(document))
                with self.assertRaises(ValueError):self.verify_status()
        self.status.write_bytes(encoded(self.new_status))

    def test_status_complete_state_and_boolean_type_are_preserved(self):
        for field,value in [('execution','run'),('missing_entrypoints',['LICENSE']),
                            ('observed_sparse_patterns',['/repository']),('source_selection_verified',0)]:
            changed=copy.deepcopy(self.new_status);changed['projects'][0][field]=value
            self.status.write_bytes(encoded(changed))
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify_status()
        self.status.write_bytes(encoded(self.new_status))

    def test_linked_file_parent_directory_hardlink_and_lexical_traversal_rejected(self):
        link=self.root/'linked-lock';link.symlink_to(self.lock)
        with self.assertRaises(ValueError):history.verify_lock_binding(self.old_lock_sha,['used'],current_lock=link,snapshots=self.snapshots)
        link=self.root/'linked-root';link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):history.verify_lock_binding(self.old_lock_sha,['used'],current_lock=link/self.lock.name,snapshots=self.snapshots)
        link=self.root/'hard-linked';os.link(self.lock,link)
        try:
            with self.assertRaises(ValueError):self.verify_lock()
        finally:link.unlink()
        with self.assertRaises(ValueError):history.verify_lock_binding(self.old_lock_sha,['used'],current_lock=self.root/'x/../SOURCES.lock.json',snapshots=self.snapshots)
        link=self.root/'linked-snapshots';link.symlink_to(self.snapshots,target_is_directory=True)
        with self.assertRaises(ValueError):history.verify_lock_binding(self.old_lock_sha,['used'],current_lock=self.lock,snapshots=link)
        snapshot=self.snapshots/f'SOURCES.{self.old_lock_sha}.json';raw=snapshot.read_bytes()
        snapshot.unlink();snapshot.symlink_to(self.lock)
        with self.assertRaises(ValueError):self.verify_lock()
        snapshot.unlink();snapshot.write_bytes(raw)

    def test_calls_never_write_even_when_validation_fails(self):
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(Path,'write_bytes',side_effect=AssertionError('unexpected write')),\
             patch.object(Path,'write_text',side_effect=AssertionError('unexpected write')),\
             patch.object(Path,'mkdir',side_effect=AssertionError('unexpected mkdir')):
            self.verify_lock();self.verify_status()
            with self.assertRaises(ValueError):self.verify_lock('0'*64)
        self.assertEqual({p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()},before)


class RealHistoricalSourceTests(unittest.TestCase):
    def test_exact_historical_bytes_and_all_hundred_old_entries_are_preserved(self):
        path=history.SNAPSHOT_ROOT/f'SOURCES.{OLD_LOCK_SHA}.json';self.assertEqual(sha(path.read_bytes()),OLD_LOCK_SHA)
        old=json.loads(path.read_text());ids=[p['id'] for p in old['projects']]
        self.assertEqual(len(ids),100);result=history.verify_lock_binding(OLD_LOCK_SHA,ids)
        self.assertEqual(result['records'],{p['id']:p for p in old['projects']})
        current=json.loads(history.LOCK.read_text());self.assertEqual(len(current['projects']),107)
        self.assertEqual({p['id'] for p in current['projects']}-set(ids),
                         {'danse-python','danse-wola','paderwasn','tidanseplus-batch','wasn-platform','libricss',
                          'hybrid-tdoa-multi-calib'})

    def test_chapter03_index_addition_preserves_all_106_previous_source_records(self):
        lock_sha='6dab41b1542cfa2cd4731ef4c8a3e807b343dc807f209c4eea17503e203ebf59'
        status_sha='1142cc2290d93ea33e6b2dcdf72b9b52307efdd837b84c1739262c2c968186d6'
        raw_lock=(history.SNAPSHOT_ROOT/f'SOURCES.{lock_sha}.json').read_bytes()
        raw_status=(history.SNAPSHOT_ROOT/f'SOURCE_STATUS.{status_sha}.json').read_bytes()
        self.assertEqual(sha(raw_lock),lock_sha);self.assertEqual(sha(raw_status),status_sha)
        old=json.loads(raw_lock);ids=[row['id'] for row in old['projects']]
        self.assertEqual(len(ids),106)
        self.assertTrue(history.verify_lock_binding(lock_sha,ids)['historical'])
        result=history.verify_status_binding(status_sha,lock_sha,ids)
        self.assertTrue(result['historical'])
        self.assertEqual(result['records'],{row['id']:row for row in json.loads(raw_status)['projects']})
        current=json.loads(history.LOCK.read_bytes())
        entry=next(row for row in current['projects'] if row['id']=='hybrid-tdoa-multi-calib')
        self.assertEqual(entry['revision'],'4cc21cb06b9f82f83cc90d65a418f2b100748254')
        self.assertIs(entry['fetch_enabled'],False)
        self.assertEqual((entry['license'],entry['acquisition']),('NOASSERTION','index_only'))
        status=json.loads(history.STATUS.read_bytes())
        row=next(row for row in status['projects'] if row['id']==entry['id'])
        self.assertEqual(row,{'id':entry['id'],'revision':entry['revision'],'status':'index_only'})

    def test_previous_105_lock_preserves_104_records_and_rejects_changed_wasn_policy(self):
        path=history.SNAPSHOT_ROOT/f'SOURCES.{HARMONY_LOCK_SHA}.json'
        self.assertEqual(sha(path.read_bytes()),HARMONY_LOCK_SHA)
        old=json.loads(path.read_text());self.assertEqual(len(old['projects']),105)
        records={p['id']:p for p in old['projects'] if p['id']!='wasn-platform'}
        result=history.verify_lock_binding(HARMONY_LOCK_SHA,list(records))
        self.assertTrue(result['historical'])
        self.assertEqual(result['records'],records)
        with self.assertRaisesRegex(ValueError,'project record differs'):
            history.verify_lock_binding(HARMONY_LOCK_SHA,['wasn-platform'])

    def test_previous_105_status_preserves_all_104_actual_states_including_failures(self):
        path=history.SNAPSHOT_ROOT/f'SOURCE_STATUS.{HARMONY_STATUS_SHA}.json'
        self.assertEqual(sha(path.read_bytes()),HARMONY_STATUS_SHA)
        old=json.loads(path.read_text());self.assertEqual(len(old['projects']),105)
        records={p['id']:p for p in old['projects'] if p['id']!='wasn-platform'}
        result=history.verify_status_binding(HARMONY_STATUS_SHA,HARMONY_LOCK_SHA,list(records))
        self.assertTrue(result['historical'])
        self.assertEqual(result['records'],records)
        self.assertEqual(records['aec-challenge']['status'],'failed')
        self.assertEqual(sum(p['status']=='source_selection_mismatch' for p in records.values()),22)
        self.assertNotIn('execution',records['aec-challenge'])
        self.assertTrue(all(p['execution']=='not_run' for p in records.values() if 'execution' in p))

    def test_real_tracking_status_keeps_three_selection_mismatches(self):
        path=history.SNAPSHOT_ROOT/f'SOURCE_STATUS.{OLD_STATUS_SHA}.json';self.assertEqual(sha(path.read_bytes()),OLD_STATUS_SHA)
        ids=['odas','spatial-audio-framework','filterpy','stonesoup']
        result=history.verify_status_binding(OLD_STATUS_SHA,OLD_LOCK_SHA,ids)
        self.assertEqual({k:v['source_selection_verified'] for k,v in result['records'].items()},
                         {'odas':True,'spatial-audio-framework':False,'filterpy':False,'stonesoup':False})
        self.assertTrue(result['historical']);self.assertTrue(result['lock_binding']['historical'])

    def test_unavailable_older_wpe_status_is_not_fabricated(self):
        with self.assertRaises(ValueError):history.verify_status_binding(
            '422868a25676cd436dc23b8e8aa9d907d8cff3273e7bc77975dd1306fb41c3ac',OLD_LOCK_SHA,['nara_wpe','nemo_wpe'])


if __name__=='__main__':unittest.main()
