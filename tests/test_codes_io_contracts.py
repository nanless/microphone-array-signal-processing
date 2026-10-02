"""Finite IO contracts: hostile paths never change an external or prior file."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from codes.chapters.ch00.io_contracts import (
    validate_parent_chain, validate_asset_directory, strict_json_loads,
    same_metadata, validate_report_destination, write_json_report,
)


class IOContractsTest(unittest.TestCase):
    def test_json_rejects_duplicates_nonfinite_overflow_and_bad_utf8(self):
        for text in ('{"a":1,"a":2}', '{"outer":{"x":0,"x":1}}',
                     '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
                     '{"x":1e999}', b'\xff', 'not JSON'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json_loads(text)
        self.assertEqual(strict_json_loads(b'{"x":[null,true,1,0.5]}'),
                         {'x':[None,True,1,.5]})

    def test_metadata_types_do_not_coerce_boolean_or_integer(self):
        self.assertFalse(same_metadata({'a':[True]}, {'a':[1]}))
        self.assertFalse(same_metadata(1, 1.))
        self.assertTrue(same_metadata(1, 1., allow_int_for_float=True))
        self.assertFalse(same_metadata(True, 1., allow_int_for_float=True))
        self.assertFalse(same_metadata({'a':1, 'b':2}, {'a':1}))
        self.assertFalse(same_metadata(float('nan'), float('nan')))

    def test_ancestor_link_and_lexical_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); real=root/'real';real.mkdir();link=root/'link';link.symlink_to(real)
            for path in (link/'new', link/'..'/'new', root/'..'/'new'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_parent_chain(path)
            self.assertFalse((real/'new').exists())
            self.assertEqual(validate_parent_chain(root/'new'),root/'new')

    def test_asset_exact_set_and_regular_single_link_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';out.mkdir();outside=root/'outside';outside.write_bytes(b'old')
            validate_asset_directory(out, ('one','two'), check=False)
            (out/'one').write_bytes(b'a');(out/'two').write_bytes(b'b')
            validate_asset_directory(out, ('one','two'), check=True)
            for kind in ('link','hardlink','directory','extra','missing'):
                with self.subTest(kind=kind):
                    (out/'two').unlink()
                    if kind=='link':(out/'two').symlink_to(outside)
                    elif kind=='hardlink':os.link(outside,out/'two')
                    elif kind=='directory':(out/'two').mkdir()
                    elif kind=='extra':(out/'two').write_bytes(b'b');(out/'extra').write_bytes(b'x')
                    for checking in (True,False):
                        with self.assertRaises(ValueError):validate_asset_directory(out,('one','two'),check=checking)
                    if (out/'two').is_dir():(out/'two').rmdir()
                    elif (out/'two').exists() or (out/'two').is_symlink():(out/'two').unlink()
                    (out/'extra').unlink(missing_ok=True);(out/'two').write_bytes(b'b')
            self.assertEqual(outside.read_bytes(),b'old')
            with self.assertRaises(ValueError):validate_asset_directory(out,('one','two'),check='yes')

    def test_report_preflight_protected_paths_and_serialization_are_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);report=root/'report.json';report.write_bytes(b'old')
            original=(report.read_bytes(),report.stat().st_mtime_ns)
            for data in ({'x':float('nan')},{'x':object()}):
                with self.assertRaises(ValueError):write_json_report(report,data)
                self.assertEqual((report.read_bytes(),report.stat().st_mtime_ns),original)
            cache=root/'cache';cache.mkdir();lock=root/'LOCK.json';lock.write_bytes(b'lock')
            for target in (cache/'report.json',cache,lock):
                with self.assertRaises(ValueError):validate_report_destination(target,forbidden_roots=(cache,lock))
            with self.assertRaises(ValueError):validate_report_destination(root/'future',forbidden_roots=(root/'future/cache',))
            new=root/'new/report.json'
            with self.assertRaises(ValueError):write_json_report(new,{'x':float('inf')})
            self.assertFalse(new.parent.exists())
            write_json_report(report,{'x':1})
            self.assertEqual(json.loads(report.read_text()),{'x':1})
            self.assertEqual(list(root.glob('.report-*')),[])

    def test_report_links_dangling_links_and_special_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);outside=root/'outside';outside.write_bytes(b'untouched')
            for kind in ('symlink','dangling','hardlink','fifo'):
                target=root/kind
                if kind=='symlink':target.symlink_to(outside)
                elif kind=='dangling':target.symlink_to(root/'absent')
                elif kind=='hardlink':os.link(outside,target)
                else:os.mkfifo(target)
                with self.subTest(kind=kind),self.assertRaises(ValueError):write_json_report(target,{'ok':1})
                target.unlink()
            self.assertEqual(outside.read_bytes(),b'untouched')


if __name__=='__main__':unittest.main()
