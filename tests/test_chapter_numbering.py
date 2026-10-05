"""Topic preservation during the explicit 12--15 permutation."""
import io
from pathlib import Path
import unittest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject, NullObject
from scripts import build_pdf, build_site, chapter_identity

ROOT = Path(__file__).resolve().parents[1]

class ChapterNumberingTests(unittest.TestCase):
    def test_current_numbers_keep_original_pdf_topics(self):
        expected = {'12_acoustic-imaging.md':14, '13_distributed-enhancement.md':15,
                    '14_appendix-symbols-math.md':12, '15_appendix-guide.md':13}
        for name, topic in expected.items():
            self.assertEqual(build_pdf.chapter_number(name), topic)
        self.assertEqual(set(chapter_identity.LEGACY_HTML_ROUTES), {
            '14_acoustic-imaging.html', '15_distributed-enhancement.html',
            '12_appendix-symbols-math.html', '13_appendix-guide.html'})

    def reader(self, wrong_alias=False):
        writer = PdfWriter()
        for _ in range(2): writer.add_blank_page(width=300, height=400)
        target = ArrayObject([writer.pages[1].indirect_reference, NameObject('/XYZ'),
                              NumberObject(21), NumberObject(317), NullObject()])
        writer.add_named_destination_array('/heading', target)
        if wrong_alias:
            writer.add_named_destination_array('/old', ArrayObject([
                writer.pages[0].indirect_reference, NameObject('/XYZ'),
                NumberObject(21), NumberObject(317), NullObject()]))
        out=io.BytesIO();writer.write(out);out.seek(0)
        return PdfReader(out)

    def test_alias_preserves_exact_page_and_coordinates(self):
        reader=self.reader();writer=PdfWriter(clone_from=reader)
        build_pdf.add_pdf_named_aliases(reader,writer,{'old':'heading'})
        out=io.BytesIO();writer.write(out);out.seek(0);after=PdfReader(out)
        build_pdf.validate_pdf_named_aliases(after,{'old':'heading'})
        destination=build_pdf.heading_destination(after,'old')
        self.assertEqual(after.get_destination_page_number(destination),1)
        self.assertEqual(list(destination.dest_array[1:4]),['/XYZ',21,317])

    def test_wrong_existing_alias_and_missing_target_are_rejected(self):
        reader=self.reader(wrong_alias=True)
        with self.assertRaises(SystemExit):
            build_pdf.add_pdf_named_aliases(reader,PdfWriter(clone_from=reader),{'old':'heading'})
        reader=self.reader()
        with self.assertRaises(SystemExit):
            build_pdf.add_pdf_named_aliases(reader,PdfWriter(clone_from=reader),{'old':'missing'})

    def test_chrome_catalog_destinations_keep_originals_and_new_aliases(self):
        writer=PdfWriter()
        writer.add_blank_page(width=300,height=400)
        target=ArrayObject([writer.pages[0].indirect_reference,NameObject('/XYZ'),
                            NumberObject(17),NumberObject(319),NullObject()])
        writer.root_object[NameObject('/Dests')]=DictionaryObject({
            NameObject('/heading'):target})
        out=io.BytesIO();writer.write(out);out.seek(0);reader=PdfReader(out)
        cloned=PdfWriter(clone_from=reader)
        build_pdf.add_pdf_named_aliases(reader,cloned,{'old':'heading'})
        out=io.BytesIO();cloned.write(out);out.seek(0);after=PdfReader(out)
        build_pdf.validate_pdf_named_aliases(after,{'old':'heading'})
        self.assertEqual(set(after.named_destinations),{'/heading','/old'})

    def test_real_old_exercises_bind_same_current_teaching_topic(self):
        aliases=build_pdf.pdf_alias_targets()
        self.assertEqual(aliases['ch-14-e14-17'],'ch-14-e12-17')
        self.assertEqual(aliases['ch-15-e15-25'],'ch-15-e13-25')
        self.assertEqual(aliases['ch-12-e12-20'],'ch-12-e14-20')
        self.assertEqual(aliases['ch-13-e13-02'],'ch-13-e15-02')
        self.assertEqual(aliases['ch-14-sec-14-1'],'ch-14-sec-12-1')
        self.assertEqual(aliases['ch-15-sec-15-1'],'ch-15-sec-13-1')
