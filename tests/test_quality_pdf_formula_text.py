"""Independent PDF fixtures for the vector-formula text heuristic.

The rectangle is a minimal paint contract, not a claim to render mathematics.
"""
import io
import unittest

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                          NameObject, NumberObject, TextStringObject)

from scripts.quality_check import pdf_formula_index, pdf_formula_page_text, pdf_page_is_empty


def fixture(stream=None):
    writer = PdfWriter()
    page = writer.add_blank_page(300, 400)
    page[NameObject('/StructParents')] = NumberObject(0)
    speech = 'one plus two'
    formula = DictionaryObject({
        NameObject('/Type'): NameObject('/StructElem'),
        NameObject('/S'): NameObject('/Formula'),
        NameObject('/T'): TextStringObject('pdf-math-000001'),
        NameObject('/Pg'): page.indirect_reference,
        NameObject('/K'): NumberObject(0),
        NameObject('/Alt'): TextStringObject(speech),
        NameObject('/ActualText'): TextStringObject(speech),
        NameObject('/Lang'): TextStringObject('en')})
    reference = writer._add_object(formula)
    parents = DictionaryObject({NameObject('/Nums'): ArrayObject([
        NumberObject(0), ArrayObject([reference])])})
    tree = DictionaryObject({NameObject('/Type'): NameObject('/StructTreeRoot'),
                             NameObject('/K'): ArrayObject([reference]),
                             NameObject('/ParentTree'): writer._add_object(parents)})
    writer.root_object[NameObject('/StructTreeRoot')] = writer._add_object(tree)
    content = DecodedStreamObject()
    content.set_data(stream if stream is not None else
                     b'/Formula << /MCID 0 /ActualText (one plus two) >> BDC 10 20 30 40 re f EMC')
    page[NameObject('/Contents')] = writer._add_object(content)
    return writer, formula, tree


def read(writer):
    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return PdfReader(output)


class VectorFormulaTextTests(unittest.TestCase):
    def test_vector_paint_has_no_text_layer_but_keeps_real_semantic_text(self):
        reader = read(fixture()[0])
        before = reader.pages[0].get_contents().get_data()
        self.assertEqual(reader.pages[0].extract_text(), '')
        index = pdf_formula_index(reader)
        self.assertEqual(index, {0: [(0, 'one plus two')]})
        text = pdf_formula_page_text(reader, 0, index[0])
        self.assertEqual(text, 'one plus two')
        self.assertFalse(pdf_page_is_empty(text, 0))
        self.assertEqual(before, reader.pages[0].get_contents().get_data())

    def test_metadata_without_a_real_mark_or_paint_is_rejected(self):
        for stream in (b'10 20 30 40 re f',
                       b'/Formula << /MCID 0 /ActualText (one plus two) >> BDC q Q EMC'):
            with self.subTest(stream=stream):
                reader = read(fixture(stream)[0])
                with self.assertRaisesRegex(ValueError, '绘制'):
                    pdf_formula_page_text(reader, 0, pdf_formula_index(reader)[0])

    def test_lookalike_role_or_wrong_content_speech_is_rejected(self):
        for stream in (
            b'/Figure << /MCID 0 /ActualText (one plus two) >> BDC 10 20 30 40 re f EMC',
            b'/Formula << /MCID 0 /ActualText (wrong) >> BDC 10 20 30 40 re f EMC'):
            with self.subTest(stream=stream):
                reader = read(fixture(stream)[0])
                with self.assertRaisesRegex(ValueError, '说明不符'):
                    pdf_formula_page_text(reader, 0, pdf_formula_index(reader)[0])

    def test_duplicate_mcid_mark_and_duplicate_requested_nodes_are_rejected(self):
        stream = b'/Formula << /MCID 0 /ActualText (one plus two) >> BDC 10 20 30 40 re f EMC '
        reader = read(fixture(stream * 2)[0])
        with self.assertRaisesRegex(ValueError, '唯一'):
            pdf_formula_page_text(reader, 0, pdf_formula_index(reader)[0])
        with self.assertRaisesRegex(ValueError, '重复'):
            pdf_formula_page_text(reader, 0, [(0, 'one plus two')] * 2)

    def test_invalid_identity_language_and_disagreeing_semantics_are_rejected(self):
        for key, value in (('/T', 'formula-lookalike'), ('/Lang', 'zh-CN'),
                           ('/ActualText', 'different'), ('/Alt', '')):
            with self.subTest(key=key):
                writer, formula, _ = fixture()
                formula[NameObject(key)] = TextStringObject(value)
                with self.assertRaisesRegex(ValueError, '身份'):
                    pdf_formula_index(read(writer))

    def test_wrong_parent_tree_membership_is_rejected(self):
        writer, _, tree = fixture()
        tree['/ParentTree']['/Nums'][1] = ArrayObject([])
        with self.assertRaisesRegex(ValueError, '父树'):
            pdf_formula_index(read(writer))

    def test_duplicate_structure_reference_is_rejected(self):
        writer, _, tree = fixture()
        tree['/K'].append(tree['/K'][0])
        with self.assertRaisesRegex(ValueError, '重复'):
            pdf_formula_index(read(writer))

    def test_plain_span_lookalike_cannot_supply_formula_text(self):
        writer, formula, _ = fixture()
        formula[NameObject('/S')] = NameObject('/Span')
        reader = read(writer)
        self.assertEqual(pdf_formula_index(reader), {})
        self.assertEqual(pdf_formula_page_text(reader, 0, []), '')


if __name__ == '__main__':
    unittest.main()
