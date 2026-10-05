"""Independent authored-HTML and single-MCID PDF formula fixtures.

The page content below paints an actual black rectangle. These fixtures check
source/structure/content identity; they do not establish speech correctness,
visible mathematical glyphs, assistive-technology behavior or PDF/UA compliance.
"""

import hashlib
import io
import json
import unittest
from html.parser import HTMLParser

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, ContentStream, DecodedStreamObject,
                          DictionaryObject, NameObject, NumberObject,
                          TextStringObject)

from scripts.build_pdf import (label_pdf_math_sources, pdf_math_sources,
                               tag_pdf_formulas)


TEX = '$x+1$'
DIGEST = hashlib.sha256(TEX.encode('utf-8')).hexdigest()
SOURCE = {'id': '000001', 'tex': TEX, 'sha256': DIGEST}


class SourceHTML(HTMLParser):
    """Read only the real JSON script and real span attributes, never regex text."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts = []
        self.spans = []
        self.collecting = False

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == 'script' and fields.get('id') == 'pdf-math-sources':
            self.collecting = True
            self.scripts.append('')
        if tag == 'span' and 'data-pdf-math-id' in fields:
            self.spans.append(fields)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.collecting = False

    def handle_data(self, data):
        if self.collecting:
            self.scripts[-1] += data


def source_page(payload):
    return ('<html><body><script type="application/json" '
            'id="pdf-math-sources">' + json.dumps(payload) + '</script></body></html>')


def formula_fixture(*, alt=None, role='/Figure', content=None, mcid=0):
    """A real page, StructParents/ParentTree array and one painted MCID."""
    writer = PdfWriter()
    page = writer.add_blank_page(100, 100)
    page[NameObject('/StructParents')] = NumberObject(0)
    tree = DictionaryObject({NameObject('/Type'): NameObject('/StructTreeRoot')})
    tree_ref = writer._add_object(tree)
    writer.root_object[NameObject('/StructTreeRoot')] = tree_ref
    document = DictionaryObject({NameObject('/Type'): NameObject('/StructElem'),
                                 NameObject('/S'): NameObject('/Document'),
                                 NameObject('/P'): tree_ref})
    document_ref = writer._add_object(document)
    figure = DictionaryObject({NameObject('/Type'): NameObject('/StructElem'),
                               NameObject('/S'): NameObject(role),
                               NameObject('/P'): document_ref,
                               NameObject('/Pg'): page.indirect_reference,
                               NameObject('/K'): NumberObject(mcid),
                               NameObject('/Alt'): TextStringObject(
                                   alt if alt is not None else
                                   f'MASP-MATH:000001:{DIGEST}:x plus 1')})
    figure_ref = writer._add_object(figure)
    document[NameObject('/K')] = ArrayObject([figure_ref])
    tree[NameObject('/K')] = document_ref
    parent_tree = DictionaryObject({NameObject('/Nums'): ArrayObject([
        NumberObject(0), ArrayObject([figure_ref])])})
    tree[NameObject('/ParentTree')] = writer._add_object(parent_tree)
    stream = DecodedStreamObject()
    stream.set_data(content if content is not None else
                    b'/Figure <</MCID 0>> BDC\n0 0 0 rg\n10 10 20 20 re f\nEMC\n')
    page[NameObject('/Contents')] = writer._add_object(stream)
    return writer, page, tree, figure, figure_ref, parent_tree


class PDFMathSourceTests(unittest.TestCase):
    def test_literal_code_attributes_comments_and_escaped_dollars_are_not_math(self):
        html = (r'<html><head><title>$not body$</title></head><body>'
                r'<p>$x+1$ and $$y^2$$; \$5 and \$6.</p>'
                r'<pre><code>$code$ $$also code$$</code></pre>'
                r'<code>$inline code$</code>'
                r'<a title="$attribute$">literal</a>'
                r'<!-- $comment$ -->'
                r'<p>$a&lt;b$</p></body></html>')
        labelled = label_pdf_math_sources(html)
        reader = SourceHTML()
        reader.feed(labelled)
        self.assertEqual(len(reader.scripts), 1)
        parsed = json.loads(reader.scripts[0])
        authored = ['$x+1$', '$$y^2$$', '$a&lt;b$']
        self.assertEqual([row['tex'] for row in parsed], authored)
        self.assertEqual([row['id'] for row in parsed], ['000001', '000002', '000003'])
        self.assertEqual([row['sha256'] for row in parsed],
                         [hashlib.sha256(x.encode('utf-8')).hexdigest() for x in authored])
        self.assertEqual([x['data-pdf-math-id'] for x in reader.spans],
                         ['000001', '000002', '000003'])
        self.assertIn('<code>$inline code$</code>', labelled)
        self.assertIn('title="$attribute$"', labelled)
        self.assertEqual(pdf_math_sources(labelled), parsed)

    def test_json_script_cannot_be_closed_by_tex_literal_less_than(self):
        html = '<html><body><p>$x < 2$</p></body></html>'
        labelled = label_pdf_math_sources(html)
        reader = SourceHTML()
        reader.feed(labelled)
        self.assertEqual(json.loads(reader.scripts[0]), [{
            'id': '000001', 'tex': '$x < 2$',
            'sha256': hashlib.sha256(b'$x < 2$').hexdigest()}])
        self.assertNotIn('<', reader.scripts[0])
        self.assertIn(r'$x \lt  2$', labelled)

    def test_missing_duplicate_out_of_order_and_wrong_sha_sources_reject(self):
        with self.assertRaises(SystemExit):
            pdf_math_sources('<html><body></body></html>')
        for payload in ([SOURCE, SOURCE], [{**SOURCE, 'id': '000002'}],
                        [{**SOURCE, 'sha256': '0' * 64}]):
            with self.subTest(payload=payload), self.assertRaises(SystemExit):
                pdf_math_sources(source_page(payload))


class PDFMathPaintTests(unittest.TestCase):
    def test_painted_formula_roundtrip_preserves_parenttree_mcid_and_geometry(self):
        writer, page, tree, figure, ref, pt = formula_fixture()
        original_parent = figure.raw_get('/P')
        original_pg = figure.raw_get('/Pg')
        old_pt = str(pt)
        before = ContentStream(page.get_contents(), writer).operations
        self.assertEqual(tag_pdf_formulas(writer, [SOURCE]), 1)
        self.assertEqual(figure['/S'], '/Formula')
        self.assertEqual(figure['/T'], 'pdf-math-000001')
        self.assertNotIn('/ID', figure)
        self.assertEqual(figure['/K'], 0)
        self.assertEqual(figure.raw_get('/P'), original_parent)
        self.assertEqual(figure.raw_get('/Pg'), original_pg)
        self.assertEqual(str(pt), old_pt)
        self.assertEqual(pt['/Nums'][1][0], ref)
        after = ContentStream(page.get_contents(), writer).operations
        self.assertEqual([(a, op) for a, op in before if op != b'BDC'],
                         [(a, op) for a, op in after if op != b'BDC'])
        marking = [(a, op) for a, op in after if op == b'BDC']
        self.assertEqual(len(marking), 1)
        self.assertEqual(marking[0][0][0], '/Formula')
        self.assertEqual(marking[0][0][1]['/MCID'], 0)
        self.assertEqual(marking[0][0][1]['/ActualText'], 'x plus 1')
        output = io.BytesIO()
        writer.write(output)
        reader = PdfReader(output)
        item = reader.trailer['/Root']['/StructTreeRoot']['/K']['/K'][0].get_object()
        self.assertEqual(item['/S'], '/Formula')
        self.assertEqual(item['/Alt'], 'x plus 1')
        self.assertEqual(item['/ActualText'], 'x plus 1')
        self.assertEqual(item['/Lang'], 'en')
        self.assertEqual(item['/K'], 0)
        tree_read = reader.trailer['/Root']['/StructTreeRoot']
        self.assertEqual(tree_read['/ParentTree']['/Nums'][1][0].get_object(), item)

    def test_unlabelled_figure_is_not_relabelled_or_used_as_missing_formula(self):
        writer, _, _, figure, _, _ = formula_fixture(alt='MASP-MATH-like:ordinary diagram')
        self.assertEqual(tag_pdf_formulas(writer, []), 0)
        self.assertEqual(figure['/S'], '/Figure')
        writer, *_ = formula_fixture(alt='ordinary figure')
        with self.assertRaises(SystemExit):
            tag_pdf_formulas(writer, [SOURCE])

    def test_bad_marker_role_sha_empty_speech_and_unknown_source_reject(self):
        for options in ({'alt': 'MASP-MATH:malformed'},
                        {'role': '/P'},
                        {'alt': f'MASP-MATH:000001:{"0" * 64}:x plus 1'},
                        {'alt': f'MASP-MATH:000001:{DIGEST}:   '},
                        {'alt': f'MASP-MATH:000002:{DIGEST}:x plus 1'}):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                writer, *_ = formula_fixture(**options)
                tag_pdf_formulas(writer, [SOURCE])

    def test_no_paint_mismatched_content_mcid_duplicate_bdc_and_unbalanced_reject(self):
        for data in (b'/Figure <</MCID 0>> BDC EMC',
                     b'/Figure <</MCID 1>> BDC 10 10 20 20 re f EMC',
                     b'/Figure <</MCID 0>> BDC 10 10 20 20 re f EMC '
                     b'/Figure <</MCID 0>> BDC 10 10 20 20 re f EMC',
                     b'/Figure <</MCID 0>> BDC 10 10 20 20 re f',
                     b'EMC /Figure <</MCID 0>> BDC 10 10 20 20 re f EMC'):
            with self.subTest(data=data), self.assertRaises(SystemExit):
                writer, *_ = formula_fixture(content=data)
                tag_pdf_formulas(writer, [SOURCE])

    def test_duplicate_source_figure_is_rejected(self):
        writer, _, tree, figure, _, _ = formula_fixture()
        second = DictionaryObject(figure)
        tree['/K']['/K'].append(writer._add_object(second))
        with self.assertRaises(SystemExit):
            tag_pdf_formulas(writer, [SOURCE])

    def test_parenttree_must_refer_to_the_formula_own_mcid(self):
        writer, _, _, _, _, parent_tree = formula_fixture()
        wrong = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/StructElem'),
                                                    NameObject('/S'): NameObject('/P')}))
        parent_tree['/Nums'][1][0] = wrong
        with self.assertRaises(SystemExit):
            tag_pdf_formulas(writer, [SOURCE])

    def test_formula_page_must_belong_to_writer_pages(self):
        writer, _, _, figure, _, _ = formula_fixture()
        detached = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Page')}))
        figure[NameObject('/Pg')] = detached
        with self.assertRaises(SystemExit):
            tag_pdf_formulas(writer, [SOURCE])

    def test_parenttree_kids_and_nested_marked_content_keep_real_association(self):
        writer, page, tree, figure, ref, parent_tree = formula_fixture(
            content=b'/Figure <</MCID 0>> BDC /Span BMC 10 10 20 20 re f EMC EMC')
        leaf = writer._add_object(DictionaryObject(parent_tree))
        parent_tree.clear()
        parent_tree[NameObject('/Kids')] = ArrayObject([leaf])
        old_nums = str(leaf.get_object()['/Nums'])
        self.assertEqual(tag_pdf_formulas(writer, [SOURCE]), 1)
        self.assertEqual(figure['/K'], 0)
        self.assertEqual(leaf.get_object()['/Nums'][1][0], ref)
        self.assertEqual(str(leaf.get_object()['/Nums']), old_nums)

    def test_wrong_page_parenttree_key_and_out_of_range_mcid_reject(self):
        for kind in ('missing_page_key', 'different_page_key', 'negative_mcid', 'large_mcid'):
            with self.subTest(kind=kind), self.assertRaises(SystemExit):
                writer, page, tree, figure, ref, parent_tree = formula_fixture()
                if kind == 'missing_page_key':
                    del page['/StructParents']
                elif kind == 'different_page_key':
                    page[NameObject('/StructParents')] = NumberObject(7)
                elif kind == 'negative_mcid':
                    figure[NameObject('/K')] = NumberObject(-1)
                else:
                    figure[NameObject('/K')] = NumberObject(8)
                tag_pdf_formulas(writer, [SOURCE])


if __name__ == '__main__':
    unittest.main()
