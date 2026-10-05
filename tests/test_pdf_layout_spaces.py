"""Real font/content fixtures for the limited PDF layout-space contract.

The fixtures originate as raw PDF bytes and real TrueType glyph programs. They
check semantic exclusion and unchanged text advance, not PDF/UA compliance or
the visual correctness of an entire book.
"""

import hashlib
import io
import unittest

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, ContentStream, DecodedStreamObject,
                          DictionaryObject, NameObject, NumberObject,
                          TextStringObject)

from scripts.build_pdf import (keep_imaging_hand_example, label_pdf_math_sources,
                               pdf_math_sources, tag_pdf_formulas,
                               verified_layout_space_font)


def true_type_bytes(*, visible_space=False, space_advance=600):
    """Glyph index 3 is a real blank or a deliberately visible rectangle."""
    order = ['.notdef', 'one', 'two', 'space', 'letter']
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({32: 'space', 65: 'letter'})
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(None)
        if name == 'letter' or (name == 'space' and visible_space):
            pen.moveTo((50, 0))
            pen.lineTo((550, 0))
            pen.lineTo((550, 700))
            pen.lineTo((50, 700))
            pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (
        space_advance if name == 'space' else 700, 0) for name in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({'familyName': 'Independent Fixture',
                            'styleName': 'Regular',
                            'uniqueFontIdentifier': 'IndependentFixture-Regular',
                            'fullName': 'Independent Fixture Regular',
                            'psName': 'IndependentFixture-Regular'})
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200,
                     usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    builder.setupMaxp()
    buffer = io.BytesIO()
    builder.save(buffer)
    return buffer.getvalue()


def unicode_map(mapping=b'<0003> <0020>\n<0004> <0041>', *, count=2):
    return (b'/CIDInit /ProcSet findresource begin\n12 dict begin\n'
            b'begincmap\n/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) '
            b'/Supplement 0 >> def\n/CMapName /Fixture-UCS def\n/CMapType 2 def\n'
            b'1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n'
            + str(count).encode('ascii') + b' beginbfchar\n' + mapping
            + b'\nendbfchar\nendcmap\nCMapName currentdict /CMap defineresource '
            b'pop\nend\nend\n')


def font_resource(writer, *, visible_space=False, space_advance=600,
                  cmap=None, embedded=None, cid_map='/Identity'):
    def stream(data):
        value = DecodedStreamObject()
        value.set_data(data)
        return writer._add_object(value)

    descriptor = DictionaryObject({NameObject('/Type'): NameObject('/FontDescriptor'),
                                   NameObject('/FontName'): NameObject('/Fixture'),
                                   NameObject('/FontFile2'): stream(
                                       true_type_bytes(visible_space=visible_space,
                                                       space_advance=space_advance)
                                       if embedded is None else embedded)})
    descendant = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                   NameObject('/Subtype'): NameObject('/CIDFontType2'),
                                   NameObject('/BaseFont'): NameObject('/Fixture'),
                                   NameObject('/CIDToGIDMap'): NameObject(cid_map),
                                   NameObject('/FontDescriptor'): writer._add_object(descriptor),
                                   NameObject('/DW'): NumberObject(700),
                                   NameObject('/W'): ArrayObject([
                                       NumberObject(3), ArrayObject([
                                           NumberObject(space_advance), NumberObject(700)])])})
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type0'),
                             NameObject('/BaseFont'): NameObject('/Fixture'),
                             NameObject('/Encoding'): NameObject('/Identity-H'),
                             NameObject('/ToUnicode'): stream(
                                 unicode_map() if cmap is None else cmap),
                             NameObject('/DescendantFonts'): ArrayObject([
                                 writer._add_object(descendant)])})
    return writer._add_object(font)


def page_fixture(content, **font_options):
    writer = PdfWriter()
    page = writer.add_blank_page(200, 200)
    tree = DictionaryObject({NameObject('/Type'): NameObject('/StructTreeRoot'),
                             NameObject('/K'): ArrayObject([])})
    tree_ref = writer._add_object(tree)
    writer.root_object[NameObject('/StructTreeRoot')] = tree_ref
    font = font_resource(writer, **font_options)
    page[NameObject('/Resources')] = DictionaryObject({
        NameObject('/Font'): DictionaryObject({NameObject('/Fblank'): font})})
    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject('/Contents')] = writer._add_object(stream)
    return writer, page, tree, font


def operations(page, pdf):
    return ContentStream(page.get_contents(), pdf).operations


def plain_operations(ops):
    """Compare actual operands, including original parsed text bytes."""
    def value(item):
        if hasattr(item, '_original_bytes'):
            return ('PDF bytes', item._original_bytes)
        if isinstance(item, list):
            return tuple(value(x) for x in item)
        return item
    return [(tuple(value(x) for x in args), op) for args, op in ops
            if op not in (b'BDC', b'BMC', b'EMC')]


def artifact_texts(ops):
    stack, result = [], []
    for args, op in ops:
        if op in (b'BDC', b'BMC'):
            stack.append(args[0])
        elif op == b'EMC':
            stack.pop()
        elif op == b'Tj' and '/Artifact' in stack:
            result.append(args[0]._original_bytes)
    if stack:
        raise AssertionError('unclosed marked content')
    return result


def roundtrip(writer):
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return PdfReader(buffer)


SPACE_STREAM = b'BT /Fblank 12 Tf 1 0 0 1 10 20 Tm <0003> Tj ET\n'


class LayoutSpaceFontTests(unittest.TestCase):
    def test_real_font_glyph3_is_blank_positive_advance_and_raw_string_is_parsed(self):
        writer, page, _, font = page_fixture(SPACE_STREAM)
        with TTFont(io.BytesIO(font.get_object()['/DescendantFonts'][0]
                              ['/FontDescriptor']['/FontFile2'].get_data())) as parsed:
            name = parsed.getGlyphOrder()[3]
            self.assertEqual(parsed['glyf'][name].numberOfContours, 0)
            self.assertEqual(parsed['hmtx'][name][0], 600)
        text = [args[0] for args, op in operations(page, writer) if op == b'Tj']
        self.assertEqual(text[0]._original_bytes, b'\x00\x03')
        self.assertTrue(verified_layout_space_font(font))

    def test_visible_space_outline_remains_semantic_content(self):
        writer, page, _, font = page_fixture(SPACE_STREAM, visible_space=True)
        self.assertFalse(verified_layout_space_font(font))
        before = plain_operations(operations(page, writer))
        self.assertEqual(tag_pdf_formulas(writer, []), 0)
        self.assertEqual(artifact_texts(operations(page, writer)), [])
        self.assertEqual(plain_operations(operations(page, writer)), before)

    def test_wrong_unicode_duplicate_mapping_and_range_do_not_prove_blank_space(self):
        maps = [unicode_map(b'<0003> <0041>', count=1),
                unicode_map(b'<0003> <0020>\n<0003> <0020>', count=2),
                unicode_map() + b'1 beginbfrange\n<0003> <0004> <0020>\nendbfrange']
        for cmap in maps:
            with self.subTest(cmap=cmap):
                writer, page, _, font = page_fixture(SPACE_STREAM, cmap=cmap)
                self.assertFalse(verified_layout_space_font(font))
                tag_pdf_formulas(writer, [])
                self.assertEqual(artifact_texts(operations(page, writer)), [])

    def test_nonidentity_cid_invalid_font_and_zero_advance_are_not_decorations(self):
        for options in ({'cid_map': '/Other'}, {'embedded': b'not a font'},
                        {'space_advance': 0}):
            with self.subTest(options=options):
                writer, page, _, font = page_fixture(SPACE_STREAM, **options)
                self.assertFalse(verified_layout_space_font(font))
                tag_pdf_formulas(writer, [])
                self.assertEqual(artifact_texts(operations(page, writer)), [])

    def test_disjoint_chrome_ranges_preserve_the_unique_blank_space_mapping(self):
        cmap = unicode_map() + (b'2 beginbfrange\n<0008> <0009> <0025>\n'
                                b'<0024> <003C> <0041>\nendbfrange')
        writer, page, _, font = page_fixture(SPACE_STREAM, cmap=cmap)
        self.assertTrue(verified_layout_space_font(font))
        before = plain_operations(operations(page, writer))
        tag_pdf_formulas(writer, [])
        self.assertEqual(len(artifact_texts(operations(page, writer))), 1)
        self.assertEqual(plain_operations(operations(page, writer)), before)

    def test_unsupported_or_ambiguous_ranges_do_not_prove_blank_space(self):
        ranges = [b'1 beginbfrange\n<0002> <0004> <001F>\nendbfrange',
                  b'2 beginbfrange\n<0008> <0009> <0025>\nendbfrange',
                  b'1 beginbfrange\n<0009> <0008> <0025>\nendbfrange',
                  b'1 beginbfrange\n<0008> <0009> [<0025> <0026>]\nendbfrange',
                  b'1 beginbfrange\n<0008> <0009> <0025> trailing\nendbfrange',
                  b'beginbfrange\n<0008> <0009> <0025>\nendbfrange']
        for block in ranges:
            with self.subTest(block=block):
                writer, page, _, font = page_fixture(SPACE_STREAM, cmap=unicode_map() + block)
                self.assertFalse(verified_layout_space_font(font))
                tag_pdf_formulas(writer, [])
                self.assertEqual(artifact_texts(operations(page, writer)), [])


class LayoutSpaceContentTests(unittest.TestCase):
    def test_only_unmarked_proven_space_gets_layout_artifact_with_original_tj(self):
        writer, page, _, _ = page_fixture(SPACE_STREAM)
        before = plain_operations(operations(page, writer))
        self.assertEqual(tag_pdf_formulas(writer, []), 0)
        after = operations(page, writer)
        self.assertEqual(plain_operations(after), before)
        self.assertEqual(artifact_texts(after), [b'\x00\x03'])
        marks = [args for args, op in after if op == b'BDC']
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0], [NameObject('/Artifact'), DictionaryObject({
            NameObject('/Type'): NameObject('/Layout')})])
        self.assertNotIn('/MCID', marks[0][1])

    def test_already_semantically_marked_or_artifact_space_is_not_wrapped_again(self):
        for marking in (b'/Span BMC', b'/Span <</MCID 7>> BDC',
                        b'/Artifact <</Type /Layout>> BDC'):
            with self.subTest(marking=marking):
                writer, page, _, _ = page_fixture(
                    marking + b'\n' + SPACE_STREAM + b'EMC\n')
                before = operations(page, writer)
                tag_pdf_formulas(writer, [])
                self.assertEqual(operations(page, writer), before)
                self.assertEqual(sum(op in (b'BMC', b'BDC') for _, op in before), 1)

    def test_real_letter_unknown_font_tjarray_and_path_paint_are_not_artifacts(self):
        content = (b'BT /Fblank 12 Tf <0004> Tj [<0003> -30 <0004>] TJ '
                   b'/Funknown 12 Tf <0003> Tj ET\n'
                   b'0 0 0 rg 10 10 20 20 re f\n')
        writer, page, _, _ = page_fixture(content)
        before = plain_operations(operations(page, writer))
        tag_pdf_formulas(writer, [])
        self.assertEqual(artifact_texts(operations(page, writer)), [])
        self.assertEqual(plain_operations(operations(page, writer)), before)

    def test_q_and_Q_restore_the_actual_font_before_proving_next_space(self):
        content = (b'BT /Fblank 12 Tf <0003> Tj q '
                   b'/Fvisible 12 Tf <0003> Tj Q <0003> Tj ET\n')
        writer, page, _, _ = page_fixture(content)
        page['/Resources']['/Font'][NameObject('/Fvisible')] = font_resource(
            writer, visible_space=True)
        tag_pdf_formulas(writer, [])
        self.assertEqual(artifact_texts(operations(page, writer)),
                         [b'\x00\x03', b'\x00\x03'])
        self.assertEqual(sum(op == b'Tj' for _, op in operations(page, writer)), 3)

    def test_graphics_state_underflow_and_unclosed_save_are_rejected(self):
        for content in (b'Q\n' + SPACE_STREAM, b'q\n' + SPACE_STREAM):
            with self.subTest(content=content), self.assertRaises(SystemExit):
                writer, *_ = page_fixture(content)
                tag_pdf_formulas(writer, [])

    def test_roundtrip_retains_geometry_original_codes_and_text_advance(self):
        content = (b'q 1 0 0 1 2 3 cm BT /Fblank 12 Tf 1.5 Tc '
                   b'1 0 0 1 10 20 Tm <0003> Tj 4 5 Td <0004> Tj ET Q '
                   b'0 0 0 rg 10 10 20 20 re f\n')
        writer, page, _, _ = page_fixture(content)
        original = plain_operations(operations(page, writer))
        # PDF text advance: width/1000 * font size + character spacing.
        expected_advances = [600 / 1000 * 12 + 1.5, 700 / 1000 * 12 + 1.5]
        tag_pdf_formulas(writer, [])
        reader = roundtrip(writer)
        after = operations(reader.pages[0], reader)
        self.assertEqual(plain_operations(after), original)
        self.assertEqual(artifact_texts(after), [b'\x00\x03'])
        font = reader.pages[0]['/Resources']['/Font']['/Fblank']
        widths = font['/DescendantFonts'][0]['/W'][1]
        actual_advances = [float(widths[int(args[0]._original_bytes.hex(), 16)-3])
                           / 1000 * 12 + 1.5 for args, op in after if op == b'Tj']
        self.assertEqual(actual_advances, expected_advances)

    def test_formula_parent_identity_and_real_paint_survive_space_artifact_roundtrip(self):
        tex = '$x$'
        digest = hashlib.sha256(tex.encode()).hexdigest()
        source = {'id': '000001', 'tex': tex, 'sha256': digest}
        writer, page, tree, _ = page_fixture(
            SPACE_STREAM + b'/Figure <</MCID 0>> BDC 10 10 20 20 re f EMC\n')
        page[NameObject('/StructParents')] = NumberObject(0)
        figure = DictionaryObject({NameObject('/Type'): NameObject('/StructElem'),
                                   NameObject('/S'): NameObject('/Figure'),
                                   NameObject('/P'): tree.indirect_reference,
                                   NameObject('/Pg'): page.indirect_reference,
                                   NameObject('/K'): NumberObject(0),
                                   NameObject('/Alt'): TextStringObject(
                                       f'MASP-MATH:000001:{digest}:x')})
        ref = writer._add_object(figure)
        tree[NameObject('/K')] = ArrayObject([ref])
        parent = DictionaryObject({NameObject('/Nums'): ArrayObject([
            NumberObject(0), ArrayObject([ref])])})
        tree[NameObject('/ParentTree')] = writer._add_object(parent)
        before = plain_operations(operations(page, writer))
        self.assertEqual(tag_pdf_formulas(writer, [source]), 1)
        self.assertEqual(parent['/Nums'][1][0], ref)
        reader = roundtrip(writer)
        actual_tree = reader.trailer['/Root']['/StructTreeRoot']
        actual = actual_tree['/K'][0].get_object()
        self.assertEqual(actual_tree['/ParentTree']['/Nums'][1][0].get_object()
                         .indirect_reference, actual.indirect_reference)
        self.assertEqual(actual['/Pg'].indirect_reference, reader.pages[0].indirect_reference)
        self.assertEqual(actual['/T'], 'pdf-math-000001')
        self.assertEqual(plain_operations(operations(reader.pages[0], reader)), before)
        self.assertEqual(artifact_texts(operations(reader.pages[0], reader)), [b'\x00\x03'])


HAND = ('<h3 id="ch-14-sec-u-3179856767">两个格点的完整手算</h3>'
        '<p>输入：</p><p>$$a=1\\tag{12-10}$$</p>'
        '<p>结果：</p><p>$$b=2\\tag{12-11}$$</p><p>解释。</p>')
IMAGE = '<p><img alt="图67" src="../figures/fig67_imaging_psf.png" /></p>'


class ImagingHandLayoutTests(unittest.TestCase):
    def test_valid_group_keeps_exact_source_math_and_leaves_figure_outside(self):
        original = '<html><body><p>$z$</p>' + HAND + IMAGE + '<p>$t$</p></body></html>'
        before = pdf_math_sources(label_pdf_math_sources(original))
        grouped = keep_imaging_hand_example(original)
        self.assertEqual(grouped.count('<div class="pdf-imaging-hand-example">'), 1)
        self.assertIn(HAND + '</div>\n' + IMAGE, grouped)
        self.assertEqual(pdf_math_sources(label_pdf_math_sources(grouped)), before)

    def test_missing_group_figure_or_math_and_duplicate_contracts_reject(self):
        invalid = [IMAGE, HAND, HAND.replace('\\tag{12-10}', '') + IMAGE,
                   HAND + IMAGE + HAND + IMAGE,
                   HAND.replace('<p>输入：</p>',
                                '<h3 id="ch-14-sec-u-3179856767">两个格点的完整手算</h3>'
                                '<p>输入：</p>') + IMAGE,
                   HAND.replace('\\tag{12-11}', '\\tag{12-11} \\tag{12-11}') + IMAGE]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                keep_imaging_hand_example('<html><body>' + content + '</body></html>')

    def test_escaped_code_and_nearby_lookalikes_are_never_swallowed_into_group(self):
        code = ('<pre><code>&lt;h3 id="ch-14-sec-u-3179856767"&gt;'
                '两个格点的完整手算&lt;/h3&gt; $$literal\\tag{12-10}$$</code></pre>')
        similar = HAND.replace('3179856767', '3179856768') + IMAGE.replace('fig67_', 'fig670_')
        hidden = ['', '<!--' + HAND + IMAGE + '-->',
                  '<pre><code>' + HAND + IMAGE + '</code></pre>',
                  '<script>const literal = `' + HAND + IMAGE + '`;</script>',
                  '<style>/*' + HAND + IMAGE + '*/</style>']
        for literal in hidden:
            with self.subTest(literal=literal):
                page = ('<html><body>' + literal + code + HAND + IMAGE + similar
                        + '</body></html>')
                grouped = keep_imaging_hand_example(page)
                self.assertIn(literal + code, grouped)
                self.assertIn(similar, grouped)
                self.assertEqual(grouped.count('<div class="pdf-imaging-hand-example">'), 1)
                self.assertEqual(pdf_math_sources(label_pdf_math_sources(page)),
                                 pdf_math_sources(label_pdf_math_sources(grouped)))


if __name__ == '__main__':
    unittest.main()
