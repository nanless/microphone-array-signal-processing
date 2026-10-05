"""Independent valid/error/look-alike fixtures for bounded PDF export repairs."""

import io
import unittest

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DictionaryObject, NameObject,
                          NumberObject, TextStringObject)

from scripts.build_pdf import normalize_pdf_semantics


def elem(writer, role, parent=None, children=None):
    value = DictionaryObject({NameObject('/Type'): NameObject('/StructElem'),
                              NameObject('/S'): NameObject(role)})
    reference = writer._add_object(value)
    if parent is not None:
        value[NameObject('/P')] = parent
    if children is not None:
        value[NameObject('/K')] = ArrayObject(children)
    return reference


def fixture():
    writer = PdfWriter()
    page = writer.add_blank_page(595, 842)
    tree = DictionaryObject({NameObject('/Type'): NameObject('/StructTreeRoot')})
    tree_ref = writer._add_object(tree)
    writer.root_object[NameObject('/StructTreeRoot')] = tree_ref
    document = elem(writer, '/Document', tree_ref, [])
    tree[NameObject('/K')] = ArrayObject([document])
    writer.add_metadata({'/Title': '数学 & 信号'})
    return writer, page, tree, document


class PdfSemanticRepairTests(unittest.TestCase):
    def test_list_body_preserves_existing_children_and_marker_parent(self):
        writer, page, tree, document = fixture()
        li = elem(writer, '/LI', document, [])
        marker = elem(writer, '/Lbl', li)
        text = elem(writer, '/NonStruct', li)
        link = elem(writer, '/Link', li)
        nested = elem(writer, '/L', li)
        li.get_object()[NameObject('/K')] = ArrayObject([marker, text, link, nested])
        document.get_object()['/K'].append(li)
        normalize_pdf_semantics(writer, [])
        children = li.get_object()['/K']
        self.assertEqual(children[0], marker)
        self.assertEqual(marker.get_object().raw_get('/P'), li)
        body = children[1].get_object()
        self.assertEqual(body['/S'], '/LBody')
        self.assertEqual(body['/K'], [text, link, nested])
        self.assertEqual(body.raw_get('/P'), li)
        for original in (text, link, nested):
            self.assertEqual(original.get_object().raw_get('/P'), children[1])
        normalize_pdf_semantics(writer, [])
        self.assertEqual(li.get_object()['/K'], children)

    def test_raw_mcid_is_rejected_instead_of_guessing_parent_tree(self):
        writer, _, _, document = fixture()
        li = elem(writer, '/LI', document, [NumberObject(17)])
        document.get_object()['/K'].append(li)
        with self.assertRaisesRegex(SystemExit, 'MCID'):
            normalize_pdf_semantics(writer, [])

    def test_same_named_non_list_and_valid_lbody_are_preserved(self):
        writer, _, _, document = fixture()
        paragraph = elem(writer, '/P', document, [NumberObject(3)])
        li = elem(writer, '/LI', document, [])
        body = elem(writer, '/LBody', li, [NumberObject(9)])
        li.get_object()[NameObject('/K')] = ArrayObject([body])
        document.get_object()['/K'].extend([paragraph, li])
        normalize_pdf_semantics(writer, [])
        self.assertEqual(paragraph.get_object()['/K'], [3])
        self.assertEqual(li.get_object()['/K'], [body])
        self.assertEqual(body.get_object()['/K'], [9])

    def test_link_labels_use_real_target_and_do_not_invent_untagged_semantics(self):
        writer, page, _, _ = fixture()
        def link(target, tagged=True, contents=None):
            value = DictionaryObject({NameObject('/Subtype'): NameObject('/Link'),
                                      NameObject('/Dest'): NameObject('/' + target)})
            if tagged:
                value[NameObject('/StructParent')] = NumberObject(4)
            if contents is not None:
                value[NameObject('/Contents')] = TextStringObject(contents)
            return writer._add_object(value)
        current, old, exercise, helper, existing, old_exercise = (
            link('ch-14-sec-12-1'), link('ch-14-sec-14-1'),
            link('ch-14-E12-17'), link('ch-0', False), link('ch-14', True, '原说明'),
            link('ch-14-e14-17'))
        external = writer._add_object(DictionaryObject({
            NameObject('/Subtype'): NameObject('/Link'),
            NameObject('/StructParent'): NumberObject(5),
            NameObject('/A'): DictionaryObject({NameObject('/S'): NameObject('/URI'),
                NameObject('/URI'): TextStringObject('https://example.org/math?q=1&n=2')}),
        }))
        page[NameObject('/Annots')] = ArrayObject(
            [current, old, exercise, helper, existing, old_exercise, external])
        for target in ('ch-14-sec-12-1', 'ch-14-sec-14-1', 'ch-14-E12-17', 'ch-14-e14-17'):
            writer.add_named_destination(target, 0)
        outline = [('第12章', 'ch-14', [('12.1 声压量', 'ch-14-sec-12-1', [])])]
        normalize_pdf_semantics(writer, outline, {
            'ch-14-sec-14-1': 'ch-14-sec-12-1', 'ch-14-e14-17': 'ch-14-e12-17'})
        self.assertEqual(current.get_object()['/Contents'], '跳转到：12.1 声压量')
        self.assertEqual(old.get_object()['/Contents'], current.get_object()['/Contents'])
        self.assertEqual(exercise.get_object()['/Contents'], '查看练习：E12-17')
        self.assertNotIn('/Contents', helper.get_object())
        self.assertEqual(existing.get_object()['/Contents'], '原说明')
        self.assertEqual(old_exercise.get_object()['/Contents'], '查看练习：E12-17')
        self.assertEqual(external.get_object()['/Contents'],
                         '打开外部链接：https://example.org/math?q=1&n=2')

    def test_chrome_catalog_name_keys_match_real_annotation_targets(self):
        writer, page, _, _ = fixture()
        destination = ArrayObject([page.indirect_reference, NameObject('/Fit')])
        writer.root_object[NameObject('/Dests')] = DictionaryObject({
            NameObject('/ch-0'): destination})
        annotation = DictionaryObject({NameObject('/Subtype'): NameObject('/Link'),
                                      NameObject('/StructParent'): NumberObject(4),
                                      NameObject('/Dest'): NameObject('/ch-0')})
        page[NameObject('/Annots')] = ArrayObject([writer._add_object(annotation)])
        normalize_pdf_semantics(writer, [('导读', 'ch-0', [])])
        self.assertEqual(annotation['/Contents'], '跳转到：导读')
        self.assertEqual(annotation['/Dest'], '/ch-0')
        self.assertEqual(set(writer.named_destinations), {'/ch-0'})
        del annotation['/Contents']
        annotation[NameObject('/Dest')] = NameObject('/ch-00')
        with self.assertRaisesRegex(SystemExit, '目标不存在'):
            normalize_pdf_semantics(writer, [('导读', 'ch-0', [])])

    def test_missing_target_and_conflicting_role_map_fail(self):
        writer, page, tree, _ = fixture()
        tree[NameObject('/RoleMap')] = DictionaryObject({NameObject('/Strong'): NameObject('/Figure')})
        with self.assertRaisesRegex(SystemExit, '角色映射冲突'):
            normalize_pdf_semantics(writer, [])
        del tree['/RoleMap']
        page[NameObject('/Annots')] = ArrayObject([writer._add_object(DictionaryObject({
            NameObject('/Subtype'): NameObject('/Link'),
            NameObject('/StructParent'): NumberObject(4),
            NameObject('/Dest'): NameObject('/unknown'),
        }))])
        with self.assertRaisesRegex(SystemExit, '目标不存在'):
            normalize_pdf_semantics(writer, [])

    def test_fake_exercise_target_cannot_invent_a_real_destination(self):
        writer, page, _, _ = fixture()
        page[NameObject('/Annots')] = ArrayObject([writer._add_object(DictionaryObject({
            NameObject('/Subtype'): NameObject('/Link'),
            NameObject('/StructParent'): NumberObject(4),
            NameObject('/Dest'): NameObject('/nonexistent-e99-99'),
        }))])
        with self.assertRaisesRegex(SystemExit, '目标不存在'):
            normalize_pdf_semantics(writer, [])

    def test_roundtrip_xmp_is_valid_unicode_metadata_without_ua_claim(self):
        writer, page, tree, _ = fixture()
        normalize_pdf_semantics(writer, [])
        self.assertEqual(page['/Tabs'], '/S')
        self.assertEqual(tree['/RoleMap'], {'/Strong': '/Span', '/Em': '/Span'})
        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        reader = PdfReader(output)
        self.assertEqual(reader.pdf_header, '%PDF-1.7')
        metadata = reader.trailer['/Root']['/Metadata']
        self.assertEqual(metadata['/Type'], '/Metadata')
        self.assertEqual(metadata['/Subtype'], '/XML')
        self.assertNotIn(b'pdfuaid', metadata.get_data())
        self.assertEqual(reader.xmp_metadata.dc_title, {'x-default': '数学 & 信号'})
        self.assertEqual(reader.xmp_metadata.dc_language, ['zh-CN'])


if __name__ == '__main__':
    unittest.main()
