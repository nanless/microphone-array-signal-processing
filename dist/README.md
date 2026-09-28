# 合订本与历史 PDF

`microphone-array-tutorial.pdf` 是当前发布的合订本，由仓库根目录的 `scripts/build_pdf.py` 从 `chapters/` 和 `figures/` 生成。`combined.html` 是同一构建的中间产物，受 Git 忽略；修改正文、图片或构建器后应重新生成并运行发布检查，而不是直接编辑 PDF。

另外两份 PDF 是历史审查快照，不是当前教程版本：

| 文件 | 用途 |
|---|---|
| `microphone-array-tutorial-before-reorder-2026-09-23.pdf` | 2026-09-23 章节重排前的对照快照 |
| `microphone-array-tutorial-round4-preview.pdf` | 第四轮审查的排版预览，见[当时记录](../codes/chapters/ch00/reviews/ROUND4_REVIEW.md) |

历史快照保留当时的章节、页码与内容；不能用来核对当前公式、练习或图号。当前发布物的页数、书签与图片完整性由 `scripts/quality_check.py` 检查，PDF 的公式可读性和辅助技术阅读顺序仍需人工抽查。
