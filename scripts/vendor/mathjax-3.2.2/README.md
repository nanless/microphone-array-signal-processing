# MathJax 3.2.2 与 SRE 4.0.6：合订 PDF 的离线公式资源

本目录供 `scripts/build_pdf.py` 打印合订本使用。当前公式由MathJax3.2.2生成SVG字形，并用Speech Rule Engine（SRE）4.0.6的英文规则描述完整MathML结构。网页站点仍由自身构建器加载在线公式资源。这里固定版本用于复现，不声称它们是最新版本。

## 官方来源与许可

MathJax来自[官方npm归档](https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz)，归档SHA-256为`1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09`；其Apache-2.0许可原文保存在[LICENSE](LICENSE)。资源保留原字节，仅去掉上游路径的`es5/`前缀。

英文规则来自[SRE4.0.6官方npm归档](https://registry.npmjs.org/speech-rule-engine/-/speech-rule-engine-4.0.6.tgz)，归档SHA-256为`a8ca04e68178cd934ac57866c30af78c246fcba448ef590071398ac6486e0846`。MathJax的`a11y/sre.js`内嵌SRE4.0.6；所配规则保持同一版本。SRE的Apache-2.0原文另保存在[SRE-LICENSE](SRE-LICENSE)，与[固定官方提交的许可](https://github.com/Speech-Rule-Engine/speech-rule-engine/blob/ba19057d31620aa1f7985893ad5cff090bc70e6d/LICENSE)字节一致。未引入训练模型或新增浏览器外部依赖。

| 本地文件 | 用途 | SHA-256 |
|---|---|---|
| `tex-svg.js` | TeX解析与SVG字形路径 | `d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882` |
| `input/tex/extensions/boldsymbol.js` | 本书的粗体数学扩展 | `d6771fee0772db2657796c8d0e20e1878bb3237f6d3ed1e828e1834a4ff743ca` |
| `a11y/sre.js` | 内嵌SRE4.0.6浏览器接口 | `a41dc487037b75d6dae75677e36fe2f03d1d2684f232d96f0422a11b35051660` |
| `sre/mathmaps/base.json` | 基础规则 | `83311887b069476a6a4d0c6afe7f8e55075c85b44b029588739c0d835fc912b8` |
| `sre/mathmaps/en.json` | 英文规则 | `19d9b309dc0d25d2bde2d3130bd3772b6d7c3f8316ebbd85306bef55ac89b2a7` |
| `LICENSE` | MathJax许可 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `SRE-LICENSE` | SRE许可 | `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594` |
| `output/chtml/fonts/woff-v2/MathJax_Math-Regular.woff` | SVG排版单位度量，不绘制SVG路径字形 | `c01d3321e89b403c4b811aa153c4e618eda3421f92d8a072a02c8d190782a191` |

## 构建核验与辅助文本边界

构建命令为仓库根目录的 `.venv/bin/python scripts/build_pdf.py --build-date 2026-10-05`；详细流程见[构建手册](../../README.md)。`check_mathjax_assets`核SVG主脚本、粗体扩展、SRE接口/规则/许可和单位度量字体的完整摘要；MathJax许可另核存在。缺文件或摘要不符会拒绝构建。浏览器实际读取度量字体及base/en时再次核完整字节摘要，并用固定分数说明检查规则初始化；只等ready或只查非空不能发现缺基础规则后的退化。

SVG路径的尺寸以TeX的`ex`为单位，其固定x-height为`0.442em`。直接继承中文字体会把公式额外放大并使整书打印缩小，因此仅对公式的预度量和SVG容器使用原`MathJax_Math-Regular.woff`；浏览器先实际加载字体再排版，禁用合成粗斜体。正文仍为16px，公式保留所在段落或表格的字号；中文SVG文字使用正文中文字体。该WOFF用于单位和宽度换算，数学可见字形仍由SVG路径绘制，不能把外框强缩到页面宽度当作正确排版。

每个公式出现绑定精确TeX摘要；浏览器从实际MathJax数学树导出完整MathML，再生成非空英文说明，只给真实SVG元素赋辅助标签。打印后必须核对每个源身份与唯一真实MCID、可见绘制内容，再保留父树和字形，把对应结构改为Formula，并在结构元素及同一内容标记写入ActualText。缺式、重复、未知宏、空说明或没有绘制均不能验收；不会把公式当装饰以绕过检查。

SRE4.0.6没有中文规则，英文说明标记为`en`。星号、横线、H上标只按可见结构描述，共轭等含义由正文定义。源公式中文文字标签仍可能出现在英文结构说明中，混合语言发音和正文符号定义须由人工另核。机器描述不证明完整阅读顺序、数学含义或辅助技术实际体验正确；PDF/UA机器检查与人工辅助技术检查仍须分别留证。

构建器仅对经嵌入字体、ToUnicode和零轮廓共同证明的未标记排版空格加Layout标记，保留原文字推进；不会将可见公式路径或未知内容改为装饰。该有限处理及正文存在性检查见构建手册，不构成PDF/UA合规声明。

## 保留的历史资源

旧`tex-mml-chtml.js`及23个`output/chtml/fonts/woff-v2/MathJax_*.woff`保留原字节，用于历史构建复现。旧主脚本SHA-256为`300480069078b5892d2363a2b65e2dfbbf30fe5c80f83edbfecf4610fd093862`。其中一个原字体现在也承担上述SVG单位度量，另外22个保留用于历史复现。兼容资源仍核23个字体数量和非空性；当前度量字体另核完整摘要，不能泛称其余字体均已逐字节核验。

PDF源摘要纳入本目录全部文件和公式辅助脚本，以识别发布输入变化；源摘要本身不能证明第三方来源或合规。版本和资源选集变化时，须独立核官方归档、许可与实际渲染，不手改压缩脚本、规则或字体。
