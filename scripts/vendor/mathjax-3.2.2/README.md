# MathJax 3.2.2：合订 PDF 的本地公式资源

这些文件来自 MathJax 官方 npm 包 `mathjax@3.2.2`，只供
`scripts/build_pdf.py` 在本地打印合订 PDF 使用。上游项目：
<https://github.com/mathjax/MathJax>；许可为随附的 Apache-2.0 `LICENSE`。

来源归档：`https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz`，
SHA-256 `1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09`。
本目录保留上游 `es5/tex-mml-chtml.js`、其按需加载的
`es5/input/tex/extensions/boldsymbol.js` 和 `es5/output/chtml/fonts/woff-v2/`
下的 23 个 WOFF 字体；去掉了路径中的 `es5/` 前缀，文件内容未改。

构建脚本会核对主脚本、`boldsymbol` 扩展的 SHA-256、字体数量和许可文件。
本书中使用 `\boldsymbol`；如果只复制主脚本而漏掉按需扩展，MathJax
可能停止处理整本书的公式。网页站点仍由其自身的构建脚本管理公式资源，
本目录不作为站点的公开依赖。
