/* PDF-only source-bound SVG mathematics. Fixed MathJax 3.2.2 / SRE 4.0.6. */
(function () {
  "use strict";
  async function loadMetricsFont() {
    await verifyAssetBytes({
      "output/chtml/fonts/woff-v2/MathJax_Math-Regular.woff":
        "c01d3321e89b403c4b811aa153c4e618eda3421f92d8a072a02c8d190782a191"
    });
    const loaded = await document.fonts.load('16px "MASP TeX Metrics"', "x");
    if (loaded.length !== 1 || loaded[0].status !== "loaded"
        || !document.fonts.check('16px "MASP TeX Metrics"', "x"))
      throw Error("Missing fixed SVG unit metrics font");
  }
  async function verifyAssetBytes(hashes) {
    for (const [name, expected] of Object.entries(hashes)) {
      const url = new URL("../scripts/vendor/mathjax-3.2.2/" + name, location.href);
      const bytes = await new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("GET", url.href); request.responseType = "arraybuffer";
        request.onload = () => {
          if ((request.status !== 200 && !(url.protocol === "file:" && request.status === 0))
              || !request.response || !request.response.byteLength)
            reject(Error("Missing fixed mathematical resource: " + name));
          else resolve(request.response);
        };
        request.onerror = () => reject(Error("Cannot read fixed mathematical resource: " + name));
        request.send();
      });
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
        .map(value => value.toString(16).padStart(2, "0")).join("");
      if (digest !== expected) throw Error("Mismatched fixed mathematical resource: " + name);
    }
  }
  async function prepare() {
    if (MathJax.version !== "3.2.2" || MathJax.startup.output.name !== "SVG")
      throw Error("Unexpected MathJax version or output");
    const sources = JSON.parse(document.getElementById("pdf-math-sources").textContent);
    const expected = new Map(sources.map(source => [source.id, source]));
    if (expected.size !== sources.length) throw Error("Duplicate math source identity");
    await verifyAssetBytes({
      "sre/mathmaps/base.json": "83311887b069476a6a4d0c6afe7f8e55075c85b44b029588739c0d835fc912b8",
      "sre/mathmaps/en.json": "19d9b309dc0d25d2bde2d3130bd3772b6d7c3f8316ebbd85306bef55ac89b2a7"
    });
    const wrappers = document.querySelectorAll("span[data-pdf-math-id]");
    if (wrappers.length !== sources.length) throw Error("Incomplete live mathematical sources");
    const engine = MathJax._.a11y.sre.Sre;
    await engine.setupEngine({locale: "en", domain: "mathspeak", style: "default"});
    await engine.sreReady();
    const setup = engine.engineSetup();
    if (setup.locale !== "en" || setup.domain !== "mathspeak")
      throw Error("Unexpected speech language or rule domain");
    const probe = '<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow><mi>c</mi></mfrac></math>';
    if (engine.toSpeech(probe).replace(/\s+/g, " ").trim() !== "StartFraction a plus b Over c EndFraction")
      throw Error("Mathematical speech rules did not initialize correctly");
    const staged = [], seen = new Set();
    for (const item of MathJax.startup.document.math) {
      const root = item.typesetRoot;
      const wrapper = root && root.closest("[data-pdf-math-id]");
      if (!wrapper || !root.isConnected || !wrapper.isConnected)
        throw Error("Unbound or detached mathematical content");
      const source = expected.get(wrapper.dataset.pdfMathId);
      if (!source || seen.has(source.id) || source.sha256 !== wrapper.dataset.pdfMathSha)
        throw Error("Unknown, repeated or mismatched mathematical source");
      const mml = MathJax.startup.toMML(item.root);
      if (/<merror(?:\s|>)/.test(mml)) throw Error("Unknown or invalid TeX: " + source.id);
      const rawSpeech = engine.toSpeech(mml);
      const speech = rawSpeech.replace(/\s+/g, " ").trim();
      const svgs = root.querySelectorAll(":scope > svg");
      if (!speech || svgs.length !== 1 || !svgs[0].querySelector("path,text,rect,line,polygon"))
        throw Error("Missing speech or actual SVG paint: " + source.id);
      const decoder = document.createElement("textarea");
      decoder.innerHTML = source.tex.replace(/</g, "\\lt ");
      const tex = decoder.value.replace(/^\\*(\$\$?)/, "").replace(/\$\$?$/, "");
      if (item.math !== tex) throw Error("MathJax source differs from authored TeX: " + source.id);
      staged.push({source, root, wrapper, svg: svgs[0], mml, rawSpeech, speech});
      seen.add(source.id);
    }
    if (seen.size !== expected.size) throw Error("Incomplete mathematical rendering");
    // Assign export labels only after every source has succeeded. A partially
    // initialized Chrome print is subsequently rejected by the PDF source check.
    for (const {source, root, wrapper, svg, speech} of staged) {
      root.removeAttribute("aria-hidden");
      svg.removeAttribute("aria-hidden");
      svg.setAttribute("role", "img");
      svg.setAttribute("lang", "en");
      svg.setAttribute("aria-label", "MASP-MATH:" + source.id + ":" + source.sha256 + ":" + speech);
      for (const assistive of root.querySelectorAll("mjx-assistive-mml")) assistive.remove();
      wrapper.replaceWith(...wrapper.childNodes);
    }
    window.MaspPdfMath = staged.map(({source, mml, rawSpeech, speech}) => ({...source, mml, rawSpeech, speech}));
    await document.fonts.ready;
    ArrayTutorialLayout.apply();
    document.documentElement.dataset.pdfMathReady = String(seen.size);
    return seen.size;
  }
  window.MathJax = {
    loader: {paths: {sre: "../scripts/vendor/mathjax-3.2.2/sre/mathmaps"},
      load: ["a11y/sre", "[tex]/boldsymbol"]},
    tex: {inlineMath: [["$", "$"], ["\\(", "\\)"]], displayMath: [["$$", "$$"]],
      packages: {"[+]": ["boldsymbol"], "[-]": ["noundefined"]}},
    svg: {fontCache: "none"},
    startup: {pageReady: () => loadMetricsFont().then(() => MathJax.startup.defaultPageReady()).then(prepare)}
  };
})();
