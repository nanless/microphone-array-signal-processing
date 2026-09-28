# An Accessible Guide to Microphone Array Signal Processing

A beginner-friendly Chinese tutorial on microphone array signal processing at graduate-entry level. It starts with “why use an array of microphones” and covers DOA estimation, beamforming, acoustic echo cancellation (AEC), dereverberation (WPE), speech separation, source tracking, engineering practice, and system selection.

The tutorial provides key derivations, reproducible numerical examples, validity limits, 49 script-generated figures, and NumPy/standard-library teaching code mapped to the equations. It includes 228 executable exercises and 109 main-manifest synthetic audio files in 27 groups; see [exercises and audio experiments](codes/chapters/ch00/research/05_exercises_and_audio.md) for inputs, answers and listening conditions. Chapter 6 exercises E06-07–20 are dimensionless model checks; two AEC audio groups use related but not identical parameters and are not substitutes for the exercise answers. These 109 files are not natural speech or formal listening-test data. A separate set of 4 WAV files contains a real 16-channel DEMAND river-noise excerpt and three derivatives under CC BY-SA 3.0, for noise-power and interchannel-correlation experiments.

Appendix B also has [18 separate synthetic room WAVs](codes/chapters/appendix_b/room_audio/MANIFEST.json) across six source positions, one chart, and a [machine-readable result report](codes/chapters/appendix_b/room_audio/RESULTS.json) from the executed simulation. They contain white noise, not recorded speech, and are separate from the 109 main synthetic files.

Two further independent synthetic sets provide [five guided-separation WAVs and intermediate state](codes/chapters/ch08/gss_audio/MANIFEST.json) and [three moving-source WAVs with trajectory truth](codes/chapters/ch09/moving_audio/MANIFEST.json). Neither set is included in the main 109 files or represents recorded speech or device performance.

Chapter 9 also has [two separately catalogued continuous-motion WAVs and frame-level observations](codes/chapters/ch09/tracking_audio/MANIFEST.json). GCC-PHAT observations, missing-data gating, and Kalman tracking are recomputed from exported PCM, with separate state and availability timestamps. These WAVs belong to neither the 109-file main collection nor the three-file moving-source set. Run `.venv/bin/python -m codes.chapters.ch09.examples.chapter09_tracking_audio --check` for a read-only check.

Chapter 6 also reports a [SpeexDSP AEC interface experiment](codes/chapters/ch00/research/02_aec_wpe_separation.md#aec) on one real Microsoft AEC Challenge loopback/microphone pair, with zero-reference and misaligned-reference controls. The crowd recordings are not redistributed here; the measured input/output power change is neither clean-component ERLE nor a device-performance ranking.

Additional reproducible studies cover a [common-input beamformer comparison](codes/chapters/ch05/beamformer_common_input_demo.py), [pinned AuxIVA blind estimation](codes/chapters/ch08/examples/reproduce_auxiva_reference.py), [GSS activity-label errors](codes/chapters/ch08/gss_activity_error_demo.py), [AEC interfaces on one synthetic known-component input](codes/chapters/ch06/examples/aec_same_input_truth.py), and [online WPE temporal behavior](codes/chapters/ch07/wpe_temporal_contract.py). Their scopes are respectively a single-frequency model, mathematical synthetic mixtures, a fixed-density E-step, synthetic PCM, and complex STFT data; they do not rank real-speech or device performance.

Controlled studies cover [co-array covariance reconstruction](codes/chapters/ch03/coarray_covariance_exercise.py), [repeated two-source resolution](codes/chapters/ch04/doa_resolution_trials.py), a [teaching cACGMM-to-MVDR chain](codes/chapters/ch08/examples/gss_teaching_demo.py), [automatic double-talk detection](codes/chapters/ch06/aec_dtd_demo.py), [continuous free-field two-microphone motion](codes/chapters/ch09/examples/moving_source_audio.py), and an [isolated SMP-PHAT portability overlay](codes/chapters/ch04/examples/reproduce_smpphat_portable_overlay.py). Each study separates its synthetic result from official implementations that were not run.

The [Chapter 9 tracking experiment](codes/chapters/ch09/tracking_crossing_dropout_demo.py) uses deterministic angle detections to show that a correct unlabelled location set can coincide with wrong track identities. It also calculates covariance growth during missing observations and lag from a beam steering speed limit; it is not a speech recording or a complete multi-target tracker.

The [Chapter 10 engineering experiments](codes/chapters/ch10/chapter10_experiments.py) provide ten reproducible exercises on overlapping-frame real-time factor, spectral-subtraction numeric limits, streaming clock state, and whole-block AGC availability. Four [common-gain synthetic WAVs](codes/chapters/ch00/research/05_exercises_and_audio.md#32-分块-agc时间常数相同输出仍可能不同) support PCM comparisons; the [industrial handbook](codes/chapters/ch00/research/03_industrial_deployment.md) distinguishes executed interface probes, static source diagnostics, and model or device tests that have not been run.

The [Chapter 11 selection experiments](codes/chapters/ch11/chapter11_experiments.py) provide ten reproducible exercises on hard constraints, scenario composition, missing scores, joint risk, and speaker-identity scoring. Four [common-gain FIR WAVs](codes/chapters/ch00/research/05_exercises_and_audio.md) are actually processed from the same constructed mixture and measured again from PCM; Figures 46 and 47 show the decision logic and audio tradeoff. They are not speech, device, or recognition-quality tests.

[Appendix A experiments](codes/chapters/appendix_a/appendix_a_experiments.py) provide seven reproducible math exercises. Three [synthetic pulse WAVs](codes/chapters/ch00/research/05_exercises_and_audio.md#sec-u-a0ab2e82f7) compare correct blockwise linear convolution with an intentionally incorrect circular wrap; Figures 48–49 show FFT frequency conventions and measured PCM sample positions.

The [source research handbook](codes/chapters/ch00/research/README.md) explains implementation steps, industrial configuration and reproduction experiments across spatial processing and tracking, AEC/WPE/separation, and deployment and evaluation. Two further guides cover source reproduction and exercises with audio experiments. Official revisions are pinned; acquired sources reside in independent working trees under `codes/chapters/ch00/upstream/_downloads/`.

中文版：[README.md](README.md)

## Layout

| Path | Description |
|---|---|
| `chapters/` | 14 tutorial documents in Markdown (`00_overview.md` is the homepage; `01`–`11` are chapters; `12`/`13` are appendices A/B) |
| `figures/` | 49 figures (`fig01`–`fig49_*.png`), all generated by scripts and reproducible |
| `codes/chapters/` | Chapter-owned source, experiments, reports and assets for the guide, Chapters 1–11 and Appendices A/B; see the [code map](codes/chapters/README.md) |
| `codes/chapters/ch00/audio/MANIFEST.json` and chapter `audio/` folders | One manifest covers 109 book-synthesized WAVs in 27 groups, stored by chapter with parameters and hashes; generated by script |
| `codes/chapters/ch08/gss_audio/`, `codes/chapters/ch09/moving_audio/`, `codes/chapters/ch09/tracking_audio/`, `codes/chapters/appendix_b/room_audio/` | Four independent synthetic sets containing 5, 3, 2, and 18 WAVs respectively, with their own state, truth, observations or room-result report and manifests |
| `codes/chapters/ch02/real_audio/` | Real synchronized DEMAND excerpt, derived averages, separate manifest and data license |
| `codes/chapters/ch00/research/` | Detailed source research: algorithm steps, state and configuration, source entrypoints, failure experiments and industrial reproduction |
| `codes/chapters/*/reports/` | Small-scale run reports beside their algorithm chapters, separate from source acquisition and full paper benchmarks |
| `scripts/` | Plotting and build scripts (`make_figures.py`, `make_aec_figures.py`, `build_site.py`, `build_pdf.py`; see `scripts/README.md`) |
| `site/` | 20 pages: 14 tutorial pages (including the homepage) and 6 handbook pages under `research/`; reproducible build output |
| `dist/` | [Current combined PDF](dist/microphone-array-tutorial.pdf) and rebuildable HTML; the PDF has 638 bookmarks: 14 top-level, 119 second-level and 505 third-level |

## Chapters

| # | File | Topics | Level |
|---|---|---|---|
| Guide | `chapters/00_overview.md` | Navigation, three learning paths, figure map | Beginner |
| Ch 1 | `chapters/01_problem-definition.md` | Noise, reverberation, interference, self-noise, array gain, binaural cues | Beginner |
| Ch 2 | `chapters/02_basics-signal-model.md` | Delay, near/far field, signal model, reverberation, STFT/covariance, beampattern metrics | Intermediate |
| Ch 3 | `chapters/03_array-geometry.md` | Linear/circular/spherical/sparse arrays, endfire sensitivity, calibration | Intermediate |
| Ch 4 | `chapters/04_doa-estimation.md` | GCC-PHAT, SRP, TDOA geometry, Bartlett/Capon, MUSIC/ESPRIT, broadband focusing, DNN localization, CRLB | Advanced |
| Ch 5 | `chapters/05_beamforming.md` | DSB, superdirective, MVDR, LCMV, GSC, SPP, postfilters, spherical harmonics, DNN beamforming | Advanced |
| Ch 6 | `chapters/06_aec.md` | NLMS, PBFDAF, subband AEC, IPNLMS, RLS, Kalman/FDKF, double-talk, nonlinear echo, hybrid neural AEC, AEC3 | Advanced |
| Ch 7 | `chapters/07_wpe-dereverberation.md` | WPE derivation, Δ/K selection, online WPE, worked examples | Intermediate |
| Ch 8 | `chapters/08_speech-separation.md` | Mixing model, BSS, GSS, deep separation, TSE, continuous meeting separation, datasets | Intermediate |
| Ch 9 | `chapters/09_source-tracking.md` | Kalman worked example, particle filtering, PHD tracking, localization–tracking–beamforming interface | Intermediate |
| Ch 10 | `chapters/10_engineering-practice.md` | Reference pipeline, critical-path latency, SRO/calibration, resource budgets, evaluation | Intermediate |
| Ch 11 | `chapters/11_selection-guide.md` | Conditional selection, scenario constraints, verifiable specifications, exercises | Beginner |
| App A | `chapters/12_appendix-symbols-math.md` | Symbols, terminology, math refresher | Reference |
| App B | `chapters/13_appendix-guide.md` | Learning paths, field map, research frontier, debugging, the 17 comprehensive written exercises and E13-01–08 code exercises, reproduction guide | Reference |

## Quick start

```bash
# 1. Virtualenv and dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Optional room simulation uses pyroomacoustics 0.10.0 in a separate environment; see scripts/README.md

# 2. Run the equation-level teaching baselines and their regression tests
.venv/bin/python codes/chapters/ch00/cross_chapter/ch02_05_baselines.py
.venv/bin/python codes/chapters/ch00/cross_chapter/ch06_09_baselines.py
.venv/bin/python codes/chapters/ch10/examples/ch10_engineering_baselines.py
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_spatial
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_enhancement
.venv/bin/python -m codes.chapters.ch06.aec_advanced_exercises
.venv/bin/python -m codes.chapters.ch06.aec_algorithm_minicases
.venv/bin/python -m codes.chapters.ch00.cross_chapter.exercises_engineering
.venv/bin/python -m unittest discover -s tests -p 'test_codes*.py' -v

# 3. Generate 109 audio files first, then 49 figures (Figures 34–36 and 40–41 and 43–45 and 47 and 49 read the generated audio)
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_aec_figures.py

# 4. Build the multi-page site (outputs site/*.html)
.venv/bin/python scripts/build_site.py
# Open site/index.html in a browser (double-click works; formulas need internet for MathJax)

# 5. Build the combined PDF (uses bundled MathJax 3.2.2 resources offline;
#    requires local Google Chrome; set CHROME_BIN on non-macOS)
.venv/bin/python scripts/build_pdf.py
# For a reproducible cover date, add --build-date YYYY-MM-DD or set SOURCE_DATE_EPOCH

# 6. Pre-release checks
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/quality_check.py
```

Acquire official reference sources and verify local checkouts (Git required; acquisition needs network access):

```bash
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --all --report tmp/source-acquisition.json
.venv/bin/python codes/chapters/ch00/upstream/fetch_upstreams.py --verify --report tmp/source-verification.json
```

The tool preserves separate repositories and notices, omits common model/audio assets, and does not install or execute upstream programs. Source verification and experimental reproduction are recorded separately; see the [acquisition guide](codes/chapters/ch00/upstream/README.md).

The official HARKTOOL5 source archive uses separate SHA-256 locks, `fetch_archives.py`, and a verification report. The same guide documents its commands, restricted license, and selected source scope.

External formulas, algorithms, datasets, and standards should link to a DOI, standards body, or official project page whenever possible. Verify the cited title, authors, year, and exact table or section; an accessible URL alone is not sufficient evidence.

`codes/chapters/README.md` explains the boundary between the small teaching implementations and external reference systems. Review code, model, and dataset licenses separately.

## Learning paths

- **Path A (from scratch, self-paced units)**: Guide → 01 → 11.1/11.3 → 02/03 → 04 (GCC+SRP) → 05 (DSB+MVDR) → 06/07/08 → 09 → run `codes/chapters/` and reproduce all 49 figures.
- **Path B (deployment, 1 week)**: 11.1/11.2/11.3 for plan A/B/C → 05/06/07/08 → full Ch 10 → run the Ch. 10 engineering baselines → output latency/sync/calibration budgets.
- **Path C (research frontier)**: 02 (CRLB) → 03 (sparse arrays) → 04/05 frontier → 06/07/08 → 13.3 eight frontiers + 13.6 exercises.

## Conventions

- Core equations that are referenced across sections use MathJax `\tag{chapter-index}` labels and are cited as “see Eq. (5-1)”.
- Abbreviations spelled out on first use; dB uses 10log (power) / 20log (amplitude).
- Benchmark numbers always carry conditions and sources; simulation numbers carry implementation notes.
