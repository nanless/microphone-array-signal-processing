# An Accessible Guide to Microphone Array Signal Processing

A beginner-friendly Chinese tutorial on microphone array signal processing at graduate-entry level. It starts with “why use an array of microphones” and covers DOA estimation, beamforming, acoustic echo cancellation (AEC), dereverberation (WPE), speech separation, source tracking, engineering practice, and system selection.

The tutorial provides derivations, reproducible numerical examples, validity limits, 64 script-generated figures, and chapter-owned teaching code. It has 282 executable exercises and 109 [main-manifest synthetic WAVs](codes/chapters/ch00/audio/MANIFEST.json) in 27 groups. The [exercise and audio handbook](codes/chapters/ch00/research/05_exercises_and_audio.md) gives inputs, answers, listening conditions, and code entry points. These sounds are mathematical samples, not natural speech or formal listening tests; two Chapter 6 AEC audio groups also use parameters that do not exactly match exercises E06-07–20.

Chapter 4 provides [four independent known-unitary focusing WAVs](codes/chapters/ch04/focus_audio/MANIFEST.json). Steady-window phasors and actual PCM demonstrate multifrequency covariance rank for coherent sources; directions and tagged components are known, so this is not blind localization or a formal listening test.

Chapter 5 provides [four independent derivative-constraint WAVs](codes/chapters/ch05/derivative_audio/MANIFEST.json). The same two-tone target and observation noise compare single-constraint and zero-derivative weights. Target distortion, noise cost and actual PCM reference error are reported separately, with a shared causal reference and no fitted gain or time alignment. These are synthetic examples, not speech or device recordings.

Chapter 6 provides [six independent colored-reference APA WAVs](codes/chapters/ch06/apa_audio/MANIFEST.json). NLMS and two APA orders share one reference and path, train before a frozen holdout, and retain all 13 nonzero tail samples. Float component truth and actual PCM total power use separate measurements. Higher order has larger noisy holdout error in this fixed example; it is not a speech-quality or device ranking.

Chapter 7 adds [six independent known-path inverse WAVs](codes/chapters/ch07/mint_audio/MANIFEST.json). One four-tone source and shared post-path noise compare exact inversion with a constrained regularizer. The complete 512-sample tail is retained; theoretical noise gain and actual 27,200-sample PCM reference error are separate. This is not blind WPE, a measured room or a general MINT implementation.

Chapter 8 adds [six known-mask representation WAVs](codes/chapters/ch08/mask_audio/MANIFEST.json). Two tones compare bounded real, unbounded real and complex masks. A common envelope follows the full-record known-bin operation; analytic and actual 27,200-sample PCM errors are separate. No mask is estimated from a recording.

Chapter 10 adds [six noise-estimation mismatch WAVs](codes/chapters/ch10/noise_audio/MANIFEST.json). Shared target and changing noise compare a clean-prefix fixed estimate, a target-contaminated estimate and an offline known-variance control. Float target distortion, residual noise and their cross term are separate from actual PCM total error; the control uses extra truth, not blind adaptive estimation.

Chapter 11 adds [eight two-scenario selection WAVs](codes/chapters/ch11/scenario_audio/MANIFEST.json). Single- and two-tone targets share a 3500 Hz interferer and compare 3/9-tap FIR filters with a complete eight-sample tail. Actual integer PCM errors distinguish fixed scene weights, worst weights within an interval and worst individual scenes; they are not device rankings.

Chapter 3 also provides [three independent geometry WAVs](codes/chapters/ch03/geometry_audio/MANIFEST.json): a 32 kHz two-tone source and six-channel observations from two directions. Phase evidence uses the stated steady window and actual PCM, rather than playback or the transient envelope.

Six other independently catalogued synthetic sets provide [five binaural time/level cue WAVs](codes/chapters/ch01/binaural_audio/MANIFEST.json), [three finite-window STFT convolution WAVs](codes/chapters/ch02/stft_audio/MANIFEST.json), [five GSS WAVs and intermediate state](codes/chapters/ch08/gss_audio/MANIFEST.json), [three moving-source WAVs and trajectory truth](codes/chapters/ch09/moving_audio/MANIFEST.json), [two observation-to-tracking WAVs and frame records](codes/chapters/ch09/tracking_audio/MANIFEST.json), and [18 Appendix B white-noise room WAVs](codes/chapters/appendix_b/room_audio/MANIFEST.json) with a chart and [numerical report](codes/chapters/appendix_b/room_audio/RESULTS.json). None belongs to the main 109. A real synchronized DEMAND excerpt and three derivatives have separate [data and license documentation](codes/chapters/ch02/real_audio/README.md).

Use the [chapter code map](codes/chapters/README.md) to find teaching implementations, experiments, and reports, each with its stated scope. Chapter 6's [SpeexDSP interface probe on a real paired recording](codes/chapters/ch00/research/02_aec_wpe_separation.md#aec) uses local cached material only; it does not redistribute the recording or measure clean-component ERLE. The [source research handbook](codes/chapters/ch00/research/README.md) details algorithms, industrial configurations, original sources and licenses, executed experiments, and unverified boundaries.

中文版：[README.md](README.md)

## Layout

| Path | Description |
|---|---|
| `chapters/` | 14 tutorial documents in Markdown (`00_overview.md` is the homepage; `01`–`11` are chapters; `12`/`13` are appendices A/B) |
| `figures/` | 64 figures (`fig01`–`fig64_*.png`), all generated by scripts and reproducible |
| `codes/chapters/` | Chapter-owned source, experiments, reports and assets for the guide, Chapters 1–11 and Appendices A/B; see the [code map](codes/chapters/README.md) |
| `codes/chapters/ch00/audio/MANIFEST.json` and chapter `audio/` folders | One manifest covers 109 book-synthesized WAVs in 27 groups, stored by chapter with parameters and hashes; generated by script |
| `codes/chapters/ch01/binaural_audio/`, `codes/chapters/ch08/gss_audio/`, `codes/chapters/ch09/moving_audio/`, `codes/chapters/ch09/tracking_audio/`, `codes/chapters/appendix_b/room_audio/` | Five independent synthetic sets containing 5, 5, 3, 2, and 18 WAVs respectively, with their own state, truth, observations or room-result report and manifests |
| `codes/chapters/ch02/stft_audio/`, `codes/chapters/ch03/geometry_audio/` | Two further independent synthetic sets, three WAVs each; complete convolution and steady-window multifrequency phase have separate scores, source hashes and PCM manifests |
| `codes/chapters/ch04/focus_audio/` | Four independent known-unitary focusing WAVs: two mono sources and two four-channel observations, with steady-window multifrequency rank scores |
| `codes/chapters/ch05/derivative_audio/` | Four independent derivative-constraint WAVs: one mono reference, a three-channel array and two outputs, with seven source hashes and actual PCM reference errors |
| `codes/chapters/ch06/apa_audio/` | Six colored-reference APA WAVs: training and frozen holdout, a complete 13-sample tail and separate float/PCM scores |
| `codes/chapters/ch07/mint_audio/` | Six known sparse-path inverse WAVs: complete 512-sample tails, common gain and an independent parameter/score manifest |
| `codes/chapters/ch02/real_audio/` | Real synchronized DEMAND excerpt, derived averages, separate manifest and data license |
| `codes/chapters/ch00/research/` | Detailed source research: algorithm steps, state and configuration, source entrypoints, failure experiments and industrial reproduction |
| `codes/chapters/ch00/upstream/_downloads/` | Git-ignored local cache of separately acquired upstream source; it may contain local changes and is not part of the published tutorial or tracked documentation. See the [acquisition guide](codes/chapters/ch00/upstream/README.md) |
| `codes/chapters/*/reports/` | Small-scale run reports beside their algorithm chapters, separate from source acquisition and full paper benchmarks |
| `scripts/` | Plotting and build scripts (`make_figures.py`, `make_aec_figures.py`, `build_site.py`, `build_pdf.py`; see `scripts/README.md`) |
| `site/` | 20 pages: 14 tutorial pages (including the homepage) and 6 handbook pages under `research/`; reproducible build output |
| `dist/` | [Current combined PDF](dist/microphone-array-tutorial.pdf) and rebuildable HTML; the PDF has 692 bookmarks: 14 top-level, 121 second-level and 557 third-level |

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
| App B | `chapters/13_appendix-guide.md` | Learning paths, field map, research frontier, debugging, the 17 comprehensive written exercises and E13-01–10 code exercises, reproduction guide | Reference |

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

# 3. Generate 109 audio files first, then 64 figures (Figures 34–36 and 40–41 and 43–45 and 47 and 49 and 58–60 and 63–64 read the generated audio)
.venv/bin/python codes/chapters/ch00/examples/generate_audio_samples.py
.venv/bin/python -m codes.chapters.ch06.examples.generate_apa_audio
.venv/bin/python -m codes.chapters.ch07.examples.mint_teaching_demo
.venv/bin/python -m codes.chapters.ch08.examples.mask_representation_demo
.venv/bin/python -m codes.chapters.ch10.examples.generate_noise_mismatch
.venv/bin/python -m codes.chapters.ch11.examples.generate_selection_audio  # 8 scenario WAVs; --check is read-only
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

- **Path A (from scratch, self-paced units)**: Guide → 01 → 11.1/11.3 → 02/03 → 04 (GCC+SRP) → 05 (DSB+MVDR) → 06/07/08 → 09 → run the teaching code for the chapters studied and reproduce their corresponding figures.
- **Path B (deployment)**: 11.1/11.2/11.3 for plan A/B/C → 05/06/07/08 → full Ch 10 → run the Ch. 10 engineering baselines → output latency/sync/calibration budgets.
- **Path C (research frontier)**: 02 (CRLB) → 03 (sparse arrays) → 04/05 frontier → 06/07/08 → 13.3 eight frontiers + 13.6 exercises.

## Conventions

- Core equations that are referenced across sections use MathJax `\tag{chapter-index}` labels and are cited as “see Eq. (5-1)”.
- Abbreviations spelled out on first use; dB uses 10log (power) / 20log (amplitude).
- Benchmark numbers always carry conditions and sources; simulation numbers carry implementation notes.
