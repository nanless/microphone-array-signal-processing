"""Generate figure 73 from the actual E05-23 PCM and its checked manifest.

Run from the repository root: .venv/bin/python scripts/make_beamforming_figures.py
Existing figure generators retain their independent source identities.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_figures import plt, finalize_figure, C_BLUE, C_RED, C_GREEN
from codes.chapters.ch00.core.audio_samples import read_pcm16
from codes.chapters.ch05.examples.generate_phase_audio import DEFAULT_OUTPUT, check_assets
import numpy as np


def measurements():
    manifest = check_assets(DEFAULT_OUTPUT)
    inputs = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
              for path in (*manifest['source_sha256'], 'scripts/make_figures.py',
                           'scripts/make_beamforming_figures.py')}
    manifest_sha = hashlib.sha256((DEFAULT_OUTPUT/'MANIFEST.json').read_bytes()).hexdigest()
    report = {'figure': 73, 'exercise_id': 'E05-23', 'source_sha256': inputs,
              'audio_manifest_sha256': manifest_sha,
              'wav_sha256': {name: record['sha256'] for name, record in manifest['files'].items()},
              'parameters': manifest['parameters'], 'samples': manifest['samples'],
              'limits': manifest['limits'],
              'plot_scope': 'actual PCM excerpt, known-frequency projections, actual PCM power and total reference error; no noise or SNR experiment'}
    signals = {name: read_pcm16((DEFAULT_OUTPUT/sample['file']).read_bytes())[1][0]
               for name, sample in manifest['samples'].items()}
    return manifest, report, signals


def main():
    manifest, report, signals = measurements()
    report_path = ROOT/'codes/chapters/ch05/reports/figure73_phase_reference.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    names, labels, colors = ('reference', 'flip', 'quadrature'), ('参考', '高频反号', '高频正交'), (C_BLUE, C_RED, C_GREEN)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), constrained_layout=True)
    fs, start, count = 16000, 2400, 64
    time = np.arange(count)/fs*1000
    for name, label, color in zip(names, labels, colors):
        axes[0, 0].plot(time, signals[name][start:start+count], label=label, color=color, linewidth=1.5)
    axes[0, 0].set(title='(a) 实际 PCM：同钟、同增益', xlabel='从第 2400 点起的时间 / ms', ylabel='解码幅度')
    axes[0, 0].set_ylim(-.225, .275)
    axes[0, 0].legend(ncol=3, loc='upper right', fontsize=11)
    x = np.arange(2)
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        pair = np.asarray(manifest['samples'][name]['pcm_measurements']['response_real_imag'])
        phase = np.angle(pair[:, 0]+1j*pair[:, 1])*180/np.pi
        # The signed half-turn is the same phase class; choose +180 for clarity.
        phase[np.isclose(abs(phase), 180)] = 180
        axes[0, 1].plot(x+(i-1)*.03, phase, 'o', color=color, label=label)
    axes[0, 1].set(title='(b) PCM 分量相对参考的相位', xticks=x, xticklabels=['500 Hz', '1500 Hz'],
                   yticks=[-90, 0, 90, 180], ylabel='输出相对相位 / °')
    axes[0, 1].legend(loc='upper left')
    power = [manifest['samples'][n]['pcm_integer_measurements']['mean_square_all_channels'] for n in names]
    error = [manifest['samples'][n]['pcm_integer_measurements']['total_reference_mse_per_channel'][0] for n in names]
    for ax, values, title, ylabel in ((axes[1, 0], power, '(c) 稳窗实际 PCM 功率', '均方幅度'),
                                    (axes[1, 1], error, '(d) 同窗实际 PCM 总参考误差', 'MSE')):
        ax.bar(np.arange(3), values, color=colors, alpha=.8)
        ax.set(title=title, xticks=np.arange(3), xticklabels=labels, ylabel=ylabel)
        ax.set_ylim(0, max(values)*1.32)
        for i, value in enumerate(values):
            ax.text(i, value+max(values)*.035, f'{value:.8f}', ha='center')
    axes[1, 0].axhline(.01, color='black', linestyle='--', label='解析三功率均为 0.01')
    axes[1, 0].legend(loc='upper center')
    for i, value in enumerate((0., .02, .01)):
        axes[1, 1].plot([i-.3, i+.3], [value, value], 'k--', label='解析误差' if i == 0 else None)
    axes[1, 1].legend(loc='upper center')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('已知逐频相位控制：等解析功率仍有不同波形')
    finalize_figure(fig)
    metadata = {'SourceScript': 'scripts/make_beamforming_figures.py',
                'SourceScriptDigest': report['source_sha256']['scripts/make_beamforming_figures.py'],
                'GeneratorInputs': json.dumps(report['source_sha256'], sort_keys=True),
                'AudioManifestDigest': report['audio_manifest_sha256'],
                'NumericalReportDigest': hashlib.sha256(report_path.read_bytes()).hexdigest()}
    destination = ROOT/'figures/fig73_phase_reference.png'
    fig.savefig(destination, dpi=150, bbox_inches='tight', facecolor='white', metadata=metadata)
    plt.close(fig)
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
