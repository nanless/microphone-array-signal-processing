"""Figure78: prescribed channel failure, known covariance and actual PCM.

Only the explicit current figure/report or ordinary external paths are written.
This is a fixed time-domain teaching control, not fault diagnosis or a
frequency-wise covariance estimate. Asset checks and scores read actual WAVs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from codes.chapters.ch00.io_contracts import validate_parent_chain, validate_report_destination, write_json_report
from codes.chapters.ch10.examples.generate_channel_audio import (
    DEFAULT_OUTPUT, FILE_NAMES, SOURCE_PATHS as AUDIO_SOURCES, check_assets, integer_measurements, _codes)

FIGURE = ROOT/'figures/fig78_channel_failure.png'
REPORT = ROOT/'codes/chapters/ch10/reports/figure78_channel_failure.json'
SOURCE_PATHS = ('scripts/make_channel_figures.py', *AUDIO_SOURCES)
OUTPUT_ROLES = ('healthy_output', 'stale_output', 'recomputed_output')
LABELS = ('健康全阵', '故障后旧权重', '故障后子阵重算')
COLORS = ('#286ca3', '#d75545', '#258065')


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _destination(path, official, suffix):
    path = validate_report_destination(path)
    resolved = path.resolve()
    if path.suffix != suffix or (resolved.is_relative_to(ROOT) and resolved != official):
        raise ValueError('only the current figure/report or ordinary external destination is allowed')
    return path


def generate_figure(figure_path=FIGURE, report_path=REPORT, *, audio_directory=DEFAULT_OUTPUT):
    # Both destinations are checked before reading inputs or producing anything.
    figure_path = _destination(figure_path, FIGURE, '.png')
    report_path = _destination(report_path, REPORT, '.json')
    for destination in (figure_path, report_path):
        validate_report_destination(destination, forbidden_roots=(audio_directory,))
    sources = {}
    for name in SOURCE_PATHS:
        source = validate_parent_chain(ROOT/name)
        if not source.is_file():
            raise ValueError('missing ordinary figure source: '+name)
        sources[name] = _sha(source.read_bytes())
    data = check_assets(audio_directory)
    directory = validate_parent_chain(audio_directory)
    blobs = {filename: (directory/filename).read_bytes() for filename in FILE_NAMES.values()}
    input_sha = {filename: _sha(blob) for filename, blob in blobs.items()}
    input_sha['MANIFEST.json'] = _sha((directory/'MANIFEST.json').read_bytes())
    codes = {name: _codes(blobs[file], 3 if name.endswith('_array') else 1) for name, file in FILE_NAMES.items()}
    # Re-read and score all directory PCM, preserving the exact integer units.
    integers = {name: integer_measurements(blobs[file], blobs[FILE_NAMES['reference']], len(codes[name]))
                for name, file in FILE_NAMES.items()}
    for name, row in integers.items():
        if row != data['samples'][name]['pcm_integer_measurements']:
            raise ValueError('actual figure PCM scores differ from the checked manifest')
    lo, hi = 2400, 2528
    target_gain = [data['analytic'][name]['target_gain'] for name in OUTPUT_ROLES]
    diagnostic = [integers[name]['projection_gain_per_channel'][0] for name in OUTPUT_ROLES]
    nmse_analytic = [data['analytic'][name]['reference_NMSE'] for name in OUTPUT_ROLES]
    nmse_pcm = [integers[name]['reference_NMSE_per_channel'][0] for name in OUTPUT_ROLES]
    report = {'schema_version': 1, 'exercise_id': 'E10-34', 'script_sha256': sources['scripts/make_channel_figures.py'],
        'source_sha256': sources, 'input_sha256': input_sha,
        'parameters': data['parameters'], 'model': data['model'], 'analytic': data['analytic'],
        'samples': data['samples'], 'files': data['files'], 'limits': data['limits'],
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'plot_data': {'output_roles': list(OUTPUT_ROLES), 'effective_weights_after_prescribed_fault':
            [[1., -.5, .5], [0., -.5, .5], [0., .5, .5]],
            'target_gain_analytic': target_gain, 'projection_gain_pcm_diagnostic': diagnostic,
            'projection_scope': 'same-window actual PCM reference projection diagnostic; no gain compensation',
            'reference_NMSE_analytic': nmse_analytic, 'reference_NMSE_pcm': nmse_pcm,
            'waveform_interval_samples': [lo, hi], 'waveform_time_ms': (np.arange(lo, hi)*1000/16000).tolist(),
            'waveform_pcm': {name: (codes[name][0, lo:hi]/32768).tolist() for name in ('reference', *OUTPUT_ROLES)},
            'scoring_interval_samples': [2400, 29600], 'samples_per_channel': 27200,
            'known_covariance_units': 'waveform amplitude squared; broadband time-window covariance, not per-frequency SCM'}}
    json.dumps(report, allow_nan=False)
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei',
                           'Arial Unicode MS', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 10.0), gridspec_kw={'height_ratios': [.8, 1.15, 1.1]})
    axes[0].axis('off')
    axes[0].set_title('(a) 已知噪声时间协方差与解析控制：$R_n=0.005BB^T$', pad=12)
    rows = [[LABELS[i], ('[1, −0.5, 0.5]', '[0, −0.5, 0.5]', '[0, 0.5, 0.5]')[i],
             str(int(target_gain[i])), f"{data['analytic'][name]['noise_mean_square']:.4f}"]
            for i, name in enumerate(OUTPUT_ROLES)]
    table = axes[0].table(cellText=rows, colLabels=['控制', '实际参与的通道权重', '目标响应', '已知输出噪声均方'],
                         colWidths=[.23, .32, .15, .30], cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#c4c9cf')
        if row == 0:
            cell.set_facecolor('#e9eff4')
        elif col == 0:
            cell.set_facecolor('#f5f7f8')
    axes[0].text(.5, -.08, '通道 0 失效已由题设指定；其零输入使旧权重的有效首项为 0。',
                 transform=axes[0].transAxes, ha='center', fontsize=11)
    time = report['plot_data']['waveform_time_ms']
    axes[1].plot(time, report['plot_data']['waveform_pcm']['reference'], color='#333333', linewidth=2,
                 label='实际 PCM 参考（500 Hz）')
    for name, label, color in zip(OUTPUT_ROLES, LABELS, COLORS):
        axes[1].plot(time, report['plot_data']['waveform_pcm'][name], color=color, linewidth=1.2, alpha=.85, label=label)
    axes[1].set(xlabel='共同采样时轴 (ms)', ylabel='实际 PCM / 32768',
                title='(b) 同一 128 点片段；各输出仍保留所声明的噪声分量')
    axes[1].grid(alpha=.2); axes[1].legend(ncol=2, loc='upper right')
    x = np.arange(3); width = .32
    bars1 = axes[2].bar(x-width/2, diagnostic, width, color='#5591b7', label='实际 PCM 参考投影（诊断）')
    bars2 = axes[2].bar(x+width/2, nmse_pcm, width, color='#d77959', label='实际 PCM 总 NMSE')
    axes[2].plot(x-width/2, target_gain, 'k_', markersize=17, markeredgewidth=2, label='解析目标响应')
    axes[2].plot(x+width/2, nmse_analytic, 'kd', markersize=5, label='解析总 NMSE')
    for bars in (bars1, bars2):
        for bar in bars:
            axes[2].annotate(f'{bar.get_height():.6f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                             xytext=(0, 7), textcoords='offset points', ha='center', fontsize=10)
    axes[2].set(xticks=x, xticklabels=LABELS, ylim=(-.13, 2.0), ylabel='无量纲比值',
                title='(c) 总误差相近，目标保留不同；不对输出拟合增益或时移')
    axes[2].grid(axis='y', alpha=.2); axes[2].legend(ncol=2, loc='upper left')
    fig.text(.5, .015, '16 kHz · 共同增益 1 · 评分 [2400, 29600) 共 27200 点\n'
             '确定性正交频点与已知宽带时间统计；不是逐频 SCM、盲故障诊断、设备成绩或正式听测。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, 1), h_pad=2.3)
    # All numerical/metadata work succeeds before either output is touched.
    write_json_report(report_path, report, forbidden_roots=(audio_directory,))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _destination(figure_path, FIGURE, '.png')
    fig.savefig(figure_path, dpi=150, bbox_inches='tight', facecolor='white', metadata={
        'SourceScript': 'scripts/make_channel_figures.py', 'SourceScriptDigest': report['script_sha256'],
        'GeneratorInputs': json.dumps(sources, sort_keys=True),
        'AudioManifestDigest': input_sha['MANIFEST.json'], 'NumericalReportDigest': _sha(report_path.read_bytes())})
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-image', type=Path, default=FIGURE)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--audio-directory', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = generate_figure(args.output_image, args.report, audio_directory=args.audio_directory)
    print(json.dumps({'figure': str(args.output_image), 'report': str(args.report),
                      'exercise_id': data['exercise_id'], 'input_sha256': data['input_sha256']}, indent=2))


if __name__ == '__main__':
    main()
