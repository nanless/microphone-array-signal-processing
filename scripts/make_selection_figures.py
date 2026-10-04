"""Figure79: known covariance, target mismatch and explicit evidence thresholds.

The Chapter 11 physical case reuses the unique Chapter 5 solvers. This script
plots its pure numerical result, writes no audio and makes no device/ASR claim.
Only the current figure/report or ordinary external paths may be written.
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
from codes.chapters.ch11.core.selection_physics import beamformer_selection_case

FIGURE = ROOT/'figures/fig79_selection_evidence.png'
REPORT = ROOT/'codes/chapters/ch11/reports/figure79_selection_evidence.json'
SOURCE_PATHS = (
    'scripts/make_selection_figures.py',
    'codes/chapters/ch11/core/selection_physics.py',
    'codes/chapters/ch11/core/selection.py',
    'codes/chapters/ch03/core/geometry.py',
    'codes/chapters/ch05/core/beamforming.py',
    'codes/chapters/ch04/core/covariance.py',
    'codes/chapters/ch02/core/conventions.py',
    'codes/chapters/ch00/io_contracts.py',
)
CANDIDATES = ('ds', 'mvdr_0', 'mvdr_0_1', 'mvdr_1')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _destination(path, official, suffix):
    path = validate_report_destination(path)
    if path.suffix != suffix or (path.resolve().is_relative_to(ROOT) and path.resolve() != official):
        raise ValueError('only the current figure/report or an ordinary external destination is allowed')
    return path


def _sources():
    result = {}
    for relative in SOURCE_PATHS:
        source = validate_parent_chain(ROOT/relative)
        if not source.is_file():
            raise ValueError('figure source must be an ordinary file: '+relative)
        result[relative] = _sha(source.read_bytes())
    return result


def generate_figure(figure_path=FIGURE, report_path=REPORT):
    # Reject bad destinations before computing or creating any output parent.
    figure_path = _destination(figure_path, FIGURE, '.png')
    report_path = _destination(report_path, REPORT, '.json')
    sources = _sources()
    case = beamformer_selection_case()
    rows = case['candidates']
    if [row['candidate_id'] for row in rows] != list(CANDIDATES):
        raise ValueError('figure requires the four declared candidates in their fixed order')
    parameters = case['parameters']
    target_power = parameters['target_power']
    values = np.array([[row[key] for key in ('actual_noise_power', 'actual_NMSE',
                                            'response_amplitude_error', 'WNG_dB', 'DI_dB')]
                       for row in rows], dtype=float)
    if values.shape != (4, 5) or not np.all(np.isfinite(values)) or target_power != 1.:
        raise ValueError('finite fixed single-frequency scores and target power 1 are required')
    if any(row['measured_latency_ms'] is not None for row in rows) or case['device_selected'] is not None:
        raise ValueError('this figure has no measured device latency or selected device')
    labels = [row['label'] for row in rows]
    plot_data = {
        'candidate_ids': list(CANDIDATES), 'labels': labels,
        'noise_power_normalized_by_target': (values[:, 0]/target_power).tolist(),
        'actual_NMSE': values[:, 1].tolist(),
        'response_amplitude_error': values[:, 2].tolist(),
        'WNG_dB': values[:, 3].tolist(),
        'DI_dB_separate_diffuse_model': values[:, 4].tolist(),
        'response_amplitude_error_upper_limit': parameters['response_amplitude_error_upper_limit'],
        'WNG_lower_limit_dB': parameters['WNG_lower_limit_dB'],
        'acoustic_verdicts': [row['acoustic_verdict'] for row in rows],
        'device_verdicts': [row['device_verdict'] for row in rows],
        'DI_scope': 'DI uses the separate 3-D isotropic diffuse coherence Gamma, not the actual R_n',
        'decision_scope': 'known nominal/actual directions; acoustic-only eligibility does not supply missing device evidence',
    }
    report = {'schema_version': 1, 'exercise_id': 'E11-27',
              'script_sha256': sources['scripts/make_selection_figures.py'],
              'source_sha256': sources, 'numerical_case': case, 'plot_data': plot_data,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'matplotlib': matplotlib.__version__},
              'limits': 'Known single-frequency ensemble statistics and prescribed 5-degree target mismatch only; '
                        'no WAV, estimated covariance/DOA, worst-case optimizer, device latency, ASR or listening test.'}
    json.dumps(report, allow_nan=False)
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei',
                           'Arial Unicode MS', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'font.size': 11.5, 'axes.titlesize': 12.5,
        'axes.labelsize': 11.5, 'legend.fontsize': 11.5,
        'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5})
    # Smaller physical canvas and larger text preserve readability when the
    # complete image is embedded at 165 mm or 860 CSS pixels (AGENTS 5.3).
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 8.3))
    x = np.arange(4)
    colors = ['#258065' if row['acoustic_verdict'] == 'pass' else '#be5146' for row in rows]
    noise, total, error, wng, di = values.T
    axes[0, 0].bar(x, noise/target_power, color='#4b83b1', width=.55,
                   label='输出噪声 / $P_s$（当前 $R_n$）')
    axes[0, 0].plot(x, total, 'kd', markersize=6, label='总 NMSE（含目标失配）')
    for position, value in zip(x, noise):
        # The alpha=0 total-error marker lies just above the noise bar.
        # Move its noise label left so the two different quantities do not collide.
        offset, alignment = ((-12, 9), 'right') if position == 1 else ((0, 9), 'center')
        axes[0, 0].annotate(f'{value:.4f}', (position, value), xytext=offset,
                           textcoords='offset points', ha=alignment, fontsize=11.5)
    axes[0, 0].set(title='(a) 当前 $R_n$：噪声与总误差', ylabel='无量纲：噪声 / $P_s$ 或 NMSE',
                   xticks=x, xticklabels=labels, ylim=(0, 1.22))
    axes[0, 0].legend(loc='upper right'); axes[0, 0].grid(axis='y', alpha=.2)
    axes[0, 1].bar(x, error, color=colors, width=.55)
    limit = parameters['response_amplitude_error_upper_limit']
    axes[0, 1].axhline(limit, color='#444444', linestyle='--', label=f'硬上限 {limit:.2f}')
    for position, value in zip(x, error):
        axes[0, 1].annotate(f'{value:.5f}', (position, value), xytext=(0, 7),
                           textcoords='offset points', ha='center', fontsize=11.5)
    axes[0, 1].set(title='(b) 真实方向 5°：目标响应偏差', ylabel='$|w^H a_{true}-1|$（无量纲）',
                   xticks=x, xticklabels=labels, ylim=(0, .285))
    axes[0, 1].legend(loc='upper right'); axes[0, 1].grid(axis='y', alpha=.2)
    axes[1, 0].bar(x, wng, color=colors, width=.55)
    wng_limit = parameters['WNG_lower_limit_dB']
    axes[1, 0].axhline(wng_limit, color='#444444', linestyle='--', label=f'硬下限 {wng_limit:g} dB')
    for position, value in zip(x, wng):
        axes[1, 0].annotate(f'{value:.2f}', (position, value), xytext=(0, 7 if value >= 0 else -16),
                           textcoords='offset points', ha='center', fontsize=11.5)
    axes[1, 0].set(title='(c) WNG：绿色满足全部声学条件', ylabel='WNG（dB）',
                   xticks=x, xticklabels=labels, ylim=(-11.5, 6.8))
    axes[1, 0].legend(loc='upper center', bbox_to_anchor=(.53, 1.0))
    axes[1, 0].grid(axis='y', alpha=.2)
    axes[1, 1].axis('off'); axes[1, 1].set_title('(d) 另模型 DI 与完整设备证据', pad=12)
    verdict = {'pass': '通过', 'fail': '失败', 'undetermined': '证据不足'}
    cells = [[label, f'{value:.2f}', verdict[row['acoustic_verdict']], verdict[row['device_verdict']]]
             for label, value, row in zip(labels, di, rows)]
    table = axes[1, 1].table(cellText=cells, colLabels=['候选', '另模型 DI (dB)', '声学条件', '完整设备'],
                            colWidths=[.20, .31, .23, .26], cellLoc='center', bbox=[0, .45, 1, .46])
    table.auto_set_font_size(False); table.set_fontsize(11.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#c4c9cf')
        if row == 0:
            cell.set_facecolor('#e9eff4')
        elif col == 2:
            cell.get_text().set_color(colors[row-1])
    axes[1, 1].text(.5, .27, 'DI 使用独立三维各向同性 $\\Gamma$，\n不是 (a) 的 $R_n$。',
                    ha='center', transform=axes[1, 1].transAxes, fontsize=11.5, linespacing=1.4)
    axes[1, 1].text(.5, .05, '声学合格者中 α=1 的实际 NMSE 更低；\n全部缺少设备延迟证据，不能宣布整机已选定。',
                    ha='center', transform=axes[1, 1].transAxes, fontsize=11.5, linespacing=1.5)
    fig.suptitle('图79  同一已知单频模型：先核硬条件，再比较损失', fontsize=16)
    fig.text(.5, .012, '三麦间距 4 cm · 1 kHz · 名义目标 0° / 干扰 20° / 实际目标 5° · $P_s=1$\n'
             '已知统计的数学控制；不产生音频，不代表未知失配保证、设备实测或语音识别成绩。',
             ha='center', fontsize=11.5)
    fig.tight_layout(rect=(0, .075, 1, .95), h_pad=2.0, w_pad=2.0)
    if _sources() != sources:
        plt.close(fig)
        raise ValueError('figure sources changed during numerical preparation')
    write_json_report(report_path, report)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _destination(figure_path, FIGURE, '.png')
    fig.savefig(figure_path, dpi=150, bbox_inches='tight', facecolor='white', metadata={
        'SourceScript': 'scripts/make_selection_figures.py', 'SourceScriptDigest': report['script_sha256'],
        'GeneratorInputs': json.dumps(sources, sort_keys=True),
        'NumericalReportDigest': _sha(report_path.read_bytes())})
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-image', type=Path, default=FIGURE)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    result = generate_figure(args.output_image, args.report)
    print(json.dumps({'figure': str(args.output_image), 'report': str(args.report),
                      'exercise_id': result['exercise_id'], 'source_sha256': result['source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
