"""Figure 75: known source-time indices and actual PCM prediction controls.

This is a single real-regressor illustration, not an STFT-WPE implementation.
The last panel is an independent statistical free-decay model, not PCM scoring.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.make_figures import plt, finalize_figure, C_BLUE, C_RED, C_GREEN
from codes.chapters.ch00.core.audio_samples import read_pcm16
from codes.chapters.ch07.examples.generate_delay_audio import DEFAULT_OUTPUT, check_assets


def measurements():
    manifest = check_assets(DEFAULT_OUTPUT)
    source_paths = (*manifest['source_sha256'], 'scripts/make_figures.py',
                    'scripts/make_delay_figures.py')
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in source_paths}
    signals = {name: read_pcm16((DEFAULT_OUTPUT/row['file']).read_bytes())[1]
               for name, row in manifest['samples'].items()}
    start, stop = 1536, 30720
    reference = np.rint(signals['reference'][0, start:stop]*32768).astype(np.int64)
    denominator = sum(int(x)*int(x) for x in reference)
    errors = {}
    for name in ('common_residual', 'aligned_residual'):
        output = np.rint(signals[name][0, start:stop]*32768).astype(np.int64)
        energy = sum((int(a)-int(b))**2 for a, b in zip(output, reference))
        errors[name] = {'E': energy, 'D': denominator, 'nmse': energy/denominator}
    hops = np.arange(76)
    # T60 is defined by POWER falling 60 dB, hence the factor 6 rather than 3.
    decay = 10.**(-6*hops*.008/.6)
    wrong_decay = 10.**(-3*hops*.008/.6)
    report = {'figure': 75, 'exercise_ids': ['E07-22', 'E07-23', 'E07-24'],
              'source_sha256': sources,
              'audio_manifest_sha256': hashlib.sha256((DEFAULT_OUTPUT/'MANIFEST.json').read_bytes()).hexdigest(),
              'wav_sha256': {name: row['sha256'] for name, row in manifest['files'].items()},
              'parameters': manifest['parameters'], 'pcm_integer_errors': errors,
              'scoring_interval_samples': [start, stop],
              'source_time_control': {'receiver_indices': [6, 7, 8, 9],
                  'reference': [0, 1, 0, 0], 'common_history': [0, 1, 0, 0],
                  'aligned_history': [1, 0, 0, 0]},
              'power_decay_control': {'T60_seconds': .6, 'hop_seconds': .008,
                  'hops': hops.tolist(), 'relative_power': decay.tolist(),
                  'wrong_amplitude_factor_used_as_power': wrong_decay.tolist()},
              'limits': manifest['limits'],
              'plot_scope': 'known integer arrivals and one real fixed-window regression, then applied to the full retained signal; free-decay panel is a separate statistical model; no blind delays or full WPE'}
    return manifest, report, signals


def main():
    _manifest, report, signals = measurements()
    report_path = ROOT/'codes/chapters/ch07/reports/figure75_prediction_delays.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.3), constrained_layout=True)
    t = np.arange(6, 10)
    for offset, key, color, label in (
            (-.14, 'reference', C_BLUE, '当前参考'),
            (0., 'common_history', C_RED, '统一历史：同源时刻'),
            (.14, 'aligned_history', C_GREEN, '校正历史：早 3 单位')):
        values = report['source_time_control'][key]
        axes[0, 0].vlines(t+offset, 0, values, colors=color, linewidth=2)
        axes[0, 0].plot(t+offset, values, 'o', color=color, label=label)
    axes[0, 0].set(title='(a) E22 四点手算：统一历史可含当前目标',
                   xlabel='接收时间索引 t / 无量纲单位', ylabel='给定脉冲幅度',
                   xticks=t, ylim=(-.13, 1.6))
    axes[0, 0].legend(loc='upper right', fontsize=10)
    begin, end = 1536, 3072
    times = np.arange(end-begin)/16000*1000
    for key, label, color, style in (
            ('reference', '当前参考（实际 PCM）', C_BLUE, '-'),
            ('common_history', '统一历史：与参考逐点相同', C_RED, '--'),
            ('aligned_history', '校正历史：晚 24 ms', C_GREEN, '-')):
        axes[0, 1].plot(times, signals[key][0, begin:end], style,
                        color=color, linewidth=1.1, label=label)
    axes[0, 1].set(title='(b) E24 同一接收时轴：一个 96 ms 周期',
                   xlabel='从第 1536 点起 / ms', ylabel='PCM 解码幅度',
                   xlim=(0, 96), ylim=(-.14, .18))
    axes[0, 1].legend(loc='upper left', fontsize=10)
    values = [report['pcm_integer_errors'][k]['nmse']
              for k in ('common_residual', 'aligned_residual')]
    axes[1, 0].bar([0, 1], [1, 0], width=.5, color=[C_RED, C_GREEN],
                   alpha=.3, label='给定控制的解析误差')
    axes[1, 0].plot([0, 1], values, 'ko', markersize=7, label='实际 PCM 的 E/D')
    axes[1, 0].set(title='(c) E24 目标误消：共同参考、29184 点',
                   xticks=[0, 1], xticklabels=['统一历史：g=1', '校正历史：g=0'],
                   ylabel='参考归一误差 E/D / 无量纲', ylim=(-.1, 1.45))
    axes[1, 0].text(.5, 1.15, 'D = 9715561194\n不拟合输出增益或时延',
                    ha='center', va='center', fontsize=10)
    axes[1, 0].legend(loc='center right', fontsize=10)
    decay = report['power_decay_control']
    axes[1, 1].semilogy(decay['hops'], decay['relative_power'], color=C_BLUE,
                        label='正确：功率每帧乘 0.831764')
    axes[1, 1].semilogy(decay['hops'], decay['wrong_amplitude_factor_used_as_power'],
                        '--', color=C_RED, label='错误：把幅度倍率用于功率')
    axes[1, 1].set(title='(d) E23 独立自由衰减模型：T60=0.6 s',
                   xlabel='已停止激励后的 8 ms 步数', ylabel='相对晚期谱功率',
                   xlim=(0, 75), ylim=(5e-7, 3))
    axes[1, 1].text(73, 7e-5, '−60 dB', ha='right', fontsize=10, color=C_BLUE)
    axes[1, 1].text(73, .02, '只有 −30 dB', ha='right', fontsize=10, color=C_RED)
    axes[1, 1].legend(loc='upper right', fontsize=9)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('保护延迟先核源时刻；晚期衰减先核功率单位')
    finalize_figure(fig)
    metadata = {'SourceScript': 'scripts/make_delay_figures.py',
                'SourceScriptDigest': report['source_sha256']['scripts/make_delay_figures.py'],
                'GeneratorInputs': json.dumps(report['source_sha256'], sort_keys=True),
                'AudioManifestDigest': report['audio_manifest_sha256'],
                'NumericalReportDigest': hashlib.sha256(report_path.read_bytes()).hexdigest()}
    destination = ROOT/'figures/fig75_prediction_delays.png'
    fig.savefig(destination, dpi=150, bbox_inches='tight', facecolor='white', metadata=metadata)
    plt.close(fig)
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
