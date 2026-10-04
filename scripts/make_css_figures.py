"""Figure 76: fixed source-slot polarity/gain stitching and actual PCM errors.

The signed-amplitude curves are prescribed model coefficients. PCM scores use
one common reference and explicit windows, without reference gain alignment.
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
from codes.chapters.ch08.examples.generate_css_audio import DEFAULT_OUTPUT, check_assets


def measurements():
    manifest = check_assets(DEFAULT_OUTPUT)
    paths = (*manifest['source_sha256'], 'scripts/make_figures.py', 'scripts/make_css_figures.py')
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    signals = {key: read_pcm16((DEFAULT_OUTPUT/row['file']).read_bytes())[1]
               for key,row in manifest['samples'].items()}
    report = {'figure': 76, 'exercise_ids': ['E08-29','E08-32'], 'source_sha256':sources,
        'audio_manifest_sha256':hashlib.sha256((DEFAULT_OUTPUT/'MANIFEST.json').read_bytes()).hexdigest(),
        'wav_sha256':{n:r['sha256'] for n,r in manifest['files'].items()},
        'parameters':manifest['parameters'], 'overlap_fit':manifest['overlap_fit'],
        'pcm_integer_measurements':{k:r['pcm_integer_measurements'] for k,r in manifest['samples'].items()},
        'waveform_display_interval_samples':[14800,15120],
        'plot_scope':'signed model coefficient and actual PCM of fixed, already correctly ordered source slots; two explicit uncompensated scoring windows; not a blind CSS or speaker-identity result',
        'limits':manifest['limits']}
    return manifest, report, signals


def main():
    manifest,report,signals=measurements()
    report_path=ROOT/'codes/chapters/ch08/reports/figure76_css_polarity_gain.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10.6,8.2),constrained_layout=True)
    index=np.arange(12800,19200);r=np.linspace(0,1,6400)
    for values,label,color in ((1-3*r,'不校正：1−3r',C_RED),(1+r,'仅翻极性：1+r',C_GREEN),(np.ones(6400),'完整校正：1',C_BLUE)):
        axes[0,0].plot(index/16000,values,color=color,label=label)
    axes[0,0].axhline(0,color='gray',linewidth=.8)
    axes[0,0].set(title='(a) 已知模型：重叠区的有符号幅度系数',xlabel='共同接收时轴 / s',ylabel='相对同槽参考的倍率',xlim=(.8,1.2),ylim=(-2.3,2.7))
    axes[0,0].legend(fontsize=10)
    begin,end=report['waveform_display_interval_samples']
    for key,label,color,style in (('reference','参考＝完整校正',C_BLUE,'-'),('naive','不校正',C_RED,'-'),('polarity','仅翻极性',C_GREEN,'--')):
        axes[0,1].plot(np.arange(begin,end)/16000,signals[key][0,begin:end],style,color=color,linewidth=1.1,label=label)
    axes[0,1].set(title='(b) 第一槽实际 PCM：经过相消位置',xlabel='共同接收时轴 / s',ylabel='PCM 解码幅度',ylim=(-.17,.19))
    axes[0,1].set_xticks([.925,.930,.935,.940,.945])
    axes[0,1].legend(fontsize=10,loc='upper right')
    for ax,window,title,ylim in ((axes[1,0],'primary','(c) 主评分窗：含 400 ms 渐变',(-.12,5.2)),(axes[1,1],'post_overlap','(d) 后评分窗：渐变结束之后',(-.2,10.8))):
        keys=('naive','polarity','corrected');values=[]
        for key in keys:
            stats=report['pcm_integer_measurements'][key][window]
            values.append(sum(stats['integer_reference_error_squared_sum_per_channel'])/sum(stats['integer_reference_squared_sum_per_channel']))
        ax.bar(range(3),values,color=[C_RED,C_GREEN,C_BLUE],width=.55,alpha=.7)
        for i,value in enumerate(values):ax.text(i,value+.1,f'{value:.4f}',ha='center',fontsize=11)
        interval=manifest['parameters']['scoring_intervals_samples'][window]
        ax.set(title=title,xlabel=f'半开采样窗 [{interval[0]}, {interval[1]})；合并两槽线性误差',ylabel='实际整数误差 E / 参考整数能量 D',xticks=range(3),xticklabels=['不校正','仅翻极性','完整校正'],ylim=ylim)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('槽位排列正确后，仍须检查极性与增益')
    finalize_figure(fig)
    metadata={'SourceScript':'scripts/make_css_figures.py','SourceScriptDigest':report['source_sha256']['scripts/make_css_figures.py'],
        'GeneratorInputs':json.dumps(report['source_sha256'],sort_keys=True),'AudioManifestDigest':report['audio_manifest_sha256'],
        'NumericalReportDigest':hashlib.sha256(report_path.read_bytes()).hexdigest()}
    destination=ROOT/'figures/fig76_css_polarity_gain.png'
    fig.savefig(destination,dpi=150,bbox_inches='tight',facecolor='white',metadata=metadata);plt.close(fig)
    print(destination.relative_to(ROOT))

if __name__=='__main__':main()
