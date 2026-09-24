"""历史标题锚：章节重排后仍指向原来的教学主题。

键是旧版发布过的片段，值是当前标题片段。改版前的序号别名由固定
清单保存；语义编号和无编号标题摘要的少量变化也在本文件登记。
"""

import json
from pathlib import Path


# 改版前发布过的 sec-N 指向当时第 N 个标题。新标题增删后不能把
# 它机械地重新指向当前第 N 个标题，故固定旧版逐标题映射。
HISTORICAL_SEQUENTIAL_IDS = json.loads(
    Path(__file__).with_name("legacy_sequential_anchors.json").read_text(encoding="utf-8")
)

HISTORICAL_SECTION_IDS = {
    "04_doa-estimation.md": {
        "sec-u-442f837dd2": "sec-u-5d3aebe0fb",  # MUSIC 旧标题
        "sec-u-8dda6571c9": "sec-u-e9bdae6530",  # ESPRIT 旧标题
        "sec-4-6-1": "sec-u-08146bdaaf",  # 原 4.6.1 宽带聚焦
        "sec-4-7-1": "sec-u-00305faff5",  # 近场模型失配
        "sec-4-7-2": "sec-u-374bae76e9",  # 可执行基线与外部实现
    },
    "03_array-geometry.md": {
        "sec-u-5834a7d67e": "sec-3-3-1",
        "sec-u-20a0d88f74": "sec-3-3-2",
        "sec-u-d14b5899a5": "sec-3-3-3",
    },
    "06_aec.md": {
        "sec-6-1-3": "sec-6-2",
        **{f"sec-6-1-{index}": f"sec-6-{index - 1}"
           for index in range(4, 19)},
    },
    "07_wpe-dereverberation.md": {
        "sec-7-1-1": "sec-7-8",
        "sec-7-1-2": "sec-7-9",
        "sec-7-1-3": "sec-7-10",
    },
    "08_speech-separation.md": {
        **{f"sec-8-1-{index}": f"sec-8-{index + 2}"
           for index in range(1, 7)},
    },
    "09_source-tracking.md": {
        "sec-u-f7a201edb9": "sec-9-6",
    },
    "10_engineering-practice.md": {
        "sec-u-38edc89a29": "sec-u-f9e36547b8",  # 有状态重采样旧标题
    },
    "13_appendix-guide.md": {
        "sec-u-b74f82c81d": "sec-u-36ad23ae3f",
        "sec-13-3-1": "sec-u-36ad23ae3f",
    },
}


def historical_aliases(source_name: str, current_id: str) -> tuple[str, ...]:
    aliases = {}
    for mapping in (HISTORICAL_SECTION_IDS.get(source_name, {}),
                    HISTORICAL_SEQUENTIAL_IDS.get(source_name, {})):
        for old_id, target in mapping.items():
            if target == current_id:
                aliases[old_id] = None
    return tuple(aliases)


def has_historical_sequential_aliases(source_name: str) -> bool:
    return source_name in HISTORICAL_SEQUENTIAL_IDS
