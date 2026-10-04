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
    "00_overview.md": {
        "sec-u-726cf32300": "sec-u-e445ebf98d",  # 72→73张图，仍是同一插图地图
        "sec-u-dacd520d2c": "sec-u-726cf32300",  # 明示编号图，仍是同一插图地图
        "sec-u-2afd7c6509": "sec-u-461c9549d0",  # 15文件→16文件，同一文件地图
        "sec-u-9c77d218fe": "sec-u-dacd520d2c",  # 69图→72图，同一插图地图
        "sec-u-06e6cf7b78": "sec-u-2afd7c6509",  # 14文件→15文件，原文件地图
        "sec-u-ec3f837100": "sec-u-9c77d218fe",  # 66图→69图，仍是同一插图地图
        "sec-u-8657cda209": "sec-u-ec3f837100",  # 65 figure map keeps its original topic
        "sec-u-0d7a1562fa": "sec-u-ec3f837100",  # 64 张历史标题
        "sec-u-8a5dd76e40": "sec-u-ec3f837100",  # 63 张历史标题
        "sec-u-f19f47c7c1": "sec-u-ec3f837100",  # 61 张历史标题
        "sec-u-2b17484f3b": "sec-u-ec3f837100",  # 60 张历史标题
        "sec-u-cb1ede6c60": "sec-u-ec3f837100",  # 59 张历史标题
        "sec-u-5edae53c5f": "sec-u-ec3f837100",  # 58 张历史标题
        "sec-u-5c63f4d83e": "sec-u-ec3f837100",  # 56 张历史标题
        "sec-u-36fa20efde": "sec-u-ec3f837100",  # 54 张历史标题
        "sec-u-3ebc13ca5a": "sec-u-ec3f837100",  # 52 张历史标题
        "sec-u-1e72416790": "sec-u-ec3f837100",  # 50 张历史标题
        "sec-u-3a0278b879": "sec-u-ec3f837100",  # 49 张历史标题
        "sec-u-b7a71b077a": "sec-u-3937b1b94e",  # 路径 A：去掉无依据的周数
        "sec-u-efc5552983": "sec-u-f958564d39",  # 路径 B：去掉无依据的周数
        "sec-u-6ef8e18126": "sec-u-ec3f837100",  # 插图地图：40 张历史标题
        "sec-u-d538d6d0a5": "sec-u-ec3f837100",  # 41 张历史标题
        "sec-u-0943a9ed3c": "sec-u-ec3f837100",  # 43 张历史标题
        "sec-u-1e5d8a2bad": "sec-u-ec3f837100",  # 44 张历史标题
        "sec-u-d8fd1002de": "sec-u-ec3f837100",  # 45 张历史标题
        "sec-u-627ed3907e": "sec-u-ec3f837100",  # 47 张历史标题
    },
    "01_problem-definition.md": {
        "sec-u-81896116fa": "sec-u-5e074a7e92",  # 理想收益标题，原任务与收益主题
        "sec-u-05ec932a34": "sec-u-c36b7ac16e",  # 客厅里的远场语音问题：标题改为远距离
    },
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
    "11_selection-guide.md": {
        "sec-u-65c140a10b": "sec-u-1cf1ad108e",  # E11-01旧题4标题
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
            # The immutable original sequential map may name a subsequently
            # renamed semantic anchor. Follow that rename without rewriting
            # the old map or assigning its number to a different topic.
            renames = HISTORICAL_SECTION_IDS.get(source_name, {})
            visited = set()
            while target in renames:
                if target in visited:
                    raise ValueError(f"历史标题别名循环：{source_name}:{target}")
                visited.add(target)
                target = renames[target]
            if target == current_id:
                aliases[old_id] = None
    return tuple(aliases)


def has_historical_sequential_aliases(source_name: str) -> bool:
    return source_name in HISTORICAL_SEQUENTIAL_IDS
