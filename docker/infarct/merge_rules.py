"""把 SynthSeg 的細分區（aparc+aseg，FreeSurfer 編碼）重組成任務專用的粗類。

設計原則：**永遠保留最細的 aparc+aseg 當唯一真相來源**，粗類一律是它的衍生物。
合併過的 mask 回不去 —— 想換分群規則時，只要改這裡的規則再重跑一次 merge，
不用重跑 SynthSeg 推論（那才是貴的部分）。

    from mutp.preprocessing.synthseg import apply_merge_rule, MERGE_RULES

    merged = apply_merge_rule(aparc_aseg_array, "infarct10")   # uint8 0..9

新增規則：在 ``MERGE_RULES`` 加一筆 :class:`MergeRule` 即可，
``groups`` 依序套用（後者覆寫前者），所以要把大集合放前面、特例放後面。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# ============================================================
# FreeSurfer / Desikan-Killiany 編碼小工具
# ============================================================


def _dk(*base: int) -> list[int]:
    """Desikan-Killiany 皮質分區 base 編號 → 左右兩側 (1xxx / 2xxx)。"""
    return [1000 + b for b in base] + [2000 + b for b in base]


#: aparc 皮質分區的編碼範圍（FreeSurfer 慣例：1000-1035 左、2000-2035 右）
CORTEX_RANGE = (1000, 3000)

#: ctx-corpuscallosum 掛在皮質編碼下，但解剖上是白質（FreeSurfer 歷史命名遺留）
CC_AS_WM = _dk(4)


# ============================================================
# 規則資料結構
# ============================================================


@dataclass(frozen=True)
class MergeGroup:
    """一個輸出類別。``labels`` 為空且 ``cortex_catchall=True`` 時代表「所有皮質」。"""

    index: int
    name: str
    labels: tuple[int, ...] = ()
    cortex_catchall: bool = False
    note: str = ""


@dataclass(frozen=True)
class MergeRule:
    name: str
    description: str
    source: str  # 吃哪種輸入：aparc_aseg / aseg33
    groups: tuple[MergeGroup, ...] = field(default=())

    @property
    def n_classes(self) -> int:
        return max(g.index for g in self.groups) + 1

    def class_names(self) -> dict[str, str]:
        """給 ``--mask-classes`` / dataset.json 用的 {"0": "bg", ...}。"""
        return {str(g.index): g.name for g in sorted(self.groups, key=lambda g: g.index)}

    def values(self) -> list[int]:
        """給 ``mask_values`` 用的非背景類別值。"""
        return [g.index for g in sorted(self.groups, key=lambda g: g.index) if g.index != 0]


# ============================================================
# infarct 10 類 —— DWI 梗塞模型的解剖 prior channel
# ============================================================
#
# 規則來源：infarct/SynthSeg_merged_classes.md
#
# ⚠ 已對實際產物驗證過：infarct/SynthSeg_merged/ 的 273 個 mask 用的是
#   class 3 只含 **外側顳葉 4 個分區**（9/15/30/34）的版本。
#   舊腳本 infarct/code/merge_synthseg.py 裡的 TEMPORAL_DK_BASE 還留著
#   iter-2 的 8 個分區（多了 6/7/16/33），與實際產物**不符**，不要照抄。
#   驗證方式：比對 case 00016467_20180516 的 class3/皮質總量比例 —
#   4-label 版 0.1375 vs merged 0.1346（Δ=0.003）；8-label 版 0.2143（Δ=0.080）。
#
INFARCT10 = MergeRule(
    name="infarct10",
    description="DWI 梗塞模型的解剖 prior：切出 3 個 FP 警戒區（腦室 CP、乳突、額竇）",
    source="aparc_aseg",
    groups=(
        # 先鋪底：所有皮質 → class 5，之後由 FP 警戒區覆寫
        MergeGroup(5, "cortical_GM_other", cortex_catchall=True,
                   note="含下內側顳葉 fusiform/entorhinal/parahippocampal/temporalpole —— "
                        "baseline 實證這區 artifact 不顯著，不另外切出來"),
        MergeGroup(7, "cerebral_WM", labels=(2, 41, *CC_AS_WM),
                   note="corpuscallosum 從皮質編碼撥回白質"),
        # FP 警戒區（覆寫 class 5）
        MergeGroup(3, "FP_lateral_temporal", labels=tuple(_dk(9, 15, 30, 34)),
                   note="乳突氣房 air-bone 介面 → EPI susceptibility"),
        MergeGroup(4, "FP_frontal_inferior", labels=tuple(_dk(12, 14, 19, 32)),
                   note="額竇 + 篩竇 + 眶骨上緣 → EPI susceptibility"),
        # 以下 aseg 編碼與皮質範圍互斥，順序無所謂
        MergeGroup(1, "CSF_ventricle", labels=(4, 5, 14, 15, 43, 44),
                   note="脈絡叢 (CP) 只長在腦室內 —— CP FP 警戒區"),
        MergeGroup(2, "CSF_subarachnoid", labels=(24,),
                   note="腦溝/腦池，沒有 CP"),
        MergeGroup(6, "deep_GM", labels=(10, 11, 12, 13, 17, 18, 26, 28,
                                         49, 50, 51, 52, 53, 54, 58, 60),
                   note="thalamus/caudate/putamen/pallidum/hippo/amygdala/accumbens/VentralDC"),
        MergeGroup(8, "brainstem", labels=(16,),
                   note="aseg 16 = Brain-Stem，別跟 15 (4th ventricle) 搞混"),
        MergeGroup(9, "cerebellum", labels=(7, 8, 46, 47),
                   note="aseg 7/8 = Left-Cerebellum-WM/Cortex，別跟 DK 7 (fusiform) 搞混"),
        MergeGroup(0, "background", labels=(0,), note="腦外 + 未分類"),
    ),
)


MERGE_RULES: dict[str, MergeRule] = {r.name: r for r in (INFARCT10,)}


# ============================================================
# 套用
# ============================================================


def apply_merge_rule(seg: np.ndarray, rule: str | MergeRule) -> np.ndarray:
    """aparc+aseg 陣列 → 依 ``rule`` 合併後的 uint8 label map。

    ``groups`` 依宣告順序套用，後面的覆寫前面的 —— 所以 cortex catch-all
    要放在 FP 警戒區前面，否則警戒區會被蓋掉。
    """
    if isinstance(rule, str):
        try:
            rule = MERGE_RULES[rule]
        except KeyError:
            raise KeyError(f"未知的 merge rule '{rule}'，可用：{sorted(MERGE_RULES)}") from None

    a = np.asarray(seg).astype(np.int32)
    out = np.zeros(a.shape, dtype=np.uint8)
    lo, hi = CORTEX_RANGE
    for g in rule.groups:
        if g.cortex_catchall:
            out[(a >= lo) & (a < hi)] = g.index
        if g.labels:
            out[np.isin(a, np.asarray(g.labels, dtype=np.int32))] = g.index
    return out


def unmapped_labels(seg: np.ndarray, rule: str | MergeRule) -> dict[int, int]:
    """回傳 ``seg`` 裡沒被 ``rule`` 涵蓋到（會落成 background）的 label → 體素數。

    換規則或換 SynthSeg 版本時拿來自我檢查：非 0 的結果代表有分區被默默丟掉。
    """
    if isinstance(rule, str):
        rule = MERGE_RULES[rule]
    a = np.asarray(seg).astype(np.int32)
    lo, hi = CORTEX_RANGE
    covered: set[int] = set()
    has_catchall = any(g.cortex_catchall for g in rule.groups)
    for g in rule.groups:
        covered |= set(g.labels)
    vals, counts = np.unique(a, return_counts=True)
    out: dict[int, int] = {}
    for v, c in zip(vals.tolist(), counts.tolist()):
        if v in covered or v == 0:
            continue
        if has_catchall and lo <= v < hi:
            continue
        out[int(v)] = int(c)
    return out


def merge_files(
    in_paths: Iterable[str],
    out_dir: str,
    rule: str | MergeRule = "infarct10",
    suffix: str = "_merged.nii.gz",
    check_unmapped: bool = True,
) -> list[dict]:
    """批次把 aparc+aseg 檔案合併輸出。回傳每個 case 的 dict（含 unmapped 統計）。"""
    import pathlib

    import nibabel as nib

    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for p in in_paths:
        src = pathlib.Path(p)
        img = nib.load(str(src))
        seg = np.asarray(img.dataobj)
        merged = apply_merge_rule(seg, rule)
        stem = src.name.replace(".nii.gz", "").replace(".nii", "")
        dst = od / f"{stem}{suffix}"
        nib.save(nib.Nifti1Image(merged, img.affine, img.header), str(dst))
        rec = {"input": str(src), "output": str(dst)}
        if check_unmapped:
            rec["unmapped"] = unmapped_labels(seg, rule)
        results.append(rec)
    return results
