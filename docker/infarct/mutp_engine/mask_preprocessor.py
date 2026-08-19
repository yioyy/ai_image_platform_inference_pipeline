"""Mask-channel-aware preprocessor for mutp mask-fusion architectures
(DeepConcat / SPADE family).

Motivation:
  nnUNet DefaultPreprocessor 對輸入影像所有 channel 都用 cubic (order=3) resample，
  將整數 mask (vessel8: 0..8 / SynthSeg10: 0..9) 弄成浮點 (1.18, 3.6, 7.99...)。
  雖然 network._onehot_mask.round().clamp() 有救，但邊界會 shift，跟 nnUNet 對
  GT-label 的處理不一致（GT-label 用 resize_segmentation with order=1 → linear-onehot）。

Fix:
  Wrapper 讓最後一個 channel（mask）走 is_seg=True, order=1（等同 GT-label），
  前面 image channels 保持 cubic (order=3)。

Applies to training preprocessing (this file) AND inference (custom_predict.py
already does the same via dynamic class-swap).
"""
from __future__ import annotations

from typing import List, Union
import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager, ConfigurationManager,
)


# Module-level state: set before preprocess to activate mask-aware behavior.
# multi-process workers inherit via fork on Linux.
_MASK_IMAGE_CHANNELS: int | None = None


def set_mask_image_channels(n: int | None) -> None:
    """設定 mask-aware preprocessor 的 image_channels。

    Args:
        n: 前 n 個 channel 是 image (cubic resample)，第 n 個 (最後 1 個) 是 mask (nearest-linear)。
           None = 關閉 mask-aware，走標準 cubic。
    """
    global _MASK_IMAGE_CHANNELS
    _MASK_IMAGE_CHANNELS = n


def get_mask_image_channels() -> int | None:
    return _MASK_IMAGE_CHANNELS


def _per_channel_resample_factory(image_channels: int):
    """建立 per-channel resample fn：image channels cubic + mask channel linear-onehot。"""
    def _fn(data, new_shape, current_spacing, new_spacing):
        _img = data[:image_channels]
        _mask = data[image_channels:image_channels + 1]
        _img_r = resample_data_or_seg_to_shape(
            _img, new_shape, current_spacing, new_spacing,
            is_seg=False, order=3, order_z=0, force_separate_z=None,
        )
        _mask_r = resample_data_or_seg_to_shape(
            _mask, new_shape, current_spacing, new_spacing,
            is_seg=True, order=1, order_z=0, force_separate_z=None,
        )
        return np.concatenate([_img_r, _mask_r], axis=0)
    return _fn


def _make_mask_aware_configuration_manager_class(base_class, image_channels: int):
    """動態建 subclass，覆蓋 resampling_fn_data property 為 per-channel 版。"""
    per_ch_fn = _per_channel_resample_factory(image_channels)
    return type(
        'MaskAware' + base_class.__name__,
        (base_class,),
        {'resampling_fn_data': property(lambda _self: per_ch_fn)},
    )


class MaskAwareDefaultPreprocessor(DefaultPreprocessor):
    """DefaultPreprocessor + per-channel resample for mask-fusion架構.

    行為：
    - 若 module-level `_MASK_IMAGE_CHANNELS` 有值 → 動態 swap configuration_manager 的 class
      讓 resampling_fn_data 變 per-channel 版。
    - 否則 → 完全等同 DefaultPreprocessor。
    """

    def run_case(self, image_files: List[str], seg_file: Union[str, None],
                 plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str],
                 normal_file: Union[str, None] = None,
                 dilate_file: Union[str, None] = None):
        image_channels = _MASK_IMAGE_CHANNELS
        if image_channels is not None:
            _cm_class = type(configuration_manager)
            # 只 swap 一次；避免重複套（多次 subclass 巢狀）
            if not getattr(_cm_class, '__mutp_mask_aware__', False):
                _new_class = _make_mask_aware_configuration_manager_class(_cm_class, image_channels)
                setattr(_new_class, '__mutp_mask_aware__', True)
                configuration_manager.__class__ = _new_class
        return super().run_case(
            image_files, seg_file, plans_manager, configuration_manager, dataset_json,
            normal_file=normal_file, dilate_file=dilate_file,
        )
