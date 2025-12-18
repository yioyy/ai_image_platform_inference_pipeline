import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import registration


class TestRegistrationResampleMin1mm(unittest.TestCase):
    def test_only_submm_axes_are_resampled_to_1mm(self):
        data = np.zeros((4, 4, 2), dtype=np.int16)
        aff = np.diag([0.5, 2.0, 0.8, 1.0])  # x<1, z<1, y>=1

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "src.nii.gz"
            dst = d / "dst.nii.gz"
            nib.save(nib.Nifti1Image(data, aff), str(src))

            registration._resample_label_min_1mm_per_axis(src=src, dst=dst, threshold_mm=1.0)

            out = nib.load(str(dst))
            zooms = nib.affines.voxel_sizes(out.affine)[:3]
            self.assertTrue(np.allclose(zooms, [1.0, 2.0, 1.0], atol=1e-6))

            # new_shape = ceil(old_shape * old_zoom / new_zoom)
            expected_shape = (
                int(np.ceil(4 * 0.5 / 1.0)),
                int(np.ceil(4 * 2.0 / 2.0)),
                int(np.ceil(2 * 0.8 / 1.0)),
            )
            self.assertEqual(out.shape, expected_shape)

            # NOTE:
            # conform 會建立一致的幾何 header（affine/zooms），但不保證「world center 完全不動」；
            # 我們在 pipeline 內的策略是：確保同一流程的影像使用一致的 header/grid，
            # 以避免 overlap 計算在 voxel index 上失效。


if __name__ == "__main__":
    unittest.main()


