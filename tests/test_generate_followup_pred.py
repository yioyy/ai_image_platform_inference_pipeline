import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from generate_followup_pred import generate_followup_pred


class TestGenerateFollowupPred(unittest.TestCase):
    def test_merge_rules(self):
        baseline = np.zeros((3, 3, 3), dtype=np.int16)
        followup = np.zeros((3, 3, 3), dtype=np.int16)

        # baseline-only -> 1 (new)
        baseline[0, 0, 0] = 10
        # overlap -> 2 (stable)
        baseline[1, 1, 1] = 1
        followup[1, 1, 1] = 2
        # followup-only -> 3 (disappeared)
        followup[2, 2, 2] = 9

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base_path = d / "baseline.nii.gz"
            foll_path = d / "followup_reg.nii.gz"
            out_path = d / "merged.nii.gz"

            affine = np.eye(4, dtype=float)
            nib.save(nib.Nifti1Image(baseline, affine), str(base_path))
            nib.save(nib.Nifti1Image(followup, affine), str(foll_path))

            generate_followup_pred(base_path, foll_path, out_path)
            out = np.asanyarray(nib.load(str(out_path)).dataobj).astype(np.uint8)

            self.assertEqual(int(out[0, 0, 0]), 1)
            self.assertEqual(int(out[1, 1, 1]), 2)
            self.assertEqual(int(out[2, 2, 2]), 3)

    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            base_path = d / "baseline.nii.gz"
            foll_path = d / "followup_reg.nii.gz"
            out_path = d / "merged.nii.gz"

            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)), str(base_path))
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.int16), np.eye(4)), str(foll_path))

            with self.assertRaises(ValueError):
                generate_followup_pred(base_path, foll_path, out_path)


if __name__ == "__main__":
    unittest.main()


