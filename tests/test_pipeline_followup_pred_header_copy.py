import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import pipeline_followup


class TestPipelineFollowupPredHeaderCopy(unittest.TestCase):
    def test_overwrite_pred_geometry_from_synthseg_copies_affine(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pred = d / "Pred_CMB.nii.gz"
            syn = d / "SynthSEG_CMB.nii.gz"

            data = np.zeros((5, 6, 7), dtype=np.int16)
            pred_aff = np.diag([1.0, 1.0, 1.0, 1.0])
            syn_aff = np.array(
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, -20.0],
                    [0.0, 0.0, 1.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            nib.save(nib.Nifti1Image(data, pred_aff), str(pred))
            nib.save(nib.Nifti1Image(np.zeros_like(data), syn_aff), str(syn))

            pipeline_followup._overwrite_pred_geometry_from_synthseg(pred_file=pred, synthseg_file=syn)

            out = nib.load(str(pred))
            self.assertTrue(np.allclose(out.affine, syn_aff, atol=1e-6))


if __name__ == "__main__":
    unittest.main()


