import unittest

import numpy as np

from code_ai.pipeline.dicomseg.seg_to_nifti import build_series_geometry_from_headers


class TestSegToNiftiGeometry(unittest.TestCase):
    def test_sorting_and_affine_shape(self):
        # Simple axial-ish: row_dir=[1,0,0], col_dir=[0,1,0], normal=[0,0,1]
        headers = [
            {
                "SOPInstanceUID": "sop2",
                "ImagePositionPatient": [0, 0, 2],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                "PixelSpacing": [2.0, 3.0],
                "Rows": 4,
                "Columns": 5,
            },
            {
                "SOPInstanceUID": "sop0",
                "ImagePositionPatient": [0, 0, 0],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                "PixelSpacing": [2.0, 3.0],
                "Rows": 4,
                "Columns": 5,
            },
            {
                "SOPInstanceUID": "sop1",
                "ImagePositionPatient": [0, 0, 1],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                "PixelSpacing": [2.0, 3.0],
                "Rows": 4,
                "Columns": 5,
            },
        ]
        slices, affine_ras, rows, cols = build_series_geometry_from_headers(headers)
        self.assertEqual((rows, cols), (4, 5))
        self.assertEqual([s.sop_instance_uid for s in slices], ["sop0", "sop1", "sop2"])
        self.assertEqual(affine_ras.shape, (4, 4))
        self.assertTrue(np.isfinite(affine_ras).all())


if __name__ == "__main__":
    unittest.main()

