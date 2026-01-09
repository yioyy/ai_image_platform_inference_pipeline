import unittest

import numpy as np

from code_ai.pipeline.dicomseg.seg_to_nifti import (
    SegFrame,
    SeriesSlice,
    assemble_label_volume,
)


class TestSegToNiftiAssembleVolume(unittest.TestCase):
    def test_skip_empty_slices_fill_zero(self):
        sorted_slices = [
            SeriesSlice("sop0", np.array([0.0, 0.0, 0.0]), np.zeros(6), 0.0),
            SeriesSlice("sop1", np.array([0.0, 0.0, 1.0]), np.zeros(6), 1.0),
            SeriesSlice("sop2", np.array([0.0, 0.0, 2.0]), np.zeros(6), 2.0),
        ]
        rows, cols = 2, 3
        # only one slice appears in SEG frames
        frames = [
            SegFrame("sop1", 7, np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)),
        ]
        vol = assemble_label_volume(sorted_slices=sorted_slices, rows=rows, cols=cols, frames=frames, overlap_mode="max")
        self.assertEqual(vol.shape, (rows, cols, 3))
        self.assertTrue(np.all(vol[:, :, 0] == 0))
        self.assertTrue(np.all(vol[:, :, 2] == 0))
        self.assertEqual(int(vol[0, 0, 1]), 7)
        self.assertEqual(int(vol[1, 2, 1]), 7)

    def test_multi_label_max_overlap(self):
        sorted_slices = [SeriesSlice("sop0", np.array([0.0, 0.0, 0.0]), np.zeros(6), 0.0)]
        rows, cols = 2, 2
        frames = [
            SegFrame("sop0", 2, np.array([[1, 0], [0, 1]], dtype=np.uint8)),
            SegFrame("sop0", 5, np.array([[1, 1], [0, 0]], dtype=np.uint8)),
        ]
        vol = assemble_label_volume(sorted_slices=sorted_slices, rows=rows, cols=cols, frames=frames, overlap_mode="max")
        # overlap at (0,0): max(2,5)=5
        self.assertEqual(int(vol[0, 0, 0]), 5)
        # (0,1) only seg 5
        self.assertEqual(int(vol[0, 1, 0]), 5)
        # (1,1) only seg 2
        self.assertEqual(int(vol[1, 1, 0]), 2)


if __name__ == "__main__":
    unittest.main()

