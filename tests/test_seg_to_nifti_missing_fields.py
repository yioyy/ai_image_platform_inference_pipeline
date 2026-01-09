import unittest

from code_ai.pipeline.dicomseg.seg_to_nifti import MissingDicomHeaderFieldsError, build_series_geometry_from_headers


class TestSegToNiftiMissingFields(unittest.TestCase):
    def test_missing_fields_are_listed(self):
        headers = [
            {
                # SOPInstanceUID missing
                "ImagePositionPatient": [0, 0, 0],
                "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                "PixelSpacing": [1.0, 1.0],
                "Rows": 2,
                "Columns": 3,
            }
        ]
        with self.assertRaises(MissingDicomHeaderFieldsError) as ctx:
            build_series_geometry_from_headers(headers)
        self.assertIn("SOPInstanceUID", ctx.exception.missing_fields)


if __name__ == "__main__":
    unittest.main()

