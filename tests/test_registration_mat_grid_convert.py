import unittest

import numpy as np

import registration


class TestRegistrationMatGridConvert(unittest.TestCase):
    def test_identity_stays_identity_between_grids(self):
        # from grid: 1mm isotropic
        a_in_from = np.diag([1.0, 1.0, 1.0, 1.0])
        a_ref_from = np.diag([1.0, 1.0, 1.0, 1.0])

        # to grid: anisotropic (e.g., 0.5x0.5x2.0)
        a_in_to = np.diag([0.5, 0.5, 2.0, 1.0])
        a_ref_to = np.diag([0.5, 0.5, 2.0, 1.0])

        m_from = np.eye(4, dtype=np.float64)
        m_to = registration._convert_flirt_mat_between_grids(
            mat_vox=m_from,
            in_affine_from=a_in_from,
            ref_affine_from=a_ref_from,
            in_affine_to=a_in_to,
            ref_affine_to=a_ref_to,
        )

        self.assertTrue(np.allclose(m_to, np.eye(4), atol=1e-10))

    def test_pure_translation_in_mm_roundtrips(self):
        # Same grid sizes for simplicity; translation in voxel space differs when voxel sizes differ.
        a_in_from = np.diag([1.0, 1.0, 1.0, 1.0])
        a_ref_from = np.diag([1.0, 1.0, 1.0, 1.0])

        a_in_to = np.diag([0.5, 0.5, 2.0, 1.0])
        a_ref_to = np.diag([0.5, 0.5, 2.0, 1.0])

        # In 1mm grid, translate +10mm in Z => +10 vox in Z (since 1mm)
        m_from = np.eye(4, dtype=np.float64)
        m_from[2, 3] = 10.0

        m_to = registration._convert_flirt_mat_between_grids(
            mat_vox=m_from,
            in_affine_from=a_in_from,
            ref_affine_from=a_ref_from,
            in_affine_to=a_in_to,
            ref_affine_to=a_ref_to,
        )

        # In "to" grid, Z voxel is 2mm, so +10mm => +5 vox.
        expected = np.eye(4, dtype=np.float64)
        expected[2, 3] = 5.0
        self.assertTrue(np.allclose(m_to, expected, atol=1e-10))

    def test_predgrid_mat_preserves_world_transform(self):
        # Simulate: synthseg grid and pred grid share same orientation but different translations.
        a_base_ss = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        a_foll_ss = np.array(
            [
                [1.0, 0.0, 0.0, -5.0],
                [0.0, 1.0, 0.0, 7.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        a_base_pred = a_base_ss.copy()
        a_base_pred[:3, 3] += np.array([2.0, -3.0, 4.0], dtype=np.float64)
        a_foll_pred = a_foll_ss.copy()
        a_foll_pred[:3, 3] += np.array([-1.0, 5.0, -2.0], dtype=np.float64)

        # synthseg voxel mat: translate +10 in X voxel
        m_ss = np.eye(4, dtype=np.float64)
        m_ss[0, 3] = 10.0

        t_mm = a_base_ss @ m_ss @ np.linalg.inv(a_foll_ss)
        m_pred = np.linalg.inv(a_base_pred) @ t_mm @ a_foll_pred

        v = np.array([1.2, 3.4, 5.6, 1.0], dtype=np.float64)
        w_from_ss = t_mm @ (a_foll_pred @ v)  # treat v as followup_pred voxel -> mm, then world transform
        w_from_pred = a_base_pred @ (m_pred @ v)
        self.assertTrue(np.allclose(w_from_ss, w_from_pred, atol=1e-8))


if __name__ == "__main__":
    unittest.main()


