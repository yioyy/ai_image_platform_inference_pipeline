import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import followup_report


class TestFollowupReportPreserveUid(unittest.TestCase):
    def _mk_sorted(self, items):
        # items: list[tuple[str, float]]
        return [followup_report.SortedSlice(sop_instance_uid=uid, projection=proj) for uid, proj in items]

    def test_matched_lesion_preserves_previous_followup_dicom_uid(self):
        baseline_instances = {
            1: {
                "mask_index": "1",
                "diameter": "10",
                "dicom_sop_instance_uid": "b2",
                "followup_dicom_sop_instance_uid": "prev_f2",
            }
        }
        followup_instances = {
            2: {
                "mask_index": "2",
                "diameter": "10",
                "dicom_sop_instance_uid": "new_f2",
            }
        }
        baseline_to_followup = {1: 2}

        followup_report._update_baseline_instances_with_followup(
            baseline_payload={},
            baseline_instances=baseline_instances,
            followup_instances=followup_instances,
            baseline_to_followup=baseline_to_followup,
            baseline_sorted=self._mk_sorted([("b1", 0.0), ("b2", 1.0), ("b3", 2.0)]),
            followup_sorted=self._mk_sorted([("f1", 0.0), ("f2", 1.0), ("f3", 2.0)]),
            mat=np.eye(4, dtype=np.float64),
            transform_direction="F2B",
        )

        self.assertEqual(
            baseline_instances[1]["followup_dicom_sop_instance_uid"],
            "prev_f2",
        )

    def test_matched_lesion_sets_followup_dicom_uid_if_missing(self):
        baseline_instances = {
            1: {
                "mask_index": "1",
                "diameter": "10",
                "dicom_sop_instance_uid": "b2",
                "followup_dicom_sop_instance_uid": "",
            }
        }
        followup_instances = {
            2: {
                "mask_index": "2",
                "diameter": "10",
                "dicom_sop_instance_uid": "f2_from_followup_json",
            }
        }
        baseline_to_followup = {1: 2}

        followup_report._update_baseline_instances_with_followup(
            baseline_payload={},
            baseline_instances=baseline_instances,
            followup_instances=followup_instances,
            baseline_to_followup=baseline_to_followup,
            baseline_sorted=self._mk_sorted([("b1", 0.0), ("b2", 1.0), ("b3", 2.0)]),
            followup_sorted=self._mk_sorted([("f1", 0.0), ("f2", 1.0), ("f3", 2.0)]),
            mat=np.eye(4, dtype=np.float64),
            transform_direction="F2B",
        )

        self.assertEqual(
            baseline_instances[1]["followup_dicom_sop_instance_uid"],
            "f2_from_followup_json",
        )

    def test_new_status_recomputes_followup_dicom_uid_by_matrix(self):
        baseline_instances = {
            1: {
                "mask_index": "1",
                "diameter": "10",
                "dicom_sop_instance_uid": "b2",
                "followup_dicom_sop_instance_uid": "prev_should_be_overwritten_for_new",
            }
        }
        followup_instances = {}
        baseline_to_followup = {1: None}

        followup_report._update_baseline_instances_with_followup(
            baseline_payload={},
            baseline_instances=baseline_instances,
            followup_instances=followup_instances,
            baseline_to_followup=baseline_to_followup,
            baseline_sorted=self._mk_sorted([("b1", 0.0), ("b2", 1.0), ("b3", 2.0)]),
            followup_sorted=self._mk_sorted([("f1", 0.0), ("f2", 1.0), ("f3", 2.0)]),
            mat=np.eye(4, dtype=np.float64),
            transform_direction="F2B",
        )

        # identity mapping keeps projection (proj=1.0 => nearest proj=1.0)
        self.assertEqual(baseline_instances[1]["status"], "new")
        self.assertEqual(baseline_instances[1]["followup_dicom_sop_instance_uid"], "f2")

    def test_disappeared_appended_entry_maps_dicom_uid_by_matrix(self):
        baseline_payload = {"mask": {"series": [{"instances": []}]}}
        followup_payload = {}
        followup_instances = {
            9: {
                "mask_index": "9",
                "diameter": "10",
                "dicom_sop_instance_uid": "f3",
                "type": "test",
                "location": "test",
            }
        }

        followup_report._append_disappeared_followup_instances(
            baseline_payload=baseline_payload,
            followup_payload=followup_payload,
            followup_instances=followup_instances,
            used_followup_labels=set(),
            baseline_sorted=self._mk_sorted([("b1", 0.0), ("b2", 1.0), ("b3", 2.0)]),
            followup_sorted=self._mk_sorted([("f1", 0.0), ("f2", 1.0), ("f3", 2.0)]),
            mat=np.eye(4, dtype=np.float64),
            transform_direction="F2B",
        )

        instances = baseline_payload["mask"]["series"][0]["instances"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["status"], "disappeared")
        self.assertEqual(instances[0]["followup_dicom_sop_instance_uid"], "f3")
        self.assertEqual(instances[0]["dicom_sop_instance_uid"], "b3")


if __name__ == "__main__":
    unittest.main()


