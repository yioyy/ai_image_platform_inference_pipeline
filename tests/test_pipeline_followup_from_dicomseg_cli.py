import unittest

import pipeline_followup_from_dicomseg


class TestPipelineFollowupFromDicomsegCli(unittest.TestCase):
    def test_parser_requires_core_args(self):
        # Should exit with SystemExit due to missing required args
        with self.assertRaises(SystemExit):
            pipeline_followup_from_dicomseg.main([])


if __name__ == "__main__":
    unittest.main()

