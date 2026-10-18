"""Offline report requires independent human labels and clean controls."""
import tempfile
import unittest
from pathlib import Path

from evaluate_small_defects import load_cases, metrics, report


class SensitivityStudyTests(unittest.TestCase):
    def load(self, content):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"labels.csv"
            path.write_text(content,encoding="utf-8")
            return load_cases(path)

    def test_metrics_do_not_confuse_false_alarms_with_defects(self):
        cases=self.load(
            "sample_id,position,actual,score\n"
            "positive,media-centro,DEFECTO,54.00\n"
            "clean,control,BUENA,52.00\n"
        )
        strict=metrics(cases,55.20)
        self.assertEqual((strict["tp"],strict["fn"],strict["fp"],strict["tn"]),
                         (0,1,0,1))
        loose=metrics(cases,51.00)
        self.assertEqual((loose["tp"],loose["fp"]),(1,1))
        self.assertEqual(loose["specificity"],0.)

    def test_missing_labels_do_not_become_normal_by_default(self):
        cases=self.load(
            "sample_id,position,actual,score\n"
            "unknown,media-centro,,47.00\n"
            "clear,control,BUENA,23.00\n"
        )
        self.assertEqual(len(cases),1)
        text=report(cases)
        self.assertIn("INCOMPLETO",text)
        self.assertIn("DEFECTO=0",text)

    def test_two_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicados"):
            self.load(
                "sample_id,position,actual,score\n"
                "dup,media-centro,DEFECTO,55.0\n"
                "dup,media-centro,BUENA,24.0\n"
            )

    def test_positions_and_score_limits_validated(self):
        with self.assertRaises(ValueError):
            self.load(
                "sample_id,position,actual,score\n"
                "a,invalid,DEFECTO,45.0\n"
            )
        with self.assertRaises(ValueError):
            self.load(
                "sample_id,position,actual,score\n"
                "a,media-centro,DEFECTO,115.0\n"
            )


if __name__=="__main__":
    unittest.main()
