"""Independent application semantics and honest scoring for the demo."""
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "query_teacher", Path(__file__).resolve().parents[2] / "scripts/run-query-teacher.py"
)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class QueryTeacherTests(unittest.TestCase):
    def test_rejects_code_and_extra_forms(self):
        for text in ['#.(delete-file "x")', "(issues :open :any :week) (quit)",
                     "(issues :open :any :year)"]:
            self.assertIsNone(demo.query_ids(text))

    def test_filters_and_age_boundaries(self):
        self.assertEqual(demo.query_ids("(issues :open :storage :month)"), [])
        self.assertEqual(demo.query_ids("(issues :closed :any :month)"), [2, 4])
        self.assertEqual(demo.query_ids("(issues :any :any :week)"), [1, 3])

    def test_same_results_are_not_same_command(self):
        # Both queries match issue 1 in this tiny database. Only one is correct.
        result = demo.metrics([{
            "split": "test", "held_combination": True,
            "target": "(issues :open :tokenizer :week)",
            "prediction": "(issues :any :tokenizer :week)",
        }])["test_held_combinations"]
        self.assertEqual(result, {"count": 1, "exact": 0, "valid": 1, "execution_match": 1})


if __name__ == "__main__":
    unittest.main()
