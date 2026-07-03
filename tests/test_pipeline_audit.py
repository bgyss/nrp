import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.audit_pipeline_claims import audit_document, extract_out_paths  # noqa: E402


class PipelineAuditTests(unittest.TestCase):
    def test_extract_out_paths_deduplicates_and_strips_punctuation(self):
        text = "`out/foo/report.json`, and (out/foo/report.json). See out/bar/a-b.json."
        self.assertEqual(
            extract_out_paths(text),
            ["out/foo/report.json", "out/bar/a-b.json"],
        )

    def test_audit_reports_missing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "out" / "ok").mkdir(parents=True)
            (root / "out" / "ok" / "report.json").write_text("{}")
            doc = root / "docs" / "pipeline-feasibility.md"
            doc.write_text("Uses out/ok/report.json and out/missing/report.json.")
            report = audit_document(doc, root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing_out_paths"], ["out/missing/report.json"])


if __name__ == "__main__":
    unittest.main()
