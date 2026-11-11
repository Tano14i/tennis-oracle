"""Prevent build regressions in the Docker and HF deployment package."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_docker_copies_all_local_python_imports(self):
        docker = (ROOT / "Dockerfile").read_text()
        # Source modules needed by the API must exist in the runtime image.
        required = (
            "tennis_api.py",
            "tennis_dataset.py", "tennis_models.py",
            "tennis_odds.py", "tennis_schedule.py", "tennis_analysis.py",
            "tennis_model_gate.py", "tennis_daily_picks.py",
            "tennis_daily_tracker.py", "tennis_temporal.py",
        )
        # COPY modules can span whitespace, but not package names.
        copy_lines = "\n".join(line for line in docker.splitlines()
                               if line.lstrip().startswith("COPY "))
        for name in required:
            with self.subTest(name=name):
                self.assertIn(name, copy_lines)

    def test_hf_workflow_upload_includes_imported_sources(self):
        flow = (ROOT / ".github/workflows/update_models.yml").read_text()
        section = flow.split("# Aggiungi solo i file di deployment", 1)[-1]
        for name in (
            "tennis_temporal.py", "tennis_model_gate.py",
            "tennis_daily_picks.py", "tennis_daily_tracker.py",
        ):
            with self.subTest(name=name):
                self.assertIn(name, section)


if __name__ == "__main__":
    unittest.main()
