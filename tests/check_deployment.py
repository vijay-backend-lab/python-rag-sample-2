"""Deployment checks: run after build_zip.py with python tests/check_deployment.py."""
from pathlib import Path
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


class DeploymentTests(unittest.TestCase):
    def test_deployment_zip(self):
        with ZipFile(ROOT / "dist/lambda-query-api.zip") as archive:
            names = archive.namelist()
            for name in ["main.py", "lambda_function.py", "ingestion.py",
                         "ingestion_lambda.py", "fastapi/__init__.py",
                         "mangum/__init__.py", "httpx/__init__.py", "pypdf/__init__.py",
                         "multipart/__init__.py"]:
                self.assertIn(name, names)
            for name in ["main.py", "lambda_function.py", "ingestion.py", "ingestion_lambda.py"]:
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
            self.assertTrue(any(name.endswith(".so") and "pydantic_core" in name for name in names))
            self.assertFalse(any(name.endswith(".pyd") for name in names))
            self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
