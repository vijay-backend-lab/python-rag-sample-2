import json
from pathlib import Path
import unittest
from urllib.parse import urlencode
from zipfile import ZipFile

from fastapi.testclient import TestClient
from lambda_function import lambda_handler
from main import app

ROOT = Path(__file__).resolve().parents[1]


class ApiTests(unittest.TestCase):
    def test_strings_preserved(self):
        with TestClient(app) as client:
            for query in ["hello", "", "  space  ", "it's a query", "नमस्ते", "a&b+c"]:
                with self.subTest(query=query):
                    response = client.get("/query", params={"query": query})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.text, f"Your query was '{query}'")
                    self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")

    def test_validation_and_routes(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/query").status_code, 422)
            self.assertEqual(client.get("/missing").status_code, 404)
            self.assertEqual(client.post("/query", params={"query": "hello"}).status_code, 405)
            schema = client.get("/openapi.json").json()
            self.assertTrue(schema["paths"]["/query"]["get"]["parameters"][0]["required"])

    def test_lambda_v2(self):
        for query in ["hello", "", "नमस्ते & +"]:
            event = json.loads((ROOT / "events/query.json").read_text())
            event["rawQueryString"] = urlencode({"query": query})
            event["queryStringParameters"] = {"query": query}
            response = lambda_handler(event, None)
            self.assertEqual(response["statusCode"], 200)
            self.assertEqual(response["body"], f"Your query was '{query}'")
            self.assertFalse(response["isBase64Encoded"])

    def test_lambda_v1(self):
        event = {"resource": "/query", "path": "/query", "httpMethod": "GET",
                 "headers": {"host": "localhost"}, "requestContext": {},
                 "queryStringParameters": {"query": "hello"}, "body": None}
        self.assertEqual(lambda_handler(event, None)["body"], "Your query was 'hello'")

    def test_deployment_zip(self):
        with ZipFile(ROOT / "dist/lambda-query-api.zip") as archive:
            names = archive.namelist()
            for name in ["main.py", "lambda_function.py", "fastapi/__init__.py", "mangum/__init__.py"]:
                self.assertIn(name, names)
            for name in ["main.py", "lambda_function.py"]:
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
            self.assertTrue(any(name.endswith(".so") and "pydantic_core" in name for name in names))
            self.assertFalse(any(name.endswith(".pyd") for name in names))
            self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()

