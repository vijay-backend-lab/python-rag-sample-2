import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from lambda_function import lambda_handler
from main import app, EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[1]
REAL_CLIENT = httpx.AsyncClient


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.status = 200
        self.payload = {"embedding": {"values": [0.1, -0.2, 0.3]}}
        self.timeout = False

        def respond(request):
            self.requests.append(request)
            if self.timeout:
                raise httpx.ReadTimeout("private upstream details", request=request)
            return httpx.Response(self.status, json=self.payload)

        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        mock = patch("main.httpx.AsyncClient", side_effect=lambda **kwargs: REAL_CLIENT(
            transport=httpx.MockTransport(respond), **kwargs))
        mock.start()
        self.addCleanup(mock.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_questions_embedded(self):
        for question in ["What is RAG?", "Explain embeddings", "यह क्या है?", "  How does it work?  "]:
            with self.subTest(question=question):
                response = self.client.get("/query", params={"query": question})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"query": question.strip(),
                    "embedding": [0.1, -0.2, 0.3], "model": EMBEDDING_MODEL})
                request = self.requests[-1]
                self.assertEqual(request.headers["x-goog-api-key"], "test-key")
                self.assertEqual(json.loads(request.content), {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": question.strip()}]},
                    "taskType": "RETRIEVAL_QUERY"})

    def test_invalid_questions_never_call_gemini(self):
        for question in [None, "", " \t\n", "???", "123", "a" * 2001, "What\x00?"]:
            with self.subTest(question=question):
                params = {} if question is None else {"query": question}
                self.assertEqual(self.client.get("/query", params=params).status_code, 422)
        self.assertEqual(self.requests, [])

    def test_length_boundary(self):
        self.assertEqual(self.client.get("/query", params={"query": "a" * 2000}).status_code, 200)

    def test_missing_configuration(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 503)
        self.assertEqual(self.requests, [])

    def test_upstream_errors(self):
        for status in [400, 401, 429, 500]:
            self.status = status
            response = self.client.get("/query", params={"query": "Why?"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("test-key", response.text)
        self.timeout = True
        self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 504)

    def test_invalid_vectors(self):
        for payload in [{}, {"embedding": None}, {"embedding": {"values": []}},
                        {"embedding": {"values": [True]}}, {"embedding": {"values": ["bad"]}}]:
            self.payload = payload
            self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 502)

    def test_routes(self):
        self.assertEqual(self.client.get("/missing").status_code, 404)
        self.assertEqual(self.client.post("/query").status_code, 405)

    def test_lambda_v2(self):
        event = json.loads((ROOT / "events/query.json").read_text())
        response = lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"])["embedding"], [0.1, -0.2, 0.3])

    def test_lambda_v1(self):
        event = {"resource": "/query", "path": "/query", "httpMethod": "GET",
                 "headers": {"host": "localhost"}, "requestContext": {},
                 "queryStringParameters": {"query": "What is RAG?"}, "body": None}
        response = lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"])["query"], "What is RAG?")


if __name__ == "__main__":
    unittest.main()
