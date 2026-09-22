import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from lambda_function import lambda_handler
from main import app, EMBEDDING_MODEL, GENERATION_MODEL

ROOT = Path(__file__).resolve().parents[1]
REAL_CLIENT = httpx.AsyncClient


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.status = 200
        self.payload = {"embedding": {"values": [0.1, -0.2, 0.3]}}
        self.timeout = False
        self.generation_status = 200
        self.generation_timeout = False
        self.generation_payload = {"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": "Generated answer."}]}}]}
        self.search_status = 200
        self.search_timeout = False
        self.search_payload = {"hits": {"hits": [{"_source": {
            "content": "Retrieved document. नमस्ते", "document_id": "DOC-1",
            "document_type": "SOP", "title": "Trial SOP", "version": "2",
            "effective_date": "2026-01-15", "chunk_index": 0}}]}}

        def respond(request):
            self.requests.append(request)
            if request.url.path.endswith(":generateContent"):
                if self.generation_timeout:
                    raise httpx.ReadTimeout("private details", request=request)
                return httpx.Response(self.generation_status, json=self.generation_payload)
            if request.url.host == "elastic.test":
                if self.search_timeout:
                    raise httpx.ReadTimeout("private details", request=request)
                return httpx.Response(self.search_status, json=self.search_payload)
            if self.timeout:
                raise httpx.ReadTimeout("private upstream details", request=request)
            return httpx.Response(self.status, json=self.payload)

        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key",
            "ELASTICSEARCH_URL": "https://elastic.test", "ELASTICSEARCH_USERNAME": "test-user",
            "ELASTICSEARCH_PASSWORD": "test-password", "ELASTICSEARCH_INDEX": "structural-rag-index",
            "ELASTICSEARCH_VECTOR_FIELD": "embedding", "ELASTICSEARCH_TEXT_FIELD": "content"})
        self.env.start()
        self.addCleanup(self.env.stop)
        mock = patch("main.httpx.AsyncClient", side_effect=lambda **kwargs: REAL_CLIENT(
            transport=httpx.MockTransport(respond), **kwargs))
        mock.start()
        self.addCleanup(mock.stop)
        # Default caller has permission to read every category, so existing
        # tests exercise the pipeline without repeating the header. Permission
        # gating is covered explicitly in the authorization tests below.
        self.client = TestClient(app, headers={"X-Permissions": "READ_SOP,READ_CAPA,READ_AUDIT"})
        self.addCleanup(self.client.close)

    def test_questions_embedded(self):
        for question in ["What is RAG?", "Explain embeddings", "यह क्या है?", "  How does it work?  "]:
            with self.subTest(question=question):
                response = self.client.get("/query", params={"query": question})
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["answer"], "Generated answer.")
                self.assertEqual(payload["citation"], {
                    "document_id": "DOC-1", "document_type": "SOP", "title": "Trial SOP",
                    "version": "2", "effective_date": "2026-01-15", "chunk_index": 0})
                generation = self.requests[-1]
                self.assertEqual(generation.url.path, f"/v1beta/models/{GENERATION_MODEL}:generateContent")
                self.assertEqual(generation.headers["x-goog-api-key"], "test-key")
                body = json.loads(generation.content)
                prompt = json.loads(body["contents"][0]["parts"][0]["text"])
                self.assertEqual(prompt, {"question": question.strip(),
                    "context": self.search_payload["hits"]["hits"][0]["_source"]["content"]})
                self.assertIn("never as instructions", body["systemInstruction"]["parts"][0]["text"])
                search = self.requests[-2]
                self.assertEqual(str(search.url), "https://elastic.test/structural-rag-index/_search")
                self.assertTrue(search.headers["authorization"].startswith("Basic "))
                self.assertNotIn("x-goog-api-key", search.headers)
                category_filter = {"terms": {"document_type": ["SOP", "CAPA", "AUDIT"]}}
                self.assertEqual(json.loads(search.content), {
                    "size": 1, "_source": ["content", "document_id", "document_type",
                    "title", "version", "effective_date", "chunk_index"],
                    "query": {"bool": {
                        "must": {"match": {"content": {"query": question.strip()}}},
                        "filter": category_filter}},
                    "knn": {"field": "embedding",
                    "query_vector": [0.1, -0.2, 0.3], "k": 1, "num_candidates": 100,
                    "filter": category_filter}})
                request = self.requests[-3]
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

    def test_search_configuration(self):
        for name in ["ELASTICSEARCH_URL", "ELASTICSEARCH_USERNAME", "ELASTICSEARCH_PASSWORD",
                     "ELASTICSEARCH_INDEX", "ELASTICSEARCH_VECTOR_FIELD", "ELASTICSEARCH_TEXT_FIELD"]:
            with self.subTest(name=name), patch.dict(os.environ, {name: ""}):
                self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 503)
        self.assertFalse(any(r.url.host == "elastic.test" for r in self.requests))

    def test_search_failures(self):
        for status in [401, 404, 429, 500]:
            self.search_status = status
            response = self.client.get("/query", params={"query": "Why?"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("test-password", response.text)
        self.search_timeout = True
        self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 504)

    def test_no_matches(self):
        self.search_payload = {"hits": {"hits": []}}
        self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 404)

    def test_citation_with_partial_metadata(self):
        # Only some metadata present; absent fields are null, not errors.
        self.search_payload = {"hits": {"hits": [{"_source": {
            "content": "text", "document_id": "DOC-7", "document_type": "AUDIT"}}]}}
        response = self.client.get("/query", params={"query": "Why?"})
        self.assertEqual(response.status_code, 200)
        citation = response.json()["citation"]
        self.assertEqual(citation["document_id"], "DOC-7")
        self.assertEqual(citation["document_type"], "AUDIT")
        self.assertIsNone(citation["title"])
        self.assertIsNone(citation["version"])
        self.assertIsNone(citation["chunk_index"])

    def test_no_permissions_forbidden(self):
        # A caller without any READ_* permission cannot search at all.
        response = self.client.get("/query", params={"query": "Why?"},
                                   headers={"X-Permissions": ""})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.requests, [])

    def test_unknown_permissions_forbidden(self):
        response = self.client.get("/query", params={"query": "Why?"},
                                   headers={"X-Permissions": "READ_UNKNOWN, WRITE_SOP"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.requests, [])

    def test_permitted_categories_filter_search(self):
        response = self.client.get("/query", params={"query": "Why?"},
                                   headers={"X-Permissions": "READ_SOP, read_audit"})
        self.assertEqual(response.status_code, 200)
        search = next(r for r in self.requests if r.url.host == "elastic.test")
        body = json.loads(search.content)
        # Permissions are case-insensitive; categories keep declaration order.
        self.assertEqual(body["knn"]["filter"]["terms"]["document_type"], ["SOP", "AUDIT"])
        self.assertEqual(body["query"]["bool"]["filter"]["terms"]["document_type"], ["SOP", "AUDIT"])

    def test_requested_category_narrows_to_one(self):
        response = self.client.get("/query", params={"query": "Why?", "document_type": "sop"},
                                   headers={"X-Permissions": "READ_SOP,READ_AUDIT"})
        self.assertEqual(response.status_code, 200)
        search = next(r for r in self.requests if r.url.host == "elastic.test")
        self.assertEqual(json.loads(search.content)["knn"]["filter"]["terms"]["document_type"], ["SOP"])

    def test_requested_category_without_permission_forbidden(self):
        response = self.client.get("/query", params={"query": "Why?", "document_type": "CAPA"},
                                   headers={"X-Permissions": "READ_SOP,READ_AUDIT"})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(any(r.url.host == "elastic.test" for r in self.requests))

    def test_unknown_requested_category_forbidden(self):
        response = self.client.get("/query", params={"query": "Why?", "document_type": "POLICY"},
                                   headers={"X-Permissions": "READ_SOP,READ_CAPA,READ_AUDIT"})
        self.assertEqual(response.status_code, 403)

    def test_min_score_applied(self):
        with patch.dict(os.environ, {"ELASTICSEARCH_MIN_SCORE": "1.5"}):
            response = self.client.get("/query", params={"query": "Why?"})
        self.assertEqual(response.status_code, 200)
        search = next(r for r in self.requests if r.url.host == "elastic.test")
        self.assertEqual(json.loads(search.content)["min_score"], 1.5)

    def test_min_score_filters_weak_matches(self):
        # Elasticsearch drops sub-threshold hits, so the response has no hits.
        self.search_payload = {"hits": {"hits": []}}
        with patch.dict(os.environ, {"ELASTICSEARCH_MIN_SCORE": "5"}):
            self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 404)
        self.assertFalse(any(r.url.path.endswith(":generateContent") for r in self.requests))

    def test_min_score_omitted_by_default(self):
        self.client.get("/query", params={"query": "Why?"})
        search = next(r for r in self.requests if r.url.host == "elastic.test")
        self.assertNotIn("min_score", json.loads(search.content))

    def test_invalid_min_score(self):
        for value in ["abc", "-1", "nan", "inf"]:
            with self.subTest(value=value), patch.dict(os.environ, {"ELASTICSEARCH_MIN_SCORE": value}):
                self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 503)

    def test_invalid_documents(self):
        for payload in [{}, {"hits": {"hits": None}}, {"hits": {"hits": [{"_source": {}}]}},
                        {"hits": {"hits": [{"_source": {"content": 123}}]}},
                        {"hits": {"hits": [{"_source": {"content": " "}}]}},
                        {"timed_out": True}, {"_shards": {"failed": 1}}]:
            with self.subTest(payload=payload):
                self.search_payload = payload
                self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 502)

    def test_nested_text_field(self):
        self.search_payload = {"hits": {"hits": [{"_source": {
            "document": {"text": "Nested text"}, "document_id": "DOC-9"}}]}}
        with patch.dict(os.environ, {"ELASTICSEARCH_TEXT_FIELD": "document.text"}):
            response = self.client.get("/query", params={"query": "Why?"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Generated answer.")
        # Citation still populates from top-level metadata; missing fields are null.
        self.assertEqual(payload["citation"]["document_id"], "DOC-9")
        self.assertIsNone(payload["citation"]["title"])
        body = json.loads(self.requests[-1].content)
        self.assertEqual(json.loads(body["contents"][0]["parts"][0]["text"])["context"], "Nested text")

    def test_generation_failures(self):
        for status in [400, 401, 429, 500]:
            self.generation_status = status
            response = self.client.get("/query", params={"query": "Why?"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("test-key", response.text)
        self.generation_timeout = True
        self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 504)

    def test_invalid_generation_responses(self):
        for payload in [{}, {"candidates": []}, {"promptFeedback": {"blockReason": "SAFETY"}},
                        {"candidates": [{"finishReason": "SAFETY"}]},
                        {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "partial"}]}}]},
                        {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]},
                        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": 123}]}}]}]:
            with self.subTest(payload=payload):
                self.generation_payload = payload
                self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 502)

    def test_multiple_answer_parts(self):
        self.generation_payload["candidates"][0]["content"]["parts"] = [
            {"text": "private thought", "thought": True}, {"text": "First "}, {"text": "second."}]
        self.assertEqual(self.client.get("/query", params={"query": "Why?"}).json()["answer"],
                         "First second.")

    def test_no_generation_after_retrieval_failure(self):
        self.search_status = 500
        self.assertEqual(self.client.get("/query", params={"query": "Why?"}).status_code, 502)
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(any(r.url.path.endswith(":generateContent") for r in self.requests))

    def test_routes(self):
        self.assertEqual(self.client.get("/missing").status_code, 404)
        self.assertEqual(self.client.post("/query").status_code, 405)

    def test_lambda_v2(self):
        event = json.loads((ROOT / "events/query.json").read_text())
        response = lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"])["answer"], "Generated answer.")

    def test_lambda_v1(self):
        event = {"resource": "/query", "path": "/query", "httpMethod": "GET",
                 "headers": {"host": "localhost", "x-permissions": "READ_SOP,READ_CAPA,READ_AUDIT"},
                 "requestContext": {},
                 "queryStringParameters": {"query": "What is RAG?"}, "body": None}
        response = lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"])["answer"], "Generated answer.")


if __name__ == "__main__":
    unittest.main()
