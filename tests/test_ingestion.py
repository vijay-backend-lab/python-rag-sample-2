import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from ingestion import app, EMBEDDING_MODEL
from ingestion_lambda import lambda_handler

REAL_CLIENT = httpx.AsyncClient


def make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF whose page renders the given text.

    Hand-built so extract_text() returns real content without extra deps.
    """
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 24 Tf 72 700 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_position = len(pdf)
    pdf += b"xref\n0 %d\n" % (len(objects) + 1)
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += b"%010d 00000 n \n" % offset
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        len(objects) + 1, xref_position)
    return pdf


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.embed_status = 200
        self.embed_timeout = False
        self.embed_payload = {"embedding": {"values": [0.1, -0.2, 0.3]}}
        self.bulk_status = 200
        self.bulk_timeout = False
        # errors=False and one item per indexed chunk is the success shape.
        self.bulk_errors = False
        # Index lifecycle: default is "index already exists" (HEAD 200).
        self.index_exists = True
        self.create_status = 200

        def respond(request):
            self.requests.append(request)
            if request.url.path.endswith(":embedContent"):
                if self.embed_timeout:
                    raise httpx.ReadTimeout("private", request=request)
                return httpx.Response(self.embed_status, json=self.embed_payload)
            # Elasticsearch index existence check.
            if request.method == "HEAD":
                return httpx.Response(200 if self.index_exists else 404)
            # Elasticsearch index creation.
            if request.method == "PUT":
                return httpx.Response(self.create_status,
                                      json={"acknowledged": self.create_status == 200})
            # Elasticsearch _bulk
            if self.bulk_timeout:
                raise httpx.ReadTimeout("private", request=request)
            body = request.content.decode("utf-8")
            actions = [line for line in body.strip().split("\n") if '"index"' in line]
            items = [{"index": {"status": 201}} for _ in actions]
            return httpx.Response(self.bulk_status,
                                  json={"errors": self.bulk_errors, "items": items})

        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key",
            "ELASTICSEARCH_URL": "https://elastic.test", "ELASTICSEARCH_USERNAME": "test-user",
            "ELASTICSEARCH_PASSWORD": "test-password", "ELASTICSEARCH_INDEX": "gov-index",
            "ELASTICSEARCH_VECTOR_FIELD": "embedding", "ELASTICSEARCH_TEXT_FIELD": "content"})
        self.env.start()
        self.addCleanup(self.env.stop)
        mock = patch("ingestion.httpx.AsyncClient", side_effect=lambda **kwargs: REAL_CLIENT(
            transport=httpx.MockTransport(respond), **kwargs))
        mock.start()
        self.addCleanup(mock.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def post(self, pdf_bytes, data=None, filename="doc.pdf"):
        form = {"document_id": "DOC-1"}
        if data:
            form.update(data)
        return self.client.post("/ingest", data=form,
                                files={"file": (filename, pdf_bytes, "application/pdf")})

    def test_happy_path_indexes_chunks(self):
        response = self.post(make_pdf("This is a governance SOP about clinical trials."),
                             data={"document_type": "SOP", "version": "2",
                                   "effective_date": "2026-01-15", "department": "Quality",
                                   "status": "active", "title": "Trial SOP"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["document_id"], "DOC-1")
        self.assertGreaterEqual(result["chunks_indexed"], 1)
        # Embedding used the document task type.
        embed = next(r for r in self.requests if r.url.path.endswith(":embedContent"))
        self.assertEqual(embed.url.path, f"/v1beta/models/{EMBEDDING_MODEL}:embedContent")
        self.assertEqual(json.loads(embed.content)["taskType"], "RETRIEVAL_DOCUMENT")
        # Bulk request carried vector, text, and metadata.
        bulk = next(r for r in self.requests if r.url.path.endswith("/_bulk"))
        self.assertTrue(bulk.headers["authorization"].startswith("Basic "))
        lines = [json.loads(line) for line in bulk.content.decode("utf-8").strip().split("\n")]
        source = lines[1]
        self.assertIn("content", source)
        self.assertEqual(source["embedding"], [0.1, -0.2, 0.3])
        self.assertEqual(source["document_id"], "DOC-1")
        self.assertEqual(source["document_type"], "SOP")
        self.assertEqual(source["effective_date"], "2026-01-15")
        self.assertEqual(source["department"], "Quality")
        self.assertEqual(source["status"], "active")
        self.assertEqual(lines[0]["index"]["_id"], "DOC-1:0")

    def test_existing_index_not_recreated(self):
        self.index_exists = True
        response = self.post(make_pdf("hello world"))
        self.assertEqual(response.status_code, 200)
        # HEAD checked, but no PUT create when the index already exists.
        self.assertTrue(any(r.method == "HEAD" for r in self.requests))
        self.assertFalse(any(r.method == "PUT" for r in self.requests))

    def test_index_created_when_absent(self):
        self.index_exists = False
        response = self.post(make_pdf("hello world"))
        self.assertEqual(response.status_code, 200)
        create = next(r for r in self.requests if r.method == "PUT")
        mapping = json.loads(create.content)["mappings"]["properties"]
        # Vector field uses dense_vector with dims derived from the embedding.
        self.assertEqual(mapping["embedding"]["type"], "dense_vector")
        self.assertEqual(mapping["embedding"]["dims"], 3)
        self.assertEqual(mapping["embedding"]["similarity"], "cosine")
        self.assertEqual(mapping["content"]["type"], "text")
        # Metadata fields are mapped for later filtering.
        self.assertEqual(mapping["document_type"]["type"], "keyword")
        self.assertEqual(mapping["effective_date"]["type"], "date")

    def test_index_created_with_custom_similarity(self):
        self.index_exists = False
        with patch.dict(os.environ, {"ELASTICSEARCH_SIMILARITY": "dot_product"}):
            response = self.post(make_pdf("hello world"))
        self.assertEqual(response.status_code, 200)
        create = next(r for r in self.requests if r.method == "PUT")
        mapping = json.loads(create.content)["mappings"]["properties"]
        self.assertEqual(mapping["embedding"]["similarity"], "dot_product")

    def test_invalid_similarity(self):
        self.index_exists = False
        with patch.dict(os.environ, {"ELASTICSEARCH_SIMILARITY": "manhattan"}):
            self.assertEqual(self.post(make_pdf("hello world")).status_code, 503)

    def test_index_creation_failure(self):
        self.index_exists = False
        self.create_status = 500
        self.assertEqual(self.post(make_pdf("hello world")).status_code, 502)

    def test_concurrent_index_creation_treated_as_success(self):
        # A racing creator wins: PUT returns 400 resource_already_exists.
        self.index_exists = False

        def respond(request):
            self.requests.append(request)
            if request.url.path.endswith(":embedContent"):
                return httpx.Response(200, json=self.embed_payload)
            if request.method == "HEAD":
                return httpx.Response(404)
            if request.method == "PUT":
                return httpx.Response(400, json={"error": {
                    "type": "resource_already_exists_exception"}})
            body = request.content.decode("utf-8")
            actions = [line for line in body.strip().split("\n") if '"index"' in line]
            return httpx.Response(200, json={"errors": False,
                                             "items": [{"index": {}} for _ in actions]})

        with patch("ingestion.httpx.AsyncClient",
                   side_effect=lambda **kwargs: REAL_CLIENT(
                       transport=httpx.MockTransport(respond), **kwargs)):
            response = self.post(make_pdf("hello world"))
        self.assertEqual(response.status_code, 200)

    def test_missing_document_id(self):
        response = self.client.post("/ingest", data={"document_id": "  "},
            files={"file": ("d.pdf", make_pdf("hello"), "application/pdf")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])

    def test_non_pdf_rejected(self):
        response = self.post(b"not a pdf at all")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])

    def test_empty_file_rejected(self):
        response = self.post(b"")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])

    def test_invalid_effective_date(self):
        response = self.post(make_pdf("hello world"), data={"effective_date": "15-01-2026"})
        self.assertEqual(response.status_code, 422)

    def test_invalid_chunk_params(self):
        response = self.post(make_pdf("hello world"), data={"chunk_overlap": "2000", "chunk_size": "1500"})
        self.assertEqual(response.status_code, 422)

    def test_chunking_produces_multiple_chunks(self):
        response = self.post(make_pdf("word " * 800),
                             data={"chunk_size": "500", "chunk_overlap": "50"})
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["chunks_indexed"], 1)

    def test_missing_configuration(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertEqual(self.post(make_pdf("hello world")).status_code, 503)

    def test_embedding_failure(self):
        self.embed_status = 500
        self.assertEqual(self.post(make_pdf("hello world")).status_code, 502)

    def test_embedding_timeout(self):
        self.embed_timeout = True
        self.assertEqual(self.post(make_pdf("hello world")).status_code, 504)

    def test_bulk_failure(self):
        self.bulk_status = 500
        self.assertEqual(self.post(make_pdf("hello world")).status_code, 502)

    def test_bulk_item_errors(self):
        self.bulk_errors = True
        self.assertEqual(self.post(make_pdf("hello world")).status_code, 502)

    def test_bulk_timeout(self):
        self.bulk_timeout = True
        self.assertEqual(self.post(make_pdf("hello world")).status_code, 504)

    def test_lambda_handler_v2(self):
        # Smoke-test that the Mangum handler is wired to the ingestion app.
        event = {"version": "2.0", "routeKey": "GET /openapi.json", "rawPath": "/openapi.json",
                 "rawQueryString": "", "headers": {"host": "localhost"},
                 "requestContext": {"http": {"method": "GET", "path": "/openapi.json",
                 "sourceIp": "127.0.0.1", "protocol": "HTTP/1.1"}}, "isBase64Encoded": False}
        response = lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertIn("/ingest", response["body"])


if __name__ == "__main__":
    unittest.main()
