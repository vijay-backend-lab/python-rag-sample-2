"""FastAPI ingestion endpoint.

Ingests a PDF into Elasticsearch as searchable, embedded chunks:
extract text -> chunk -> enrich with metadata -> embed each chunk with Gemini
(``RETRIEVAL_DOCUMENT``) -> bulk index. The resulting index is what the query
API (`main.py`) searches, so the vector field, text field, and embedding model
must match the query side.
"""
import io
import json
import math
import os
from datetime import date
from typing import Annotated, Optional
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI(title="Ingestion API")

EMBEDDING_MODEL = "gemini-embedding-001"

# Chunking defaults (characters). Chunks overlap so context that straddles a
# boundary is preserved in at least one chunk.
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNKS = 500

# kNN vector similarity for the created index. Cosine is a safe default for
# Gemini embeddings; override with ELASTICSEARCH_SIMILARITY if the index needs
# dot_product or l2_norm.
DEFAULT_SIMILARITY = "cosine"
VALID_SIMILARITIES = ("cosine", "dot_product", "l2_norm", "max_inner_product")

load_dotenv()


class IngestResult(BaseModel):
    document_id: str
    chunks_indexed: int


def extract_pdf_text(data: bytes) -> list[str]:
    """Return the text of each page, dropping pages with no extractable text."""
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except HTTPException:
        raise
    except Exception as exc:  # pypdf raises a variety of parse errors
        raise HTTPException(422, "The uploaded file is not a readable PDF.") from exc
    pages = [page for page in pages if page]
    if not pages:
        raise HTTPException(422, "No extractable text found in the PDF.")
    return pages


def chunk_text(pages: list[str], size: int, overlap: int) -> list[str]:
    """Split page text into overlapping character chunks.

    Pages are chunked independently so a chunk never spans a page break, which
    keeps page-level context intact and avoids merging unrelated sections.
    """
    if size <= 0:
        raise HTTPException(422, "Chunk size must be a positive number.")
    if overlap < 0 or overlap >= size:
        raise HTTPException(422, "Chunk overlap must be zero or more and less than the chunk size.")
    step = size - overlap
    chunks: list[str] = []
    for page in pages:
        start = 0
        while start < len(page):
            piece = page[start:start + size].strip()
            if piece:
                chunks.append(piece)
            start += step
    if not chunks:
        raise HTTPException(422, "The PDF produced no non-empty chunks.")
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(413, f"The PDF produced too many chunks (limit {MAX_CHUNKS}).")
    return chunks


async def embed_documents(chunks: list[str]) -> list[list[float]]:
    """Embed each chunk with the document-side task type for retrieval indexing."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "Embedding service is not configured.")
    vectors: list[list[float]] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for chunk in chunks:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent",
                    headers={"x-goog-api-key": api_key},
                    json={
                        "model": f"models/{EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": chunk}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                    },
                )
                response.raise_for_status()
                values = response.json()["embedding"]["values"]
                if not isinstance(values, list) or not values or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values
                ):
                    raise ValueError("Invalid vector")
                vectors.append(values)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Embedding service timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Embedding service request failed.") from exc
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(502, "Embedding service returned an invalid vector.") from exc
    return vectors


def build_metadata(
    title: Optional[str],
    document_type: Optional[str],
    version: Optional[str],
    effective_date: Optional[str],
    department: Optional[str],
    status: Optional[str],
) -> dict:
    """Validate and normalise the enrichment metadata attached to each chunk."""
    metadata: dict[str, str] = {}
    for key, value in (
        ("title", title),
        ("document_type", document_type),
        ("version", version),
        ("department", department),
        ("status", status),
    ):
        if value is not None:
            text = value.strip()
            if text:
                metadata[key] = text
    if effective_date is not None and effective_date.strip():
        try:
            metadata["effective_date"] = date.fromisoformat(effective_date.strip()).isoformat()
        except ValueError as exc:
            raise HTTPException(422, "effective_date must be an ISO date (YYYY-MM-DD).") from exc
    return metadata


async def ensure_index(client: httpx.AsyncClient, url: str, index: str,
                       vector_field: str, text_field: str, dims: int) -> None:
    """Create the index with a RAG-ready mapping if it does not already exist.

    Idempotent: an existing index is left untouched (its mapping is assumed
    compatible). A newly created index maps the vector field as ``dense_vector``
    with the query dimensions, the text field as searchable ``text``, and the
    enrichment metadata as ``keyword``/``date``/``integer`` fields so retrieval
    filtering by document type, version, effective date, department, and status
    works later. Creation uses ``PUT`` so a concurrent creator that wins the race
    (HTTP 400 ``resource_already_exists_exception``) is treated as success.
    """
    similarity = os.environ.get("ELASTICSEARCH_SIMILARITY", "").strip() or DEFAULT_SIMILARITY
    if similarity not in VALID_SIMILARITIES:
        raise HTTPException(503, "Elasticsearch similarity must be one of: "
                            + ", ".join(VALID_SIMILARITIES) + ".")
    index_url = f"{url}/{quote(index, safe='')}"
    try:
        head = await client.head(index_url)
        if head.status_code == 200:
            return
        if head.status_code != 404:
            head.raise_for_status()
        mapping = {
            "mappings": {
                "properties": {
                    vector_field: {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": similarity,
                    },
                    text_field: {"type": "text"},
                    "document_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "title": {"type": "text"},
                    "document_type": {"type": "keyword"},
                    "version": {"type": "keyword"},
                    "effective_date": {"type": "date"},
                    "department": {"type": "keyword"},
                    "status": {"type": "keyword"},
                },
            },
        }
        create = await client.put(index_url, json=mapping)
        if create.status_code == 400 and "resource_already_exists" in create.text:
            return
        create.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Elasticsearch index creation timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Elasticsearch index creation failed.") from exc


async def index_chunks(document_id: str, chunks: list[str], vectors: list[list[float]],
                       metadata: dict) -> int:
    """Ensure the index exists, then bulk-index one document per chunk."""
    url = os.environ.get("ELASTICSEARCH_URL", "").strip().rstrip("/")
    username = os.environ.get("ELASTICSEARCH_USERNAME", "").strip()
    password = os.environ.get("ELASTICSEARCH_PASSWORD", "")
    index = os.environ.get("ELASTICSEARCH_INDEX", "").strip()
    vector_field = os.environ.get("ELASTICSEARCH_VECTOR_FIELD", "").strip()
    text_field = os.environ.get("ELASTICSEARCH_TEXT_FIELD", "").strip()
    if not all((url, username, password, index, vector_field, text_field)):
        raise HTTPException(503, "Elasticsearch is not configured.")
    if not url.startswith(("https://", "http://")):
        raise HTTPException(503, "Elasticsearch URL must use HTTP or HTTPS.")
    if "." in text_field or "." in vector_field:
        raise HTTPException(503, "Ingestion requires flat (non-dotted) vector and text field names.")
    lines: list[str] = []
    for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{document_id}:{position}"
        source = {
            text_field: chunk,
            vector_field: vector,
            "document_id": document_id,
            "chunk_index": position,
            **metadata,
        }
        lines.append(json.dumps({"index": {"_index": index, "_id": chunk_id}}))
        lines.append(json.dumps(source, ensure_ascii=False))
    payload = "\n".join(lines) + "\n"
    dims = len(vectors[0])
    try:
        async with httpx.AsyncClient(timeout=30.0, auth=(username, password), verify=False) as client:
            await ensure_index(client, url, index, vector_field, text_field, dims)
            response = await client.post(
                f"{url}/{quote(index, safe='')}/_bulk?refresh=wait_for",
                content=payload.encode("utf-8"),
                headers={"content-type": "application/x-ndjson"},
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Elasticsearch indexing timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Elasticsearch indexing failed.") from exc
    try:
        result = response.json()
        if result.get("errors"):
            raise ValueError("Bulk indexing reported item errors")
        items = result["items"]
        if not isinstance(items, list) or len(items) != len(chunks):
            raise ValueError("Unexpected bulk response")
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(502, "Elasticsearch returned an invalid indexing response.") from exc
    return len(chunks)


@app.post("/ingest", response_model=IngestResult)
async def ingest(
    file: Annotated[UploadFile, File(description="PDF document to ingest.")],
    document_id: Annotated[str, Form(description="Stable identifier for the source document.")],
    title: Annotated[Optional[str], Form()] = None,
    document_type: Annotated[Optional[str], Form(description="e.g. SOP, policy, compliance.")] = None,
    version: Annotated[Optional[str], Form()] = None,
    effective_date: Annotated[Optional[str], Form(description="ISO date YYYY-MM-DD.")] = None,
    department: Annotated[Optional[str], Form()] = None,
    status: Annotated[Optional[str], Form(description="e.g. active, superseded, draft.")] = None,
    chunk_size: Annotated[int, Form()] = DEFAULT_CHUNK_SIZE,
    chunk_overlap: Annotated[int, Form()] = DEFAULT_CHUNK_OVERLAP,
) -> IngestResult:
    identifier = document_id.strip()
    if not identifier:
        raise HTTPException(422, "document_id is required.")
    data = await file.read()
    pages = extract_pdf_text(data)
    chunks = chunk_text(pages, chunk_size, chunk_overlap)
    metadata = build_metadata(title, document_type, version, effective_date, department, status)
    vectors = await embed_documents(chunks)
    indexed = await index_chunks(identifier, chunks, vectors, metadata)
    return IngestResult(document_id=identifier, chunks_indexed=indexed)
