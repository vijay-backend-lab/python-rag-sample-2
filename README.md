# FastAPI query API

`GET /query?query=What%20is%20RAG%3F` validates the question, checks the caller's
permissions, embeds it with Gemini `gemini-embedding-001` (`RETRIEVAL_QUERY`), and
runs a hybrid Elasticsearch search that combines a BM25 keyword `match` on the text
field with a semantic kNN search on the vector field in a single request, filtered
to the document categories the caller may read. Elasticsearch runs both searches
and combines them by summing their scores, so the top hit reflects both lexical
and semantic relevance. The nearest chunk's content and the validated question
are sent to `gemini-2.5-flash-lite` to generate an answer. The response is JSON
with the generated `answer` and a `citation` object attributing it to the source
document. A companion ingestion API (`POST /ingest`, see below) populates the
index this endpoint searches.

A successful response looks like:

```json
{
  "answer": "Trials must be reviewed annually.",
  "citation": {
    "document_id": "DOC-1",
    "document_type": "SOP",
    "title": "Trial SOP",
    "version": "2",
    "effective_date": "2026-01-15",
    "chunk_index": 0
  }
}
```

The citation is built from the retrieved chunk's stored metadata (the same fields
the ingestion API writes). Any field absent from the indexed document is `null`,
so older documents indexed without full metadata still return a well-formed
citation.

The generation prompt tells Gemini to answer only from the retrieved context,
state when the context is insufficient, and treat document text as data rather
than instructions. This encourages grounding but does not guarantee factual accuracy.
The same `GEMINI_API_KEY` is used for embeddings and generation.

## Configuration

Copy `.env.example` to `.env` and replace its placeholders. Existing environment
variables take precedence over `.env`. The file name is `.env`, not `.evn`.

- `GEMINI_API_KEY`: Gemini API key.
- `ELASTICSEARCH_URL`: Elasticsearch base URL, including HTTP or HTTPS.
- `ELASTICSEARCH_USERNAME` and `ELASTICSEARCH_PASSWORD`: basic authentication.
- `ELASTICSEARCH_INDEX`: existing index to search.
- `ELASTICSEARCH_VECTOR_FIELD`: indexed `dense_vector` field.
- `ELASTICSEARCH_TEXT_FIELD`: string field in `_source` to return. Dotted paths
  such as `document.text` are supported for ordinary object fields.
- `ELASTICSEARCH_MIN_SCORE` (optional): relevance threshold. When set to a
  positive number, Elasticsearch drops hits whose combined (BM25 + kNN) score is
  below it, so weak matches never reach the prompt and yield a 404 instead. Unset
  or `0` disables filtering (the top hit is used regardless of score). Because
  hybrid scores are unbounded and index-dependent, tune this against your own
  index rather than assuming a fixed value. A non-numeric or negative value
  returns 503.

`.env.example` ships generic placeholders (for example `ELASTICSEARCH_INDEX=index_name`).
Replace them with your index's real names. For an index whose vector field is
`embedding` and text field is `content`, use:

```dotenv
ELASTICSEARCH_INDEX=your-index-name
ELASTICSEARCH_VECTOR_FIELD=embedding
ELASTICSEARCH_TEXT_FIELD=content
```

The nearest chunk's content is used as generation context; the parent document is not fetched.

All settings are required. The index must support Elasticsearch's top-level
`knn` search API. Its vectors must use the same embedding model and dimensions
as the query (Gemini's default 3072 dimensions). Index document embeddings using
`RETRIEVAL_DOCUMENT`. An existing index using a different model or dimensionality
needs compatible embeddings before this query pipeline can search it.

The search is hybrid: a BM25 `match` on `ELASTICSEARCH_TEXT_FIELD` runs alongside
the kNN vector search, and Elasticsearch combines the two by summing their scores.
This uses only core Elasticsearch features (no RRF/rank tier), so it works on the
basic tier. The text field must be a searchable `text` type for the keyword half
to contribute; a `keyword`-only field limits it to exact matches. The search
requests one nearest document with 100 candidates per shard. A relevance
threshold is applied when `ELASTICSEARCH_MIN_SCORE` is set (see Configuration);
otherwise there is no minimum similarity threshold. Empty results (including hits
dropped by the threshold) return 404; insufficient permissions return 403
(see Document categories and authorization); missing or
non-string document text and upstream failures return 502; timeouts return 504;
missing configuration returns 503. Error responses use FastAPI's JSON `detail`
format. Successful responses are JSON with `answer` and `citation` fields. Blocked,
empty, malformed, or truncated generation responses return 502.

## Validation

Questions are required and limited to 2,000 characters before trimming. Leading
and trailing whitespace is removed. Blank input, input without letters, and
control characters other than tabs and line breaks return 422 before any
upstream call. Unicode is supported. Validation checks text format rather than
whether the input semantically asks a question; a question mark is not required.

## Document categories and authorization

Every document is categorised as exactly one of `SOP`, `CAPA`, or `AUDIT`.
Retrieval is authorization-aware: a caller may only read a category if it holds
the matching `READ_<CATEGORY>` permission (`READ_SOP`, `READ_CAPA`, `READ_AUDIT`).

The caller presents its granted permissions in the `X-Permissions` request
header as a comma-separated list, for example:

```
X-Permissions: READ_SOP, READ_AUDIT
```

Permission and category names are case-insensitive and unknown tokens are
ignored. The search is always filtered to the caller's permitted categories on
both the keyword and vector halves, so a caller can never retrieve a category it
lacks permission for. A caller with no recognised `READ_*` permission gets 403
before any upstream call.

Optionally pass `document_type` as a query parameter (`SOP`, `CAPA`, or `AUDIT`)
to narrow the search to a single category. The requested category must be one the
caller is permitted to read; an unpermitted or unknown category returns 403
without disclosing whether it exists.

This is a lightweight authorization model: it trusts the `X-Permissions` header,
so in a real deployment the header must be set by a trusted layer (an API Gateway
authorizer, a reverse proxy, or a JWT-to-header mapping) and never accepted
directly from untrusted clients.

## Ingestion API

`ingestion.py` is a second FastAPI app (`POST /ingest`) that populates the index
the query API searches. It is the ingestion pipeline the query side assumes
already exists: PDF text extraction, chunking, metadata enrichment, document
embedding, and Elasticsearch indexing.

Send a `multipart/form-data` request:

- `file` (required): the PDF to ingest.
- `document_id` (required): a stable identifier for the source document. Chunks
  are indexed with `_id` of `{document_id}:{chunk_index}`, so re-ingesting the
  same `document_id` overwrites its chunks.
- `document_type` (required): the document category. Must be one of `SOP`,
  `CAPA`, or `AUDIT` (case-insensitive; stored upper-cased). Any other value
  returns 422. This category drives permission-gated retrieval on the query side.
- `title`, `version`, `effective_date`, `department`, `status` (optional):
  metadata stored on every chunk and usable for retrieval filtering.
  `effective_date` must be an ISO date (`YYYY-MM-DD`); blank optional fields are
  omitted.
- `chunk_size`, `chunk_overlap` (optional): character-based chunk length and
  overlap. Defaults are 1500 and 200. Overlap must be zero or more and smaller
  than the chunk size.

The pipeline extracts text per page (pages with no extractable text are skipped),
chunks each page independently with overlap (so a chunk never spans a page
break), embeds each chunk with Gemini `gemini-embedding-001` using
`RETRIEVAL_DOCUMENT` (matching the query side's `RETRIEVAL_QUERY`), and bulk-indexes
one document per chunk with `refresh=wait_for`. Each indexed document contains the
text field, the vector field, `document_id`, `chunk_index`, and any supplied
metadata. On success it returns `{"document_id": ..., "chunks_indexed": N}`.

Ingestion reuses the same `ELASTICSEARCH_*` and `GEMINI_API_KEY` settings as the
query API. `ELASTICSEARCH_VECTOR_FIELD` and `ELASTICSEARCH_TEXT_FIELD` must be
flat (non-dotted) names for indexing. `ELASTICSEARCH_MIN_SCORE` is query-only and
ignored here.

The index is created automatically when it does not exist. Before bulk-indexing,
ingestion issues a `HEAD` to check for the index and, if absent, `PUT`s a mapping
that sets the vector field to `dense_vector` (with `dims` taken from the actual
embedding length, so it always matches the model output), the text field to
searchable `text`, and the metadata fields (`document_id`, `chunk_index`,
`document_type`, `version`, `effective_date`, `department`, `status`, `title`) to
`keyword`/`integer`/`date`/`text` types so later retrieval filtering works. An
existing index is never modified — its mapping is assumed compatible. Set
`ELASTICSEARCH_SIMILARITY` (`cosine` default, or `dot_product`, `l2_norm`,
`max_inner_product`) to control the vector similarity of a newly created index.
A concurrent creator that wins the race is treated as success. Index-creation
failures return 502 (504 on timeout); an invalid similarity returns 503.

Errors mirror the query API: missing configuration returns 503; an empty file,
an unreadable PDF, no extractable text, a missing or invalid `document_type`, an
invalid `effective_date`, or invalid chunk parameters return 422; a PDF producing
more than 500 chunks returns 413;
embedding, index-creation, or Elasticsearch indexing failures return 502; timeouts
return 504; an invalid `ELASTICSEARCH_SIMILARITY` returns 503.

## Run locally (Python 3.11+)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m uvicorn main:app --reload       # query API
python -m uvicorn ingestion:app --reload  # ingestion API
```

Configure `.env` first, then open http://127.0.0.1:8000/docs. Run one app per
port (for example add `--port 8001` to the second).

## Build and test

```powershell
python -m unittest discover -s tests -v
python build_zip.py
python tests/check_deployment.py
```

API tests mock Gemini embeddings, Gemini generation, and Elasticsearch and require no live
credentials. They use only `unittest`, FastAPI's test client, `httpx`, and (for the
ingestion tests) `pypdf`, all installed by `pip install -e .`, so no extra test
dependencies are needed. Rebuild the deployment archive after source or dependency changes.

## AWS Lambda

The single `dist/lambda-query-api.zip` contains both apps. Deploy each as its own
Python 3.12, x86_64 function that shares the same code but a different handler.

1. Build and upload `dist/lambda-query-api.zip` to a Python 3.12, x86_64 function.
2. Set the handler:
   - Query API: `lambda_function.lambda_handler` (route `GET /query`).
   - Ingestion API: `ingestion_lambda.lambda_handler` (route `POST /ingest`).
3. Configure the variables above in each function; `.env` is not included in the
   ZIP. Allow outbound access to Gemini and Elasticsearch.
4. Connect the API Gateway route to the matching function. Both handlers support
   HTTP API payload format 2.0 and REST API payload format 1.0. For `POST /ingest`,
   enable binary media type `multipart/form-data` on the API so the uploaded PDF
   reaches the function intact.
5. Allow API Gateway to invoke the function. Route `/docs` and `/openapi.json`
   as well if Swagger UI is needed.

Use `events/query.json` for a query-side Lambda console test. Successful query
responses are JSON with `answer` and `citation` fields. Each upstream request has a 30-second
timeout; configure Lambda and gateway timeouts to accommodate the sequential
calls. Ingestion embeds one chunk per request, so give the ingestion function a
higher timeout for large documents.

References: https://ai.google.dev/api/embeddings and
https://www.elastic.co/docs/solutions/search/vector/knn

Generation API reference: https://ai.google.dev/api/generate-content
