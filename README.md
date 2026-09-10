# FastAPI query API

`GET /query?query=What%20is%20RAG%3F` validates the question, embeds it with
Gemini `gemini-embedding-001` (`RETRIEVAL_QUERY`), and uses that vector for an
Elasticsearch kNN search. The nearest document's configured text field is
returned directly as `text/plain; charset=utf-8`, with no JSON wrapper,
embedding, score, or other search metadata. No answer generation is performed.

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

For the supplied mapping, use:

```dotenv
ELASTICSEARCH_INDEX=structural-rag-index
ELASTICSEARCH_VECTOR_FIELD=embedding
ELASTICSEARCH_TEXT_FIELD=content
```

The response is the nearest chunk's content, not the entire parent document.

All settings are required. The index must support Elasticsearch's top-level
`knn` search API. Its vectors must use the same embedding model and dimensions
as the query (Gemini's default 3072 dimensions). Index document embeddings using
`RETRIEVAL_DOCUMENT`. An existing index using a different model or dimensionality
needs compatible embeddings before this query pipeline can search it.

The search requests one nearest document with 100 candidates per shard.
There is no minimum similarity threshold. Empty results return 404; missing or
non-string document text and upstream failures return 502; timeouts return 504;
missing configuration returns 503. Error responses use FastAPI's JSON `detail`
format. Only successful responses contain the document string.

## Validation

Questions are required and limited to 2,000 characters before trimming. Leading
and trailing whitespace is removed. Blank input, input without letters, and
control characters other than tabs and line breaks return 422 before any
upstream call. Unicode is supported. Validation checks text format rather than
whether the input semantically asks a question; a question mark is not required.

## Run locally (Python 3.11+)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test]"
python -m uvicorn main:app --reload
```

Configure `.env` first, then open http://127.0.0.1:8000/docs.

## Build and test

```powershell
python -m unittest discover -s tests -v
python build_zip.py
python tests/check_deployment.py
```

API tests mock both Gemini and Elasticsearch and require no live credentials.
Rebuild the deployment archive after source or dependency changes.

## AWS Lambda

1. Build and upload `dist/lambda-query-api.zip` to a Python 3.12, x86_64 function.
2. Set the handler to `lambda_function.lambda_handler`.
3. Configure the variables above in Lambda; `.env` is not included in the ZIP.
   Allow outbound access to Gemini and Elasticsearch.
4. Connect API Gateway route `GET /query` (payload format 2.0).
5. Allow API Gateway to invoke the function. Route `/docs` and `/openapi.json`
   as well if Swagger UI is needed.

Use `events/query.json` for a Lambda console test. Successful HTTP responses
contain only the matched document's text.

References: https://ai.google.dev/api/embeddings and
https://www.elastic.co/docs/solutions/search/vector/knn
