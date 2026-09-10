# FastAPI query API

`GET /query?query=What%20is%20RAG%3F` validates a user's question and generates
its vector with Gemini `gemini-embedding-001` using `RETRIEVAL_QUERY`.

The required query is limited to 2,000 characters before trimming. Leading and
trailing whitespace is removed. Blank input, text without any letters, and
embedded control characters (except tabs and line breaks) return HTTP 422
without calling Gemini. Unicode questions are supported. Validation checks text
shape, not meaning: it does not attempt to prove that text is a question or
require a question mark.

The response is now JSON rather than plain text:

```json
{"query": "What is RAG?", "embedding": [0.1, -0.2, 0.3], "model": "gemini-embedding-001"}
```

The vector above is illustrative. `embed_question()` returns the actual vector
for a future retrieval step; no vector search or answer generation is performed.

## Run locally (Python 3.11+)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test]"
$env:GEMINI_API_KEY = "your-api-key"
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.
The key is read from the environment; `.env` files are not automatically loaded.
Missing configuration returns 503, upstream failures return 502, and upstream
timeouts return 504. Provider error details and credentials are not returned.

## Build and test

```powershell
python -m unittest discover -s tests -v
python build_zip.py
python tests/check_deployment.py
```

Tests mock Gemini HTTP calls and require no API key or network access.
Rebuild the deployment archive after source or dependency changes.

## AWS Lambda

1. Build and upload `dist/lambda-query-api.zip` to a Python 3.12, x86_64 function.
2. Set the handler to `lambda_function.lambda_handler`.
3. Set the `GEMINI_API_KEY` environment variable and allow outbound HTTPS access.
4. Connect API Gateway route `GET /query` (payload format 2.0).
5. Allow API Gateway to invoke the function. Route `/docs` and `/openapi.json`
   as well if Swagger UI is needed.

Use `events/query.json` for a Lambda console test. HTTP callers receive JSON.

Gemini API reference: https://ai.google.dev/api/embeddings

