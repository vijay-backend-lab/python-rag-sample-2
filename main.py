"""FastAPI query endpoint."""
import math
import os
from dotenv import load_dotenv
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import AfterValidator, BaseModel

app = FastAPI(title="Query API")


EMBEDDING_MODEL = "gemini-embedding-001"

load_dotenv()

def validate_question(value: str) -> str:
    question = value.strip()
    if not question or not any(character.isalpha() for character in question):
        raise ValueError("Provide a question containing text.")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in question):
        raise ValueError("Question must not contain control characters.")
    return question


class QueryResponse(BaseModel):
    query: str
    embedding: list[float]
    model: str


async def embed_question(question: str) -> list[float]:
    """Create a retrieval vector that later pipeline steps can consume."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    print(bool(api_key))
    if not api_key:
        raise HTTPException(503, "Embedding service is not configured.Vijay")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent",
                headers={"x-goog-api-key": api_key},
                json={
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": question}]},
                    "taskType": "RETRIEVAL_QUERY",
                },
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Embedding service timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Embedding service request failed.") from exc
    try:
        values = response.json()["embedding"]["values"]
        if not isinstance(values, list) or not values or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in values
        ):
            raise ValueError("Invalid vector")
        return values
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(502, "Embedding service returned an invalid vector.") from exc


@app.get("/query", response_model=QueryResponse)
async def query(query: Annotated[
    str,
    Query(min_length=1, max_length=2000, description="The user's question."),
    AfterValidator(validate_question),
]) -> QueryResponse:
    embedding = await embed_question(query)
    return QueryResponse(query=query, embedding=embedding, model=EMBEDDING_MODEL)

