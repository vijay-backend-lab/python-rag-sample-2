"""FastAPI query endpoint."""
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Query API")


@app.get("/query", response_class=PlainTextResponse)
async def query(query: str) -> str:
    return f"Your query was '{query}'"

