# FastAPI query API

GET /query?query=hello returns plain text: Your query was 'hello'.
The required query parameter is a string; empty strings are accepted.
Missing parameters now return FastAPI's standard 422 validation response.

- main.py: FastAPI app and endpoint.
- lambda_function.py: separate Mangum adapter for AWS Lambda.
- pyproject.toml: project metadata and dependencies.

## Run locally (Python 3.11+)

From this project folder in VS terminal:

    python -m venv .venv
    pip install -e ".[test]"
    python -m uvicorn main:app --reload

Open http://127.0.0.1:8000/docs to try the API in Swagger UI, or run:

    curl.exe "http://127.0.0.1:8000/query?query=hello"

Use Ctrl+C to stop. Add --port 8080 to the Uvicorn command to change the port.

## Build and test
    <!-- Create the zip for aws lambda function -->
    python build_zip.py
    python -m unittest discover -s tests -v


## AWS Lambda

1. Create or configure a function with **Python 3.12**, architecture **x86_64**,
   and a basic Lambda execution role.
2. Upload dist/lambda-query-api.zip under Code > Upload from > .zip file.
3. Set the handler to **lambda_function.lambda_handler**.
4. Connect an API Gateway HTTP API route **GET /query** to the Lambda using
   payload format **2.0** and enable automatic deployment on the $default stage.
   Allow API Gateway to invoke the function during setup.
5. Call https://YOUR_API_ID.execute-api.YOUR_REGION.amazonaws.com/query?query=hello.

Paste events/query.json into a Lambda console test to test the handler directly.
The console displays the response envelope; HTTP callers receive its plain-text body.
For Swagger UI on AWS, also route /docs and /openapi.json to this Lambda.


## Test aws lambda function
```
{
  "version": "2.0",
  "routeKey": "GET /query",
  "rawPath": "/query",
  "rawQueryString": "query=hello",
  "headers": {
    "host": "localhost"
  },
  "requestContext": {
    "http": {
      "method": "GET",
      "path": "/query",
      "sourceIp": "127.0.0.1",
      "protocol": "HTTP/1.1"
    }
  },
  "queryStringParameters": {
    "query": "hello"
  },
  "isBase64Encoded": false
}
```

