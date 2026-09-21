"""AWS Lambda entry point for the ingestion API."""
from mangum import Mangum
from ingestion import app

lambda_handler = Mangum(app, lifespan="off")
