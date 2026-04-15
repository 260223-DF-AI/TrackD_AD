"""FastAPI app (API endpoints)"""
from fastapi import FastAPI

import os
import time
import logging

from app.routers import classify
from app.routers import bedrock

# implementing logger functionality
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('reporting.log')
fomatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

logger.setLevel(logging.INFO)

app = FastAPI(
    title = "TrackD_AD Estate Insight API service connecting vision models and LLM prompts",
    description = "API for connecting vision models and LLM prompts",
    version = "0.0.1"
)

@app.get("/")
def get_root():
    return {"message" : "Hello from main"}

# Routers
app.include_router(classify.router)
app.include_router(bedrock.router, prefix="/bedrock", tags=["bedrock"])