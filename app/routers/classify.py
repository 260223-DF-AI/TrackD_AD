"""
Vision model endpoint.
Accepts an image upload and returns the predicted room type with confidence scores.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.schemas import ClassifyResponse
import logging

# implementing logger functionality
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('reporting.log')
fomatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

logger.setLevel(logging.INFO)

router = APIRouter()

@router.post(
        "/classify",
        response_model=ClassifyResponse,
        summary="Classify a room image",
        description=(
        "Upload an image of a home room. The vision model will predict the room type "
        "and return confidence scores for all supported classes."
        ),
    )
def classify_room() -> ClassifyResponse:
    """Send post request to classify an image"""
    # Validate content type


    # Read and validate file size


    # Run the classification


    # Return response
    return ClassifyResponse()