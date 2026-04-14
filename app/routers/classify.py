"""
Vision model endpoint.
Accepts an image upload and returns the predicted room type with confidence scores.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from app.schemas import ClassifyResponse
import logging
import boto3
import json
from DeployedModel import get_endpoint, create_image_payload

# implementing logger functionality
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('reporting.log')
fomatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

logger.setLevel(logging.INFO)

router = APIRouter()
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
ENDPOINT_NAME = get_endpoint()

@router.post(
        "/analyze",
        response_model=ClassifyResponse,
        summary="Classify a room image",
        description=(
        "Upload an image of a home room. The vision model will predict the room type "
        "and return confidence scores for all supported classes."
        ),
    )
def classify_room(image: UploadFile):
    """Send post request to classify an image"""
    # Validate content type
    if image.content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")

    # Run the classification
    payload = create_image_payload(image.file)
    response = sagemaker_runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="application/x-image", Body=payload)

    # Return response
    return response["Body"].read()