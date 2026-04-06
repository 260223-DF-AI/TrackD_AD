"""
Pydantic schemas for API request and response models
"""

from pydantic import BaseModel, Field
from enum import Enum

class RoomType(str, Enum):
    bedroom = "bedroom"
    kitchen = "kitchen"
    basement = "basement"
    living_room = "living_room"

class ClassifyResponse(BaseModel):
    predicted_class: str = Field(..., description="Top predicted room type")
    confidence: float = Field(..., ge=0.0, le= 1.0, description="Confidence score [0-1]")
    below_threshold: bool = Field(
        False,
        description="True if confidence < CONFIDENCE_THRESHOLD : treat result with caution"
    )