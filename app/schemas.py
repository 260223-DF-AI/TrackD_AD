"""
Pydantic schemas for API request and response models
"""

from pydantic import BaseModel, Field
from enum import Enum

class RoomType(str, Enum):
    """Supported room types for classification:
    A room can be basic, old, or renovated.
    The rooms are bathroom, bedroom, exterior, kitchen, and living_room.
    """
    bathroom = "bathroom"
    bedroom = "bedroom"
    exterior = "exterior"
    kitchen = "kitchen"
    living_room = "living_room"

class Condition(str, Enum):
    """Condition labels for rooms: basic, old, or renovated."""
    basic = "basic"
    old = "old"
    renovated = "renovated"


class RoomLabel(BaseModel):
    """Combined labels for a room: its type and its condition."""
    room_type: RoomType = Field(..., description="Type of room (bathroom, bedroom, etc.)")
    condition: Condition = Field(..., description="Condition label (basic, old, renovated)")

class ClassifyResponse(BaseModel):
    predicted_class: str = Field(..., description="Top predicted room type")
    confidence: float = Field(..., ge=0.0, le= 1.0, description="Confidence score [0-1]")
    below_threshold: bool = Field(
        False,
        description="True if confidence < CONFIDENCE_THRESHOLD : treat result with caution"
    )