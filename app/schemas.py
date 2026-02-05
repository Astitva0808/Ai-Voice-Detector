from pydantic import BaseModel

class SuccessResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    model_version: str
    request_id: str | None = None

