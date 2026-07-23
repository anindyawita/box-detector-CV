from pydantic import BaseModel

class AnalyzeResponse(BaseModel):
    inside_box: int
    outside_box: int
    total: int
    overlay_image_base64: str
    processing_time_ms: float
