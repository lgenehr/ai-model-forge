from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DatasetMeta(BaseModel):
    source: str
    url: Optional[str] = None
    date: str  # ISO format YYYY-MM-DD
    language: str = "en"
    topics: List[str] = []
    entities: List[str] = []

class DatasetRow(BaseModel):
    instruction: str
    input: str
    output: str
    meta: DatasetMeta
