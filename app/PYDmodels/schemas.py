# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    user_id: int
    status: str
    created_at: datetime

class StatusResponse(BaseModel):
    session_id: str
    user_id: int
    current_state: Dict[str, int]
    warnings: list
    alerts: list
    frame_count: int
    timestamp: datetime

class ProctoringData(BaseModel):
    gaze_vertical: float
    gaze_horizontal: float
    head_vertical: float
    head_horizontal: float
    status: str