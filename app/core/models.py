from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="active")
    
    calibration_completed = Column(Boolean, default=False)
    calibration_vector = Column(JSON, nullable=True)
    head_calibration_completed = Column(Boolean, default=False)
    head_calibration_vector = Column(JSON, nullable=True)
    
    left_sphere_locked = Column(Boolean, default=False)
    right_sphere_locked = Column(Boolean, default=False)
    left_sphere_local_offset = Column(JSON, nullable=True)
    right_sphere_local_offset = Column(JSON, nullable=True)
    left_calibration_nose_scale = Column(Float, nullable=True)
    right_calibration_nose_scale = Column(Float, nullable=True)
    
    calibration_face_size = Column(Float, nullable=True)
    
    user = relationship("User", back_populates="sessions")

class Token(Base):
    __tablename__ = "tokens"
    
    token = Column(String, primary_key=True, index=True)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)