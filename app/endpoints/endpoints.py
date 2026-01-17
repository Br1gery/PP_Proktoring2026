from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt
import bcrypt
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from core.database import get_db
from core.models import User, Session as DBSession, Token as DBToken
from PYDmodels.schemas import UserCreate, UserLogin, Token, UserResponse, SessionResponse, StatusResponse
from services.proctor_service import ProctorService

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()
proctor_service = ProctorService()
active_connections = {}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    from core.config import settings
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    from core.config import settings
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    
    # Check token in database
    db_token = db.query(DBToken).filter(DBToken.token == token).first()
    if not db_token or datetime.utcnow() > db_token.expires_at:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    return user

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"User registered: {user.username}")
    
    return {
        "id": db_user.id,
        "username": db_user.username,
        "created_at": db_user.created_at
    }

@router.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Find user
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create token
    access_token = create_access_token(data={"sub": user.username})
    
    # Store token in database
    from core.config import settings
    expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    db_token = DBToken(
        token=access_token,
        username=user.username,
        expires_at=expires_at
    )
    db.add(db_token)
    db.commit()
    
    logger.info(f"User logged in: {user.username}")
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    # Remove token from database
    db.query(DBToken).filter(DBToken.token == token).delete()
    db.commit()
    
    logger.info(f"User logged out: {current_user.username}")
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "created_at": current_user.created_at
    }

@router.post("/sessions/create", response_model=SessionResponse)
async def create_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session_id = proctor_service.create_session(current_user.id)
    
    # Store session in database
    db_session = DBSession(
        id=session_id,
        user_id=current_user.id,
        status="active"
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    logger.info(f"Session created: {session_id} for user {current_user.username}")
    
    return {
        "session_id": session_id,
        "user_id": current_user.id,
        "status": "active",
        "created_at": datetime.utcnow()
    }

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find session in database
    db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if db_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this session")
    
    # Delete from proctor service
    proctor_service.delete_session(session_id)
    
    # Delete from database
    db.delete(db_session)
    db.commit()
    
    logger.info(f"Session deleted: {session_id}")
    
    return {"message": "Session deleted"}

@router.get("/sessions/{session_id}/status", response_model=StatusResponse)
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find session in database
    db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if db_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    session = proctor_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session data not found")
    
    return {
        "session_id": session_id,
        "user_id": current_user.id,
        "current_state": session.get("current_state", {}),
        "warnings": session.get("warnings", []),
        "alerts": session.get("alerts", []),
        "frame_count": session.get("frame_count", 0),
        "timestamp": datetime.utcnow()
    }

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    await websocket.accept()
    
    # Check session in database
    db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not db_session:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connected for session: {session_id}")
    
    try:
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                message = json.loads(data["text"])
                
                if message["type"] == "frame":
                    frame_data = message.get("data")
                    calibration_requested = message.get("calibration", False)
                    
                    if frame_data:
                        result = await proctor_service.process_frame(
                            frame_data, 
                            session_id, 
                            calibration_requested=calibration_requested
                        )
                        
                        if result:
                            await websocket.send_json({
                                "type": "status_update",
                                "data": result,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                
                elif message["type"] == "command":
                    command = message.get("command")
                    if command == "calibrate":
                        proctor_service.calibrate(session_id)
                        await websocket.send_json({
                            "type": "calibration_started",
                            "message": "Calibration process started. Look straight at the camera."
                        })
                    
                    elif command == "toggle_adaptive":
                        proctor_service.ADAPTIVE_CALIBRATION_ENABLED = not proctor_service.ADAPTIVE_CALIBRATION_ENABLED
                        status = "enabled" if proctor_service.ADAPTIVE_CALIBRATION_ENABLED else "disabled"
                        await websocket.send_json({
                            "type": "settings_updated",
                            "setting": "adaptive_calibration",
                            "value": status
                        })
                    
                    elif command == "toggle_yolo":
                        proctor_service.yolo_verification_enabled = not proctor_service.yolo_verification_enabled
                        status = "enabled" if proctor_service.yolo_verification_enabled else "disabled"
                        await websocket.send_json({
                            "type": "settings_updated",
                            "setting": "yolo_verification",
                            "value": status
                        })
                
                elif message["type"] == "calibration_data":
                    proctor_service.set_calibration_data(session_id, message.get("data", {}))
                    await websocket.send_json({
                        "type": "calibration_complete",
                        "message": "Calibration data received"
                    })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
    finally:
        if session_id in active_connections:
            del active_connections[session_id]

@router.get("/")
async def root():
    from core.config import settings
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "endpoints": {
            "register": "/register (POST)",
            "login": "/login (POST)",
            "logout": "/logout (POST)",
            "me": "/me (GET)",
            "create_session": "/sessions/create (POST)",
            "websocket": "/ws/{session_id} (WebSocket)"
        }
    }

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    users_count = db.query(User).count()
    sessions_count = db.query(DBSession).count()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "users_count": users_count,
        "sessions_count": sessions_count,
        "active_connections": len(active_connections)
    }