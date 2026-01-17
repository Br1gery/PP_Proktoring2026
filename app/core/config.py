import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "Proctoring System"
    VERSION = "1.0.0"
    
    # Database settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "proctoring_db")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    
    # JWT settings
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # App settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    WS_MAX_SIZE = 10 * 1024 * 1024
    
    YOLO_MODEL_PATH = "yolo11s.pt"

settings = Settings()