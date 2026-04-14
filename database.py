"""
Database layer — SQLite + SQLAlchemy
Tables: users, api_keys, predictions, rate_limit_logs
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    Boolean, DateTime, ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./visionapi.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    api_keys = relationship("APIKey", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")


class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tier = Column(String(50), default="free")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="api_keys")
    rate_logs = relationship("RateLimitLog", back_populates="api_key")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    label = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    class_index = Column(Integer, nullable=False)
    filename = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")


class RateLimitLog(Base):
    __tablename__ = "rate_limit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    api_key = relationship("APIKey", back_populates="rate_logs")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
