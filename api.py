"""
VisionAPI — FastAPI Backend
Run: uvicorn api:app --reload --port 8000
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
import jwt

from database import init_db, get_db, User, APIKey, Prediction, RateLimitLog

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-" + secrets.token_hex(16))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
def find_model_path():
    """Auto-detect model file in current directory or models/ folder."""
    search_paths = [
        os.getenv("MODEL_PATH", ""),
        "cat_dog_classifier_final.keras",
        "model.keras",
        "model.h5",
        "models/model.keras",
        "models/model.h5",
        "models/cat_dog_classifier_final.keras",
    ]
    for path in search_paths:
        if path and os.path.exists(path):
            return path
    # Also search for any .keras or .h5 file in current dir
    for f in os.listdir("."):
        if f.endswith(".keras") or f.endswith(".h5"):
            return f
    return "models/model.h5"  # fallback

MODEL_PATH = find_model_path()

RATE_LIMITS = {
    "free": 50,
    "starter": 500,
    "business": 5000,
    "enterprise": 50000,
}

app = FastAPI(title="VisionAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ Database initialized (SQLite)")
    print(f"📁 Model path: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")


# ---------------------------------------------------------------------------
# ML MODEL
# ---------------------------------------------------------------------------
_model_cache = {}


def get_model():
    if "model" not in _model_cache:
        try:
            import tensorflow as tf
            _model_cache["model"] = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Model not found, using demo mode. ({e})")
            _model_cache["model"] = None
    return _model_cache["model"]


def predict_image(image_bytes: bytes, model) -> dict:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Auto-detect image size from model, fallback to 128
    if model is not None:
        try:
            input_shape = model.input_shape
            img_size = (input_shape[1], input_shape[2])  # (height, width)
        except Exception:
            img_size = (128, 128)
    else:
        img_size = (128, 128)

    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model is not None:
        preds = model.predict(img_array)
        labels = os.getenv("CLASS_LABELS", "cat,dog").split(",")

        # Handle both sigmoid (1 output) and softmax (2+ outputs)
        if preds.shape[-1] == 1:
            # Sigmoid output: 0 = first class, 1 = second class
            prob = float(preds[0][0])
            if prob >= 0.5:
                class_idx = 1
                confidence = prob
            else:
                class_idx = 0
                confidence = 1.0 - prob
        else:
            # Softmax output
            class_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

        label = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"
        print(f"✅ Prediction: {label} ({confidence:.1%})")
    else:
        # DEMO MODE — simple heuristic, replace with real model
        avg_r = float(np.mean(img_array[0, :, :, 0]))
        avg_g = float(np.mean(img_array[0, :, :, 1]))
        avg_b = float(np.mean(img_array[0, :, :, 2]))
        demo_labels = os.getenv("CLASS_LABELS", "cat,dog").split(",")
        score = (avg_r * 0.6 + avg_g * 0.3 - avg_b * 0.1)
        class_idx = 0 if score > 0.35 else 1
        class_idx = min(class_idx, len(demo_labels) - 1)
        label = demo_labels[class_idx]
        confidence = round(0.80 + abs(score - 0.35) * 0.3, 4)
        confidence = min(confidence, 0.99)
        print(f"⚠️  DEMO: {label} ({confidence:.1%}) — load .h5 model for real results")

    return {"label": label, "confidence": confidence, "class_index": class_idx}


# ---------------------------------------------------------------------------
# AUTH HELPERS
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def check_rate_limit(api_key_row: APIKey, db: Session) -> bool:
    limit = RATE_LIMITS.get(api_key_row.tier, 50)
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    count = (
        db.query(func.count(RateLimitLog.id))
        .filter(RateLimitLog.api_key_id == api_key_row.id, RateLimitLog.timestamp >= one_hour_ago)
        .scalar()
    )
    if count >= limit:
        return False
    db.add(RateLimitLog(api_key_id=api_key_row.id))
    db.commit()
    return True


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class APIKeyCreate(BaseModel):
    tier: str = "free"


# ---------------------------------------------------------------------------
# AUTH ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/auth/signup", tags=["Auth"])
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=req.email, name=req.name, password_hash=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": str(user.id)})
    return {"message": "Account created", "access_token": token, "token_type": "bearer"}


@app.post("/api/auth/login", tags=["Auth"])
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or user.password_hash != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me", tags=["Auth"])
def get_me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at.isoformat()}


# ---------------------------------------------------------------------------
# API KEY ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/keys/generate", tags=["API Keys"])
def generate_api_key(req: APIKeyCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if req.tier not in RATE_LIMITS:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Choose: {list(RATE_LIMITS.keys())}")
    raw_key = f"vapi_{secrets.token_hex(24)}"
    api_key = APIKey(key=raw_key, user_id=user.id, tier=req.tier)
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    return {"api_key": raw_key, "tier": req.tier, "rate_limit": f"{RATE_LIMITS[req.tier]} requests/hour"}


@app.get("/api/keys/list", tags=["API Keys"])
def list_api_keys(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    keys = db.query(APIKey).filter(APIKey.user_id == user.id).all()
    return {
        "keys": [
            {
                "api_key": k.key[:12] + "..." + k.key[-6:],
                "full_key": k.key,
                "tier": k.tier,
                "is_active": k.is_active,
                "created_at": k.created_at.isoformat(),
            }
            for k in keys
        ]
    }


@app.delete("/api/keys/{api_key}", tags=["API Keys"])
def revoke_api_key(api_key: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    key_row = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not key_row:
        raise HTTPException(status_code=404, detail="API key not found")
    if key_row.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your API key")
    key_row.is_active = False
    db.commit()
    return {"message": "API key revoked"}


# ---------------------------------------------------------------------------
# PREDICTION — JWT (Streamlit UI)
# ---------------------------------------------------------------------------
@app.post("/api/predict", tags=["Prediction"])
async def predict_authenticated(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    model = get_model()
    result = predict_image(image_bytes, model)
    pred = Prediction(user_id=user.id, label=result["label"], confidence=result["confidence"], class_index=result["class_index"], filename=file.filename or "")
    db.add(pred)
    db.commit()
    return {**result, "timestamp": pred.created_at.isoformat()}


# ---------------------------------------------------------------------------
# PREDICTION — API KEY (B2B clients)
# ---------------------------------------------------------------------------
@app.post("/api/v1/predict", tags=["Prediction (API Key)"])
async def predict_with_api_key(file: UploadFile = File(...), x_api_key: str = Header(..., alias="X-API-Key"), db: Session = Depends(get_db)):
    key_row = db.query(APIKey).filter(APIKey.key == x_api_key).first()
    if not key_row or not key_row.is_active:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    if not check_rate_limit(key_row, db):
        limit = RATE_LIMITS[key_row.tier]
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({limit} req/hr).")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    model = get_model()
    result = predict_image(image_bytes, model)
    pred = Prediction(user_id=key_row.user_id, label=result["label"], confidence=result["confidence"], class_index=result["class_index"], filename=file.filename or "")
    db.add(pred)
    db.commit()
    return {**result, "timestamp": pred.created_at.isoformat()}


# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
@app.get("/api/usage", tags=["Usage"])
def get_usage(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_predictions = db.query(func.count(Prediction.id)).filter(Prediction.user_id == user.id).scalar()
    recent = db.query(Prediction).filter(Prediction.user_id == user.id).order_by(Prediction.created_at.desc()).limit(10).all()
    api_keys_count = db.query(func.count(APIKey.id)).filter(APIKey.user_id == user.id, APIKey.is_active == True).scalar()
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    user_keys = db.query(APIKey).filter(APIKey.user_id == user.id).all()
    usage_per_key = {}
    for k in user_keys:
        count = db.query(func.count(RateLimitLog.id)).filter(RateLimitLog.api_key_id == k.id, RateLimitLog.timestamp >= one_hour_ago).scalar()
        usage_per_key[k.key[:12] + "..."] = count
    return {
        "total_predictions": total_predictions,
        "recent_predictions": [
            {"label": p.label, "confidence": p.confidence, "class_index": p.class_index, "filename": p.filename, "timestamp": p.created_at.isoformat()}
            for p in recent
        ],
        "api_keys_count": api_keys_count,
        "usage_per_key_last_hour": usage_per_key,
    }


@app.get("/api/health", tags=["System"])
def health():
    return {"status": "healthy", "database": "sqlite", "model_loaded": "model" in _model_cache, "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))