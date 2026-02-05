from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from google import genai
from google.genai import types

import os
import logging
import base64
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone
import json

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get("DB_NAME", "test_database")]

# --------------------------------------------------
# GEMINI KEY ROTATION
# --------------------------------------------------

class GeminiKeyRotator:
    def __init__(self):
        keys_str = os.environ.get("GEMINI_API_KEYS", "")
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        self.current_index = 0

    def get_current_key(self):
        if not self.keys:
            raise ValueError("No Gemini API keys configured")
        return self.keys[self.current_index]

    def rotate_key(self):
        if len(self.keys) > 1:
            self.current_index = (self.current_index + 1) % len(self.keys)
            return True
        return False


key_rotator = GeminiKeyRotator()

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI()
api_router = APIRouter(prefix="/api")

# --------------------------------------------------
# MODELS
# --------------------------------------------------

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class FaceShapeMatch(BaseModel):
    shape: str
    confidence: float


class FaceShapeResponse(BaseModel):
    primaryFaceShape: str
    confidence: float
    secondaryMatches: List[FaceShapeMatch]

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@api_router.get("/")
async def root():
    return {"message": "Face Shape Detector API"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(client_name=input.client_name)

    doc = status_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()

    await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    data = await db.status_checks.find({}, {"_id": 0}).to_list(1000)

    for item in data:
        if isinstance(item["timestamp"], str):
            item["timestamp"] = datetime.fromisoformat(item["timestamp"])

    return data


@api_router.post("/analyze", response_model=FaceShapeResponse)
async def analyze_face(
    image: UploadFile = File(...),
    gender: str = Form(...)
):
    try:
        # -----------------------------
        # IMAGE VALIDATION
        # -----------------------------
        contents = await image.read()
        if len(contents) / (1024 * 1024) > 15:
            raise HTTPException(status_code=400, detail="Image must be under 15MB")

        # -----------------------------
        # PROMPT
        # -----------------------------
        prompt = f"""
Analyze this face image and determine the face shape.

Face shape categories:
1. Oval
2. Round
3. Square
4. Heart
5. Oblong
6. Diamond

Gender context: {gender}

Return ONLY valid JSON in this format:
{{
  "primaryFaceShape": "shape",
  "confidence": 76.83,
  "secondaryMatches": [
    {{ "shape": "shape", "confidence": 65.2 }},
    {{ "shape": "shape", "confidence": 54.1 }},
    {{ "shape": "shape", "confidence": 42.7 }},
    {{ "shape": "shape", "confidence": 30.4 }},
    {{ "shape": "shape", "confidence": 18.9 }}
  ]
}}
"""

        max_retries = len(key_rotator.keys)
        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                api_key = key_rotator.get_current_key()
                client_genai = genai.Client(api_key=api_key)

                await asyncio.sleep(2.5)  # artificial delay

                response = client_genai.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[
                        prompt,
                        types.Part.from_bytes(
                            data=contents,
                            mime_type=image.content_type
                        ),
                    ],
                )

                text = response.text.strip()

                if "```" in text:
                    text = text.split("```")[1].replace("json", "").strip()

                result = json.loads(text)

                return FaceShapeResponse(
                    primaryFaceShape=result["primaryFaceShape"],
                    confidence=float(result["confidence"]),
                    secondaryMatches=[
                        FaceShapeMatch(
                            shape=m["shape"],
                            confidence=float(m["confidence"])
                        )
                        for m in result["secondaryMatches"]
                    ],
                )

            except Exception as e:
                last_error = e
                logging.error(f"Gemini error (key {key_rotator.current_index}): {e}")
                if key_rotator.rotate_key():
                    attempt += 1
                else:
                    break

        raise HTTPException(
            status_code=500,
            detail=f"Face analysis failed. Last error: {last_error}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
