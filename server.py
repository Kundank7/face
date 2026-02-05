from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import base64
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# Gemini API Key Rotation Logic
class GeminiKeyRotator:
    def __init__(self):
        keys_str = os.environ.get('GEMINI_API_KEYS', '')
        self.keys = [k.strip() for k in keys_str.split(',') if k.strip()]
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

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
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

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Face Shape Detector API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

@api_router.post("/analyze", response_model=FaceShapeResponse)
async def analyze_face(
    image: UploadFile = File(...),
    gender: str = Form(...)
):
    try:
        # Validate file size (15MB limit)
        contents = await image.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > 15:
            raise HTTPException(status_code=400, detail="Image size must be less than 15MB")
        
        # Convert image to base64
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Prepare prompt for Gemini
        prompt = f"""Analyze this face image and determine the face shape. Consider these face shape categories:

1. Oval: Balanced proportions, slightly longer than wide, rounded jawline
2. Round: Width and length similar, soft curves, fuller cheeks
3. Square: Strong angular jawline, forehead and jaw similar width
4. Heart: Wider forehead, narrow pointed chin
5. Oblong: Face much longer than wide, straight sides
6. Diamond: Wide cheekbones, narrow forehead and jaw

Gender context: {gender}

Provide your analysis in this exact JSON format:
{{
  "primaryFaceShape": "[shape name]",
  "confidence": [percentage as decimal, e.g., 76.83],
  "secondaryMatches": [
    {{"shape": "[shape name]", "confidence": [percentage]}},
    {{"shape": "[shape name]", "confidence": [percentage]}},
    {{"shape": "[shape name]", "confidence": [percentage]}}
  ]
}}

Ensure all 5 remaining shapes appear in secondaryMatches, sorted by confidence descending."""
        
        # Attempt analysis with key rotation
        max_retries = len(key_rotator.keys)
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                api_key = key_rotator.get_current_key()
                
                # Create Gemini chat instance
                chat = LlmChat(
                    api_key=api_key,
                    session_id=f"face-analysis-{uuid.uuid4()}",
                    system_message="You are an expert at analyzing facial features and determining face shapes. Always respond with valid JSON."
                )
                chat.with_model("gemini", "gemini-2.5-flash-lite")
                
                # Create image content
                image_content = ImageContent(image_base64=image_base64)
                
                # Send message with image
                user_message = UserMessage(
                    text=prompt,
                    file_contents=[image_content]
                )
                
                # Add artificial delay for premium feel (2-3 seconds)
                await asyncio.sleep(2.5)
                
                # Get response
                response = await chat.send_message(user_message)
                
                # Parse response
                import json
                
                # Extract JSON from response
                response_text = response.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response_text)
                
                # Validate and return
                return FaceShapeResponse(
                    primaryFaceShape=result['primaryFaceShape'],
                    confidence=float(result['confidence']),
                    secondaryMatches=[
                        FaceShapeMatch(shape=m['shape'], confidence=float(m['confidence']))
                        for m in result['secondaryMatches']
                    ]
                )
                
            except Exception as e:
                last_error = e
                logging.error(f"Attempt {attempt + 1} failed with key index {key_rotator.current_index}: {str(e)}")
                
                # Try rotating to next key
                if key_rotator.rotate_key():
                    attempt += 1
                    continue
                else:
                    # No more keys to try
                    break
        
        # All attempts failed
        raise HTTPException(
            status_code=500,
            detail=f"Face analysis failed after trying all API keys. Last error: {str(last_error)}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in analyze_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
