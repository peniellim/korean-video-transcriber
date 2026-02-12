from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

# Create FastAPI app
app = FastAPI()

# Enable CORS (must be AFTER creating app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    # Save uploaded video temporarily
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Load OpenAI client using environment variable from Render
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Send video file to Whisper API
    with open(temp_path, "rb") as audio:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json"
        )

    # Return transcription + segments with timestamps
    return {
        "text": result.text,
        "captions": result.segments
    }
