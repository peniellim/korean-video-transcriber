from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile):

    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(temp_path, "rb") as audio:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json"
        )

    return {
        "text": result["text"],
        "captions": result["segments"]
    }
