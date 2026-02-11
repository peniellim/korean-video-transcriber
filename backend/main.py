from fastapi import FastAPI, UploadFile
import openai

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with open("temp.mp4", "wb") as f:
        f.write(await file.read())

    client = openai.OpenAI()

    audio_file = open("temp.mp4", "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json"
    )

    return {
        "full_text": transcript["text"],
        "captions": transcript["segments"]
    }
