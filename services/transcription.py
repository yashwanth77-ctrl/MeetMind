
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")

    client = Groq(api_key=api_key)

    with open(audio_path, "rb") as audio_file:

        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_file),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    transcript = []

    for segment in result.segments or []:

        if isinstance(segment, dict):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
        else:
            start = segment.start
            end = segment.end
            text = segment.text

        transcript.append({
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "text": text.strip()
        })

    return transcript
