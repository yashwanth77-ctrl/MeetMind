
import os
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTION_PROVIDER = os.getenv(
    "TRANSCRIPTION_PROVIDER", "local"
).lower()


def transcribe_audio(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # =====================================
    # GROQ CLOUD TRANSCRIPTION
    # =====================================
    if TRANSCRIPTION_PROVIDER == "groq":

        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Check Render environment variables."
            )

        client = Groq(api_key=api_key)

        with open(audio_path, "rb") as audio_file:

            result = client.audio.transcriptions.create(
                file=(
                    os.path.basename(audio_path),
                    audio_file
                ),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        transcript = []

        for segment in result.segments or []:

            # Groq returns segment dictionaries
            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]

            # Also support object-style responses
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

    # =====================================
    # LOCAL FASTER-WHISPER TRANSCRIPTION
    # =====================================
    elif TRANSCRIPTION_PROVIDER == "local":

        from faster_whisper import WhisperModel

        model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

        segments, info = model.transcribe(
            audio_path,
            beam_size=5
        )

        transcript = []

        for segment in segments:

            transcript.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment.text.strip()
            })

        return transcript

    else:
        raise ValueError(
            f"Unsupported transcription provider: {TRANSCRIPTION_PROVIDER}"
        )
