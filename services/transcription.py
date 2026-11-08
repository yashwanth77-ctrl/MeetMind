
import os

TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "local").lower()


def transcribe_audio(audio_path):

    # Cloud transcription using Groq
    if TRANSCRIPTION_PROVIDER == "groq":
        from groq import Groq

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        with open(audio_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        transcript = []

        for segment in result.segments:
            transcript.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        return transcript

    # Local transcription using Faster-Whisper
    else:
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
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        return transcript
