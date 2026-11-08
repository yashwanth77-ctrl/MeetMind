
import os
import wave
import uuid

LIVE_FOLDER = "live_recordings"

os.makedirs(LIVE_FOLDER, exist_ok=True)


def save_audio_chunk(audio_data, meeting_id):
    """
    Save an incoming browser audio chunk.
    """
    filename = f"{meeting_id}_{uuid.uuid4().hex}.webm"
    filepath = os.path.join(LIVE_FOLDER, filename)

    with open(filepath, "wb") as audio_file:
        audio_file.write(audio_data)

    return filepath
