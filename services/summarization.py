
import os
import time

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def summarize_meeting(transcript_text):

    if not transcript_text or not transcript_text.strip():
        return "No meeting transcript available."

    prompt = f"""
You are an AI meeting assistant.

Analyze the following meeting transcript.

Generate these four sections:

1. Meeting Summary
2. Key Discussion Points
3. Decisions Made
4. Action Items

Instructions:
- If a person is assigned a task, mention their name.
- Do not invent information.
- Use only the information in the transcript.
- If a section has no information, say "Not mentioned".
- Use clear headings and bullet points.
- Keep the summary informative and concise.

MEETING TRANSCRIPT:
{transcript_text}
"""

    for attempt in range(3):

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=2000
            )

            result = response.choices[0].message.content

            return result.strip() if result else (
                "AI could not generate a summary."
            )

        except Exception as error:

            print("Groq Summarization Error:", repr(error))

            if attempt < 2 and (
                "429" in str(error) or
                "500" in str(error) or
                "502" in str(error) or
                "503" in str(error) or
                "UNAVAILABLE" in str(error)
            ):
                print("Groq is temporarily unavailable. Retrying...")
                time.sleep(5 * (attempt + 1))

            else:
                return (
                    "AI summarization is temporarily unavailable. "
                    "Please try again later."
                )
