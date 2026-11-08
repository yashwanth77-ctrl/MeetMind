
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY was not found in .env")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def answer_question(transcript_text, question):

    if not transcript_text or not transcript_text.strip():
        return "No meeting transcript is available."

    if not question or not question.strip():
        return "Please enter a question."

    prompt = f"""
You are MeetMind, an AI meeting assistant.

Answer the user's question using ONLY the meeting
transcript provided below.

RULES:
1. Never use outside knowledge.
2. Never invent names, facts, decisions or deadlines.
3. If the answer is unavailable, say it was not
   mentioned in the meeting.
4. Give clear and concise answers.
5. Use bullet points when appropriate.
6. Answer in a natural conversational style.

MEETING TRANSCRIPT:
{transcript_text}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=1000
    )

    answer = response.choices[0].message.content

    return answer.strip() if answer else "I couldn't generate an answer."
