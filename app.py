
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    url_for,
    jsonify
)

import os
import time
import uuid
import traceback
from werkzeug.utils import secure_filename

from services.transcription import transcribe_audio
from services.summarization import summarize_meeting
from services.qa import answer_question
from services.project_analyzer import analyze_project


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "downloads"

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# Check allowed audio formats
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# Generate meeting summary using Gemini
def generate_summary(transcript_text):

    for attempt in range(3):

        try:
            summary = summarize_meeting(transcript_text)
            return summary

        except Exception as error:

            error_message = str(error)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ) and attempt < 2:

                print("Gemini is busy. Retrying...")
                time.sleep(5 * (attempt + 1))

            else:

                print("Gemini API Error:", error)

                return (
                    "AI summarization is temporarily unavailable. "
                    "Please try again later."
                )


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Upload and process meeting audio
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("meeting_file")

    if not file or file.filename == "":
        return "Please select a file"

    if not allowed_file(file.filename):
        return "Invalid file format"

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        # Step 1: Transcribe meeting audio
        transcript = transcribe_audio(filepath)

        # Step 2: Convert transcript segments into text
        transcript_text = "\n".join(
            segment["text"] for segment in transcript
        )

        # Step 3: Generate AI meeting summary
        summary = generate_summary(transcript_text)

        # Create unique filenames
        file_id = uuid.uuid4().hex

        transcript_filename = f"transcript_{file_id}.txt"
        summary_filename = f"summary_{file_id}.txt"

        # Save transcript
        with open(
            os.path.join(DOWNLOAD_FOLDER, transcript_filename),
            "w",
            encoding="utf-8"
        ) as f:

            for segment in transcript:
                f.write(
                    f"[{segment['start']:.2f}s] {segment['text']}\n"
                )

        # Save summary
        with open(
            os.path.join(DOWNLOAD_FOLDER, summary_filename),
            "w",
            encoding="utf-8"
        ) as f:

            f.write(summary)

    except Exception as error:

        print("Meeting processing error:", error)

        return "Unable to process this meeting. Please try again."

    # Step 4: Generate audio URL
    audio_url = url_for(
        "uploaded_audio",
        filename=filename
    )

    # Step 5: Display results
    return render_template(
        "transcript.html",
        transcript=transcript,
        audio_url=audio_url,
        summary=summary,
        transcript_filename=transcript_filename,
        summary_filename=summary_filename,
        transcript_text=transcript_text
    )


# AI Project Workflow Analyzer
@app.route("/analyze-project", methods=["POST"])
def analyze_project_route():

    project_file = request.files.get("project_file")

    if not project_file or not project_file.filename:
        return "Please choose a project ZIP file.", 400

    filename = secure_filename(project_file.filename)
    if not filename.lower().endswith(".zip"):
        return "Please upload your project as a .zip file.", 400

    try:
        analysis = analyze_project(project_file.stream, filename)
        return render_template("project_workflow.html", analysis=analysis)

    except ValueError as error:
        return str(error), 400

    except Exception as error:
        print("Project analysis error:", repr(error))
        return (
            "Unable to analyse this project right now. Check the server logs and try again.",
            500
        )


@app.errorhandler(413)
def upload_too_large(_error):
    return "The uploaded file is too large. Please keep the ZIP under 150 MB.", 413


# AI Meeting Assistant - Question and Answer
@app.route("/ask", methods=["POST"])
def ask_question():

    question = request.form.get("question", "").strip()
    transcript_text = request.form.get("transcript", "").strip()

    if not transcript_text:
        return {
            "answer": "Meeting transcript is missing."
        }, 400

    if not question:
        return {
            "answer": "Please enter a question."
        }, 400

    try:

        answer = answer_question(
            transcript_text,
            question
        )

        return {
            "answer": answer
        }

    except Exception as error:

        print("========== Q&A ERROR ==========")
        print(repr(error))
        traceback.print_exc()
        print("================================")

        return {
            "answer": "Sorry, I couldn't process your question. Please try again."
        }, 500

# Play uploaded meeting audio
@app.route("/audio/<filename>")
def uploaded_audio(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=False
    )


# Download transcript and summary
@app.route("/download/<filename>")
def download_file(filename):

    filepath = os.path.join(DOWNLOAD_FOLDER, filename)

    print("DOWNLOAD REQUEST:", filepath)
    print("FILE EXISTS:", os.path.exists(filepath))

    return send_from_directory(
        DOWNLOAD_FOLDER,
        filename,
        as_attachment=True,
        download_name=filename
    )

@app.route("/save-live-recording", methods=["POST"])
def save_live_recording():

    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "message": "No audio file received."
        }), 400

    audio_file = request.files["audio"]

    if not audio_file.filename:
        return jsonify({
            "success": False,
            "message": "Empty audio file."
        }), 400

    # Generate unique recording filename
    file_id = uuid.uuid4().hex
    filename = f"live_meeting_{file_id}.webm"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        # Step 1: Save live meeting audio
        audio_file.save(filepath)

        print("Live recording saved:", filepath)

        # Step 2: Transcribe using Faster-Whisper
        print("Starting live meeting transcription...")

        transcript = transcribe_audio(filepath)

        # Step 3: Convert transcript into text
        transcript_text = "\n".join(
            segment["text"] for segment in transcript
        )

        if not transcript_text.strip():
            return jsonify({
                "success": False,
                "message": "No speech detected in the recording."
            }), 400

        # Step 4: Generate Groq AI summary
        print("Generating meeting summary...")

        summary = generate_summary(transcript_text)

        # Step 5: Create download filenames
        transcript_filename = f"transcript_{file_id}.txt"
        summary_filename = f"summary_{file_id}.txt"

        # Step 6: Save transcript
        with open(
            os.path.join(DOWNLOAD_FOLDER, transcript_filename),
            "w",
            encoding="utf-8"
        ) as f:

            for segment in transcript:
                f.write(
                    f"[{segment['start']:.2f}s] {segment['text']}\n"
                )

        # Step 7: Save summary
        with open(
            os.path.join(DOWNLOAD_FOLDER, summary_filename),
            "w",
            encoding="utf-8"
        ) as f:

            f.write(summary)

        # Step 8: Generate audio URL
        audio_url = url_for(
            "uploaded_audio",
            filename=filename
        )

        print("Live meeting processing completed!")

        # Step 9: Display transcript and summary
        return render_template(
            "transcript.html",
            transcript=transcript,
            audio_url=audio_url,
            summary=summary,
            transcript_filename=transcript_filename,
            summary_filename=summary_filename,
            transcript_text=transcript_text
        )

    except Exception as error:

        print("Live meeting processing error:", repr(error))
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": "Unable to process live meeting. Check Flask terminal."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)