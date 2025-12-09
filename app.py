# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tfidf_check import check_text
from PyPDF2 import PdfReader
from docx import Document

UPLOAD_FOLDER = "uploads"
ALLOWED = {"txt", "pdf", "docx"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "dev_secret_key"  # change for production

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def extract_text_from_file(path):
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == "pdf":
        text = []
        try:
            reader = PdfReader(path)
            for p in reader.pages:
                txt = p.extract_text()
                if txt:
                    text.append(txt)
        except Exception:
            pass
        return "\n".join(text)
    elif ext == "docx":
        try:
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    return ""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_text = ""

    if request.method == "POST":
        # Priority: uploaded file over raw textarea
        uploaded = request.files.get("file")
        if uploaded and uploaded.filename != "":
            if allowed_file(uploaded.filename):
                filename = secure_filename(uploaded.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                uploaded.save(save_path)
                user_text = extract_text_from_file(save_path)
            else:
                flash("Invalid file type.")
                return redirect(request.url)
        else:
            user_text = request.form.get("text", "").strip()

        if user_text:
            result = check_text(user_text)

    return render_template("index.html", result=result, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)
