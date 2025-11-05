import os
from dotenv import load_dotenv
# Load environment variables from .env file (local) or GCP Secret Manager (production)
load_dotenv()
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    flash,
    jsonify,
    Response,
    stream_with_context,
)
import uuid
import io
from werkzeug.utils import secure_filename
from utils import gemini_check_is_wound, ensure_dirs
import json
import time

import tensorflow as tf
import config
from utils import (
    allowed_file,
    handle_upload_and_process_with_progress,
    get_model,
    get_progress,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

# expose config in templates (footer info)
app.jinja_env.globals.update(config=config)

# Version note
if tf.__version__ != config.REQUIRED_TF:
    print(f"[WARN] TensorFlow {config.REQUIRED_TF} required; found {tf.__version__}.")

# ⚡ PRELOAD MODEL AT STARTUP - This makes first prediction MUCH faster!
if getattr(config, "PRELOAD_MODEL", False):
    print("=" * 60)
    print("🔥 Preloading model at startup...")
    print("=" * 60)
    try:
        model = get_model()
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total params: {model.count_params():,}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Failed to preload model: {e}")
        print("=" * 60)
else:
    print(
        "Model preload disabled (PRELOAD_MODEL=False). Will run Gemini triage before loading model."
    )


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", result=None)


@app.route("/about", methods=["GET"])
def about():
    """About page route"""
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    """Contact page route"""
    if request.method == "POST":
        # Get form data
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        # Here you could add logic to send emails, store in database, etc.
        # For now, we'll just acknowledge the submission
        print(f"Contact form submitted:")
        print(f"  Name: {name}")
        print(f"  Email: {email}")
        print(f"  Subject: {subject}")
        print(f"  Message: {message}")

        return render_template("contact.html", message_sent=True)

    return render_template("contact.html", message_sent=False)


@app.route("/triage", methods=["POST"])
def triage():
    """Lightweight endpoint: accept an image, run the Gemini triage only, and
    return JSON {is_wound: bool, message: str, input_image: str}.
    This endpoint MUST NOT load the heavy model.
    """
    if "image" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    ensure_dirs()
    # sanitize name and write to upload folder
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    unique_name = f"input_{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(config.UPLOAD_FOLDER, unique_name)
    file.save(save_path)
    save_path = save_path.replace("\\", "/")

    # Run only the LLM triage (fast-fail). gemini_check_is_wound will handle
    # missing/invalid keys and exceptions by returning (True, message).
    is_wound, msg = gemini_check_is_wound(save_path)
    return jsonify(
        {"is_wound": bool(is_wound), "message": msg, "input_image": save_path}
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """Process uploaded image and return JSON result"""
    if "image" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    def progress_callback(step_message):
        """Callback to track progress"""
        # For now we don't stream progress to the client here, but this
        # callback is preserved for future use. Keep it minimal.
        pass

    # Measure wall-clock time for the entire analyze pipeline (from click -> pipeline done)
    start = time.perf_counter()
    result = handle_upload_and_process_with_progress(file, progress_callback)
    elapsed = time.perf_counter() - start

    # Add timing info to the response so the frontend can display it
    result["elapsed_seconds"] = float(f"{elapsed:.3f}")
    result["elapsed_human"] = f"{elapsed:.2f}s"

    # Print a clear terminal message when the entire pipeline completes
    print(
        """
------------------------------------------------------------
✅ ANALYZE PIPELINE FINISHED
  Input: %s
  Status: %s
  Elapsed time: %0.3fs
------------------------------------------------------------
"""
        % (
            result.get("input_image", "<unknown>"),
            result.get("status", "<no-status>"),
            elapsed,
        )
    )

    # Return result directly as JSON instead of storing in session
    return jsonify(result)


@app.route("/progress", methods=["GET"])
def progress():
    """Get current progress status"""
    return jsonify({"progress": get_progress()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
