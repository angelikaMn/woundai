import os

# Suppress TensorFlow logging messages but keep oneDNN optimizations for max performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING logs

import uuid
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.efficientnet import preprocess_input
import google.generativeai as genai
import config


# ---------------- helpers ----------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def _unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}.{ext}"

def ensure_dirs():
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

def format_gemini_markdown(text: str) -> str:
    """Convert Gemini markdown-style text to HTML"""
    import re
    
    # Replace bold (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace italic (*text*)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Replace headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Replace numbered lists (keep line breaks for CSS to handle)
    text = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap consecutive <li> in <ol>
    text = re.sub(r'(<li>.*?</li>\s*)+', lambda m: f'<ol>{m.group(0)}</ol>', text, flags=re.DOTALL)
    
    # Replace double newlines with paragraph breaks
    text = re.sub(r'\n\n+', '</p><p>', text)
    
    # Wrap in paragraph if not already wrapped
    if not text.startswith('<'):
        text = f'<p>{text}</p>'
    
    return text


# --------------- model & classes ---------------
_model_cache = None
_input_size_cache: Tuple[int, int, int] = (config.IMG_SIZE[0], config.IMG_SIZE[1], 3)

def build_model_architecture(num_classes: int = 10):
    """
    Rebuild the model architecture exactly as it was trained.
    This avoids Keras 3 serialization issues.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.efficientnet import EfficientNetB1
    
    input_layer = layers.Input(shape=(224, 224, 3))
    # Don't load imagenet weights here - we'll load our trained weights after
    base_model = EfficientNetB1(weights=None, include_top=False, input_tensor=input_layer)
    base_model.trainable = False
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(224, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

def get_model():
    """
    Load the model. If MODEL_PATH ends with .weights.h5, rebuild architecture and load weights.
    Otherwise try to load the full model.
    
    ⚡ OPTIMIZED: Model is cached in memory after first load for fast predictions!
    """
    global _model_cache, _input_size_cache
    if _model_cache is not None:
        print("✓ Using cached model (fast path)")
        return _model_cache

    try:
        if config.MODEL_PATH.endswith('.weights.h5'):
            # Rebuild architecture and load weights
            print(f"Rebuilding model architecture and loading weights from {config.MODEL_PATH}")
            num_classes = len(config.CLASS_NAMES)
            m = build_model_architecture(num_classes=num_classes)
            m.load_weights(config.MODEL_PATH)
            print("✓ Model weights loaded successfully!")
        else:
            # Try to load full model (legacy path)
            m = load_model(config.MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model at '{config.MODEL_PATH}'. "
            f"Likely the saved model has an input/channel mismatch (e.g. 1-channel). "
            f"Please re-export the model with input=(224,224,3). "
            f"Original error: {e}"
        )

    # Derive expected input size/channels from the loaded model
    ishape = m.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    # ishape: (None, H, W, C)
    H, W, C = ishape[1], ishape[2], ishape[3]
    _input_size_cache = (H or config.IMG_SIZE[0], W or config.IMG_SIZE[1], C or 3)

    # If the model itself reports C!=3, we still can run by *feeding* RGB and letting Keras handle it,
    # but for EfficientNet backbones it should be 3. Warn loudly:
    if C != 3:
        print(f"[WARN] Loaded model expects {C} input channels; EfficientNetB1 expects 3. "
              f"If predict fails, re-save the model with RGB input.")

    # ⚡ OPTIMIZATION: Run a dummy prediction to warm up the model
    # This compiles TensorFlow graph and makes subsequent predictions faster
    print("🔥 Warming up model with dummy prediction...")
    dummy_input = np.random.rand(1, H, W, C).astype(np.float32)
    _ = m.predict(dummy_input, verbose=0)
    print("✓ Model warmed up!")

    _model_cache = m
    return _model_cache

def load_class_names() -> List[str]:
    return list(config.CLASS_NAMES)


def is_probability_array(arr: np.ndarray, atol_sum: float = 1e-3) -> bool:
    """Heuristic: check if rows sum to ~1 and all entries in [0,1].
    This helps detect if model outputs are already probabilities or logits.
    """
    if arr.ndim != 2:
        return False
    row_sums = arr.sum(axis=1)
    if not np.all(np.isfinite(arr)):
        return False
    if np.any(arr < -1e-6) or np.any(arr > 1.0 + 1e-6):
        return False
    if not np.allclose(row_sums, 1.0, atol=atol_sum):
        return False
    return True


# --------------- preprocessing & predict ---------------
def load_image_for_model(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (preprocessed_batch, original_resized_rgb)
    - Reads as RGB (3-channel) to satisfy EfficientNet's stem.
    - Resizes to the height/width reported by model.input_shape (or config default).
    """
    _, _, C = _input_size_cache
    H, W, _ = _input_size_cache

    # Always convert to RGB to ensure 3 channels for EfficientNet
    img = Image.open(img_path).convert("RGB")
    original_resized = np.array(img.resize((W, H), Image.BILINEAR))

    # EfficientNet preprocessing (expects RGB in [0..255] float32)
    x = preprocess_input(original_resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    return x, original_resized


def predict_image(model, img_batch: np.ndarray):
    """
    Returns (pred_idx, confidence (0-1), probs np.ndarray)
    
    ⚡ FIXED: Now properly detects if model outputs are already probabilities
    or logits, and only applies softmax when needed. This gives REAL confidence scores!
    """
    preds = model.predict(img_batch)
    
    # Handle multiclass output (most common case)
    if preds.ndim == 2 and preds.shape[1] > 1:
        # Check if outputs are already probabilities or logits
        probs_candidate = np.asarray(preds)
        if is_probability_array(probs_candidate):
            # Already probabilities - use directly!
            probs = probs_candidate.squeeze()
        else:
            # Logits - apply softmax
            probs = tf.nn.softmax(preds, axis=1).numpy().squeeze()
        
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, probs
    
    # Handle binary output (single sigmoid)
    elif preds.ndim == 2 and preds.shape[1] == 1:
        p1 = float(tf.sigmoid(preds).numpy().squeeze())
        idx = int(p1 >= 0.5)
        probs = np.array([1.0 - p1, p1])
        return idx, float(probs[idx]), probs
    
    # Fallback: flatten and softmax
    else:
        flat = preds.reshape(1, -1)
        probs = tf.nn.softmax(flat, axis=1).numpy().squeeze()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, probs


# --------------- grad-cam ---------------
def make_gradcam_heatmap(model, img_batch: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
    last_conv_layer_name = config.LAST_CONV_LAYER
    # Validate the conv layer exists; if not, try to guess one
    try:
        _ = model.get_layer(last_conv_layer_name)
    except ValueError:
        # fallback: scan for last Conv2D
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0, 1))
    conv = conv_out[0]
    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_heatmap_and_overlay(original_path: str, heatmap: np.ndarray, alpha: float = 0.35):
    orig = Image.open(original_path).convert("RGB")
    w, h = orig.size
    heat_rgb = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb).resize((w, h), Image.BILINEAR)

    heat_name = _unique_name("heatmap", "png")
    overlay_name = _unique_name("overlay", "png")

    heat_path = os.path.join(config.OUTPUT_FOLDER, heat_name)
    heat_img.save(heat_path, "PNG")

    overlay = Image.blend(orig, heat_img, alpha=alpha)
    overlay_path = os.path.join(config.OUTPUT_FOLDER, overlay_name)
    overlay.save(overlay_path, "PNG")

    return heat_path.replace("\\", "/"), overlay_path.replace("\\", "/")


# --------------- Gemini ---------------
_gemini_model = None

def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        if not config.GEMINI_API_KEY:
            return None
        genai.configure(api_key=config.GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
    return _gemini_model

def gemini_check_is_wound(image_path: str):
    model = _get_gemini()
    if model is None:
        return True, "Gemini API key not configured; skipping wound check."

    prompt = "Answer ONLY with 'WOUND', 'NORMAL_SKIN', or 'NOT_WOUND' for this image. Use 'NORMAL_SKIN' for healthy skin without wounds, 'WOUND' for any type of wound or skin injury, and 'NOT_WOUND' for non-skin images."
    try:
        resp = model.generate_content([prompt, genai.upload_file(image_path)])
        txt = (resp.text or "").strip().upper()
        # Normalize and check whole-word patterns. Model replies sometimes include
        # variants like 'NOT WOUND', 'NOT_WOUND', or 'NOT-WOUND', and naive
        # substring checks (e.g. 'NOT_WOUND' contains 'WOUND') cause logic errors.
        import re
        
        # Check for NORMAL_SKIN first (allow normal skin to proceed)
        if re.search(r"\bNORMAL[_\-\s]?SKIN\b", txt):
            print(f"[gemini_check_is_wound] Gemini response (interpreted as NORMAL_SKIN): {txt}")
            return True, "normal skin detected - will classify as 'Normal' class"
        
        # Check negative: variants of NOT WOUND (reject non-skin images)
        if re.search(r"\bNOT[_\-\s]?WOUND\b", txt):
            print(f"[gemini_check_is_wound] Gemini response (interpreted as NOT_WOUND): {txt}")
            return False, "that is not a wound or skin image"
        
        # Then check positive 'WOUND' (but won't match 'NOT_WOUND' now)
        if re.search(r"\bWOUND\b", txt):
            print(f"[gemini_check_is_wound] Gemini response (interpreted as WOUND): {txt}")
            return True, "wound image detected"

        # If the model reply is ambiguous or doesn't contain the expected tokens,
        # fall back to permissive behavior (allow processing) but include the
        # raw Gemini text in the message for debugging.
        print(f"[gemini_check_is_wound] Gemini response ambiguous: {txt}")
        return True, f"Gemini unclear ('{txt}'); proceeding as wound."
    except Exception as e:
        return True, f"Gemini check failed ({e}); proceeding as wound."

def gemini_penanganan(pred_label: str, original_path: str, heatmap_path: str, overlay_path: str) -> str:
    model = _get_gemini()
    if model is None:
        return ("Gemini tidak terkonfigurasi (GEMINI_API_KEY kosong). "
                "Lewati analisis LLM pada mode ini.")
    prompt = f"""
Analisis gambar luka dengan overlay Grad-CAM untuk prediksi '{pred_label}'.

1. **Tipe Luka:** Jelaskan tipe luka (gunakan '{pred_label}') dan ciri-ciri yang terlihat.
2. **Penanganan:** Langkah terstruktur (bernomor) berdasarkan metode klinis dan jurnal.
3. **Keparahan:** Untuk luka parah, rekomendasikan evaluasi medis profesional.
4. **Interpretasi:** Area sorotan model dan kontribusinya pada klasifikasi '{pred_label}'.
5. **Referensi:** Sumber berkreditasi (section terpisah di bawah).

Fokus: saran praktis, klinis, informatif.
penting: koreksi jika prediksi salah. lihat original path berikan maksud sebenarnya itu luka apa, dan berikan saran penanganan yang sesuai pendek saja.
""".strip()

    try:
        resp = model.generate_content([
            prompt,
            genai.upload_file(original_path),
            genai.upload_file(overlay_path),
        ])
        return resp.text or "(Tidak ada teks dari Gemini.)"
    except Exception as e:
        return f"Analisis Gemini gagal: {e}"

# --------------- Global progress tracking ---------------
_current_progress = ""

def set_progress(message: str):
    """Set the current progress message"""
    global _current_progress
    _current_progress = message
    print(message)

def get_progress() -> str:
    """Get the current progress message"""
    global _current_progress
    return _current_progress

# --------------- main pipeline ---------------
def handle_upload_and_process(file_storage) -> dict:
    """
    - Save upload
    - Gemini triage wound/not-wound
    - Predict
    - Grad-CAM + overlay
    - Gemini penanganan
    """
    print("=" * 80)
    print("🔍 STARTING PREDICTION PIPELINE")
    print("=" * 80)

    ensure_dirs()

    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    saved_name = _unique_name("input", ext)
    saved_path = os.path.join(config.UPLOAD_FOLDER, saved_name)
    file_storage.save(saved_path)
    saved_path = saved_path.replace("\\", "/")

    print(f"✓ Step 1/7: Image uploaded and saved")
    print(f"  └─ File: {filename}")
    print(f"  └─ Path: {saved_path}")
    print()

    # LLM triage BEFORE loading the heavy model (fast-fail for non-wound images)
    print("⏳ Step 2/7: Gemini validation (checking if image contains wound)...")
    is_wound, msg = gemini_check_is_wound(saved_path)
    if not is_wound:
        print(f"✗ Gemini validation failed: {msg}")
        print("=" * 80)
        # Return quickly; no model load required
        return {"status": "not_wound", "message": msg, "input_image": saved_path}
    print(f"✓ Step 2/7: Gemini validation passed - {msg}")
    print()

    # Load model (will raise if the saved file is malformed)
    print("⏳ Step 3/7: Loading prediction model...")
    try:
        model = get_model()
        print("✓ Step 3/7: Model loaded successfully")
        print()
    except RuntimeError as e:
        print(f"✗ Model loading failed: {e}")
        print("=" * 80)
        return {"status": "error", "message": str(e)}

    # Preprocess & predict
    print("⏳ Step 4/7: Predicting wound class...")
    xbatch, original_resized = load_image_for_model(saved_path)
    pred_idx, conf, probs = predict_image(model, xbatch)
    classes = load_class_names()
    pred_label = classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}"
    confidence_pct = round(conf * 100.0, 2)

    # Log prediction with top-5 classes for verification
    topk_indices = np.argsort(probs)[::-1][:5]
    topk_str = ', '.join([f"{classes[i]}:{probs[i]*100:.2f}%" for i in topk_indices])
    print(f"✓ Step 4/7: Class predicted - {pred_label} ({confidence_pct}%)")
    print(f"  └─ Top 5 predictions: {topk_str}")
    print()

    # Grad-CAM
    print("⏳ Step 5/7: Generating GradCAM heatmap...")
    heatmap = make_gradcam_heatmap(model, xbatch, class_idx=pred_idx)
    print("✓ Step 5/7: GradCAM heatmap generated")
    print()

    print("⏳ Step 6/7: Generating overlay image...")
    heatmap_path, overlay_path = save_heatmap_and_overlay(saved_path, heatmap, alpha=0.35)
    print("✓ Step 6/7: Overlay image generated")
    print(f"  └─ Heatmap: {heatmap_path}")
    print(f"  └─ Overlay: {overlay_path}")
    print()

    # Gemini penanganan
    print("⏳ Step 7/7: Generating Gemini LLM analysis and treatment recommendations...")
    gemini_text = gemini_penanganan(pred_label, saved_path, heatmap_path, overlay_path)
    gemini_html = format_gemini_markdown(gemini_text)
    print("✓ Step 7/7: Gemini LLM analysis completed")
    print(f"  └─ Analysis length: {len(gemini_text)} characters")
    print()

    print("=" * 80)
    print("✅ PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()

    return {
        "status": "ok",
        "input_image": saved_path,
        "heatmap_image": heatmap_path,
        "overlay_image": overlay_path,
        "pred_label": pred_label,
        "confidence_pct": confidence_pct,
        "probs": probs.tolist(),
        "classes": classes,
        "gemini_text": gemini_text,
        "gemini_html": gemini_html,
    }


def handle_upload_and_process_with_progress(file_storage, progress_callback=None) -> dict:
    """
    Same as handle_upload_and_process but with progress callbacks
    - Save upload
    - Gemini triage wound/not-wound
    - Predict
    - Grad-CAM + overlay
    - Gemini penanganan
    """
    print("=" * 80)
    print("🔍 STARTING PREDICTION PIPELINE")
    print("=" * 80)

    ensure_dirs()

    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    saved_name = _unique_name("input", ext)
    saved_path = os.path.join(config.UPLOAD_FOLDER, saved_name)
    file_storage.save(saved_path)
    saved_path = saved_path.replace("\\", "/")

    print(f"✓ Step 1/5: Image uploaded and saved")
    print(f"  └─ File: {filename}")
    print(f"  └─ Path: {saved_path}")
    print()

    # Skip Gemini validation here since it was already done during file upload
    # Load model (will raise if the saved file is malformed)
    print("⏳ Step 2/5: Loading prediction model...")
    set_progress("Loading prediction model")
    if progress_callback:
        progress_callback("Loading prediction model")

    try:
        model = get_model()
        print("✓ Step 2/5: Model loaded successfully")
        print()
    except RuntimeError as e:
        print(f"✗ Model loading failed: {e}")
        print("=" * 80)
        return {"status": "error", "message": str(e)}

    # Preprocess & predict
    print("⏳ Step 3/5: Predicting wound class...")
    set_progress("Predicting wound class")
    if progress_callback:
        progress_callback("Predicting wound class")

    xbatch, original_resized = load_image_for_model(saved_path)
    pred_idx, conf, probs = predict_image(model, xbatch)
    classes = load_class_names()
    pred_label = classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}"
    confidence_pct = round(conf * 100.0, 2)

    # Log prediction with top-5 classes for verification
    topk_indices = np.argsort(probs)[::-1][:5]
    topk_str = ', '.join([f"{classes[i]}:{probs[i]*100:.2f}%" for i in topk_indices])
    print(f"✓ Step 3/5: Class predicted - {pred_label} ({confidence_pct}%)")
    print(f"  └─ Top 5 predictions: {topk_str}")
    print()

    # Grad-CAM
    print("⏳ Step 4/5: Generating GradCAM heatmap and overlay...")
    set_progress("Generating GradCAM heatmap")
    if progress_callback:
        progress_callback("Generating GradCAM heatmap")

    heatmap = make_gradcam_heatmap(model, xbatch, class_idx=pred_idx)

    set_progress("Generating overlay image")
    if progress_callback:
        progress_callback("Generating overlay image")

    heatmap_path, overlay_path = save_heatmap_and_overlay(saved_path, heatmap, alpha=0.35)
    print("✓ Step 4/5: GradCAM heatmap and overlay generated")
    print(f"  └─ Heatmap: {heatmap_path}")
    print(f"  └─ Overlay: {overlay_path}")
    print()

    # Gemini penanganan
    print("⏳ Step 5/5: Generating Gemini LLM analysis and treatment recommendations...")
    set_progress("Generating Gemini LLM analysis")
    if progress_callback:
        progress_callback("Generating Gemini LLM analysis")

    gemini_text = gemini_penanganan(pred_label, saved_path, heatmap_path, overlay_path)
    gemini_html = format_gemini_markdown(gemini_text)
    print("✓ Step 5/5: Gemini LLM analysis completed")
    print(f"  └─ Analysis length: {len(gemini_text)} characters")
    print()

    print("=" * 80)
    print("✅ PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()

    set_progress("")  # Clear progress
    if progress_callback:
        progress_callback("completed")

    return {
        "status": "ok",
        "input_image": saved_path,
        "heatmap_image": heatmap_path,
        "overlay_image": overlay_path,
        "pred_label": pred_label,
        "confidence_pct": confidence_pct,
        "probs": probs.tolist(),
        "classes": classes,
        "gemini_text": gemini_text,
        "gemini_html": gemini_html,
    }
