from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
import os
import cv2
import easyocr
import subprocess
from concurrent.futures import ThreadPoolExecutor
from docx import Document
from docx.shared import Pt
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth as firebase_auth
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS

# Load .env file
load_dotenv()

# ------------------ Flask app config ------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Updated CORS configuration
CORS(app, 
     origins=["http://127.0.0.1:5000", "http://localhost:5000"],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"],
     expose_headers=["Content-Type", "Authorization"]
)

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"
DOCX_FILE = "captions.docx"
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# ------------------ Firebase setup ------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIREBASE_KEY_FILE"))
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.getenv("FIREBASE_BUCKET")
    })

db = firestore.client()
bucket = storage.bucket()

# ------------------ OCR setup ------------------
reader = easyocr.Reader(['en'], gpu=False)

# ------------------ Add preflight handler ------------------
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
        response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization, X-Requested-With")
        response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

# ------------------ Routes ------------------

# Login page
@app.route("/", methods=["GET"])
def login():
    return render_template("login.html",
        FIREBASE_API_KEY=os.getenv("FIREBASE_API_KEY"),
        FIREBASE_AUTH_DOMAIN=os.getenv("FIREBASE_AUTH_DOMAIN"),
        FIREBASE_PROJECT_ID=os.getenv("FIREBASE_PROJECT_ID"),
        FIREBASE_STORAGE_BUCKET=os.getenv("FIREBASE_STORAGE_BUCKET"),
        FIREBASE_MSG_SENDER_ID=os.getenv("FIREBASE_MSG_SENDER_ID"),
        FIREBASE_APP_ID=os.getenv("FIREBASE_APP_ID"),
        FIREBASE_MEASUREMENT_ID=os.getenv("FIREBASE_MEASUREMENT_ID"),
        FIREBASE_CLIENT_ID=os.getenv("FIREBASE_CLIENT_ID")
    )

# Signup page
@app.route("/signup", methods=["GET"])
def signup():
    return render_template("signup.html",
        FIREBASE_API_KEY=os.getenv("FIREBASE_API_KEY"),
        FIREBASE_AUTH_DOMAIN=os.getenv("FIREBASE_AUTH_DOMAIN"),
        FIREBASE_PROJECT_ID=os.getenv("FIREBASE_PROJECT_ID"),
        FIREBASE_STORAGE_BUCKET=os.getenv("FIREBASE_STORAGE_BUCKET"),
        FIREBASE_MSG_SENDER_ID=os.getenv("FIREBASE_MSG_SENDER_ID"),
        FIREBASE_APP_ID=os.getenv("FIREBASE_APP_ID"),
        FIREBASE_MEASUREMENT_ID=os.getenv("FIREBASE_MEASUREMENT_ID"),
        FIREBASE_CLIENT_ID=os.getenv("FIREBASE_CLIENT_ID")
    )

# Dashboard page
@app.route("/index")
def index():
    if "uid" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# Handle ID token from frontend, verify & store session
@app.route("/callback", methods=["POST", "OPTIONS"])
def callback():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No JSON data received"}), 400
            
        id_token = data.get("idToken")
        if not id_token:
            return jsonify({"status": "error", "error": "Missing ID token"}), 400
        
        # Verify the ID token
        try:
            decoded = firebase_auth.verify_id_token(id_token)
        except Exception as e:
            print(f"❌ Token verification failed: {e}")
            response = jsonify({"status": "error", "error": "Invalid token"})
            response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
            response.headers.add("Access-Control-Allow-Credentials", "true")
            return response, 401
        
        uid = decoded.get("uid")
        email = decoded.get("email", "unknown@example.com")
        name = decoded.get("name", "")
        picture = decoded.get("picture", "")

        # Save/update user in Firestore
        try:
            db.collection("users").document(uid).set({
                "uid": uid, 
                "email": email, 
                "name": name, 
                "picture": picture,
                "last_login": firestore.SERVER_TIMESTAMP
            }, merge=True)
        except Exception as e:
            print(f"❌ Firestore save failed: {e}")

        # Save in session
        session['uid'] = uid
        session['email'] = email
        session['user'] = {
            "uid": uid, 
            "email": email, 
            "name": name, 
            "picture": picture
        }

        print(f"✅ Login success: {email}")
        
        # Create response with proper headers
        response = jsonify({"status": "success", "redirect": "/index"})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 200
        
    except Exception as e:
        print(f"❌ Callback error: {e}")
        response = jsonify({"status": "error", "error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 500

# Upload video, process, OCR, save docx to Firebase Storage
@app.route("/upload", methods=["POST"])
def upload():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"status": "error", "error": "Invalid token format"}), 401
    id_token = auth_header.split(" ")[1]

    try:
        decoded = firebase_auth.verify_id_token(id_token)
        uid = decoded["uid"]
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 401

    # Cleanup old files
    clear_folder(FRAME_FOLDER)
    clear_folder(UPLOAD_FOLDER)
    if os.path.exists(DOCX_FILE):
        os.remove(DOCX_FILE)

    video = request.files["video"]
    if not allowed_file(video.filename):
        return jsonify({"status": "error", "error": "Unsupported file format"}), 400

    raw_path = os.path.join(UPLOAD_FOLDER, secure_filename(video.filename))
    video_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    video.save(raw_path)

    # Compress video
    result = subprocess.run([
        "ffmpeg", "-i", raw_path,
        "-vf", "scale='if(gt(iw,1920),1920,iw)':'if(gt(ih,1080),1080,ih)'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-y", video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        return jsonify({"status": "error", "error": "FFmpeg failed"}), 500

    try:
        extract_frames(video_path)
        run_ocr_parallel()
        # Upload docx to Firebase Storage
        blob_path = f"user_uploads/{uid}/captions.docx"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(DOCX_FILE)
        blob.make_public()
        return jsonify({"status": "success", "url": blob.public_url})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return "File too large. Max 500MB.", 413

# ------------------ Utils ------------------

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder):
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_dynamic_caption_region(frames):
    bboxes = []
    for frame in frames:
        pre = preprocess_image(frame)
        result = reader.readtext(pre)
        for (bbox, text, conf) in result:
            if conf > 0.5:
                x_coords = [pt[0] for pt in bbox]
                y_coords = [pt[1] for pt in bbox]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                bboxes.append((x1, y1, x2, y2))
    if not bboxes:
        return None
    return (min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes))

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(int(total / fps), 3600)
    # Sample first few frames to detect caption region
    sample_frames = []
    sec = 0
    while sec < min(duration, 20):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
        sec += 5
    region = detect_dynamic_caption_region(sample_frames)
    # Extract frames for OCR
    sec = 0
    frame_interval = max(2, min(10, duration // 120))
    while sec < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[region[1]:region[3], region[0]:region[2]] if region else frame
        filename = os.path.join(FRAME_FOLDER, f"frame_{sec:04d}.png")
        cv2.imwrite(filename, cropped)
        sec += frame_interval
    cap.release()

def run_ocr_parallel():
    doc = Document()
    doc.add_heading("Extracted Captions", level=1)
    last_caption, last_time = "", -5
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = executor.map(process_frame, sorted(os.listdir(FRAME_FOLDER)))
        for item in futures:
            if item:
                time_sec, text = item
                if text != last_caption:
                    prefix = f"[{time_sec:02d}s] " if time_sec - last_time >= 3 else ""
                    doc.add_paragraph(f"{prefix}{text}")
                    last_caption, last_time = text, time_sec
    doc.save(DOCX_FILE)

def process_frame(fname):
    try:
        img = cv2.imread(os.path.join(FRAME_FOLDER, fname))
        pre = preprocess_image(img)
        result = reader.readtext(pre)
        text = " ".join([t for _, t, conf in result if conf > 0.6]).strip()
        if text:
            time_sec = int(fname.split('_')[1].split('.')[0])
            return (time_sec, text)
    except Exception as e:
        print(f"❌ OCR failed on {fname}: {e}")
    return None

# ------------------ Start app ------------------
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)