from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash
import cv2
import numpy as np
import os
import time
import base64
from datetime import datetime
import json
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'face_recognition_security_system'

# Global variables
camera = None
face_detector = None
recognizer = None
label_dict = {}
label_names = {}
face_recognition_active = False
door_status = "CLOSED"
detection_log = []
confidence_threshold = 70
status_message = ""

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Initialize face detector
def create_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# Train face recognizer
def train_face_recognizer():
    global status_message
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_dict, current_label = {}, 0
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        status_message = f"Error: Data directory '{data_dir}' not found."
        return recognizer, {}
        
    face_detector = create_face_detector()
    
    if not os.listdir(data_dir):
        status_message = f"Warning: No training data found in '{data_dir}'."
        return recognizer, {}
    
    status_message = "Training face recognizer with available data..."
    
    for person_name in os.listdir(data_dir):
        if person_name.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(data_dir, person_name)
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_rects = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(face_rects) > 0:
                x, y, w, h = max(face_rects, key=lambda rect: rect[2] * rect[3])
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (128, 128))
                
                name = "_".join(person_name.split("_")[:-1])
                if not name:
                    name = os.path.splitext(person_name)[0]
                    
                if name not in label_dict:
                    label_dict[name] = current_label
                    current_label += 1
                
                faces.append(face_roi)
                labels.append(label_dict[name])
    
    if faces:
        status_message = f"Training completed with {len(faces)} faces from {len(label_dict)} different people"
        recognizer.train(np.array(faces), np.array(labels))
    else:
        status_message = "Error: No faces detected in training data"
    
    return recognizer, label_dict

# Log detection event
def log_detection(name, confidence):
    global detection_log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detection_info = {"timestamp": timestamp, "name": name, "confidence": confidence, "status": "Authorized" if name != "Unknown" else "Unauthorized"}
    detection_log.append(detection_info)
    if len(detection_log) > 20:
        detection_log = detection_log[-20:]
    
    with open("detection_log.json", "w") as f:
        json.dump(detection_log, f)

# Function to capture video frames for face recognition
def generate_frames():
    global camera, face_detector, recognizer, label_names, face_recognition_active, door_status, confidence_threshold, status_message
    
    if not face_recognition_active:
        return
    
    while face_recognition_active:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 1)
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (128, 128))
            
            label, confidence = recognizer.predict(face_roi)
            name = "Unknown" if confidence >= confidence_threshold else label_names.get(label, "Unknown")
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
            log_detection(name, round(confidence, 1))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    authorized_users = list(label_names.values()) if label_names else []
    return render_template('index.html', authorized_users=authorized_users, system_active=face_recognition_active,
                           door_status=door_status, confidence_threshold=confidence_threshold,
                           detection_log=detection_log, status_message=status_message)

@app.route('/video_feed')
def video_feed():
    if face_recognition_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(open('static/placeholder.jpg', 'rb').read(), mimetype='image/jpeg')

@app.route('/start_system', methods=['POST'])
def start_system():
    global camera, face_detector, recognizer, label_dict, label_names, face_recognition_active
    
    if face_recognition_active:
        flash('System is already running', 'info')
        return redirect(url_for('index'))
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        flash('Error: Could not open camera', 'danger')
        return redirect(url_for('index'))
    
    face_detector = create_face_detector()
    recognizer, label_dict = train_face_recognizer()
    
    if not label_dict:
        camera.release()
        flash('No training data available. Please add face images first', 'warning')
        return redirect(url_for('index'))
    
    label_names = {v: k for k, v in label_dict.items()}
    face_recognition_active = True
    flash('Face Recognition System Started', 'success')
    return redirect(url_for('index'))

@app.route('/stop_system', methods=['POST'])
def stop_system():
    global camera, face_recognition_active
    
    if face_recognition_active:
        face_recognition_active = False
        if camera:
            camera.release()
        flash('Face Recognition System Stopped', 'info')
    return redirect(url_for('index'))

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global confidence_threshold
    try:
        new_threshold = int(request.form.get('threshold', 70))
        if 10 <= new_threshold <= 100:
            confidence_threshold = new_threshold
            flash(f'Confidence threshold updated to {new_threshold}', 'success')
        else:
            flash('Threshold must be between 10 and 100', 'warning')
    except ValueError:
        flash('Invalid threshold value', 'danger')
    return redirect(url_for('index'))

@app.route('/capture_face', methods=['GET', 'POST'])
def capture_face():
    if request.method == 'GET':
        return render_template('capture.html')
    if request.method == 'POST':
        person_name = request.form.get('name', '').strip()
        if not person_name:
            flash('Please enter a name', 'warning')
            return redirect(url_for('capture_face'))
        return redirect(url_for('do_capture', name=person_name))

@app.route('/do_capture/<name>')
def do_capture(name):
    return render_template('do_capture.html', name=name)

@app.route('/process_capture', methods=['POST'])
def process_capture():
    data = request.json
    person_name = data.get('name', '')
    image_data = data.get('image', '')

    if not person_name or not image_data:
        return jsonify({'success': False, 'message': 'Missing data'})

    try:
        # Decode the base64 image data
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform face detection and cropping using Mediapipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)

        if results.detections:
            largest_face = None
            largest_area = 0

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Calculate the area of the bounding box
                area = width * height

                # Check if this is the largest face so far (nearest face)
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, width, height)

            # If a nearest face is found, crop and save it
            if largest_face:
                x, y, width, height = largest_face

                # Add padding around the face bounding box
                x_pad = int(width * 0.1)  # 10% padding on width
                y_pad = int(height * 0.2)  # 20% padding on height

                x = max(0, x - x_pad)
                y = max(0, y - y_pad)
                width = min(w, width + 2 * x_pad)
                height = min(h, height + 2 * y_pad)

                cropped_face = img[y:y+height, x:x+width]  # Crop the face region

                # Save the cropped face image
                img_count = len([f for f in os.listdir('data') if f.startswith(person_name + '_')])
                file_name = f"data/{person_name}_{img_count + 1}.jpg"
                cv2.imwrite(file_name, cropped_face)

                return jsonify({'success': True, 'message': f'Image saved as {file_name}'})

        return jsonify({'success': False, 'message': 'No face detected in the image'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})

@app.route('/get_system_status')
def get_system_status():
    return jsonify({
        'active': face_recognition_active,
        'door_status': door_status,
        'confidence_threshold': confidence_threshold,
        'status_message': status_message
    })

if __name__ == '__main__':
    try:
        with open("detection_log.json", "r") as f:
            detection_log = json.load(f)
    except:
        detection_log = []
    
    app.run(debug=True)