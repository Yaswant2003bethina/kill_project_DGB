import cv2
import numpy as np
import os
import time
import threading
import pygame  # For playing alarm sound

def create_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(cascade_path)
    return face_detector

def play_alarm(duration_seconds=10, audio_file='sound_effects.mp3'):
    """Play custom alarm sound for the specified duration."""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play(-1)  # Loop indefinitely

        # Wait for the duration of the alarm, then stop
        time.sleep(duration_seconds)
        pygame.mixer.music.stop()
    except pygame.error as e:
        print(f"Error playing sound: {e}")

def simulate_door_control(action="open"):
    """Simulate door control system."""
    if action == "open":
        print("🚪 Opening door... Please wait.")
        time.sleep(2)  # Simulate door opening time
        print("✅ Door opened successfully!")
    else:
        print("🚪 Closing door...")
        time.sleep(1)  # Simulate door closing time
        print("✅ Door closed!")

def train_face_recognizer():
    """Train the face recognizer with labeled images."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_dict, current_label = {}, 0
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return recognizer, {}
        
    face_detector = create_face_detector()
    
    # Check if directory is empty
    if not os.listdir(data_dir):
        print(f"Warning: No training data found in '{data_dir}'.")
        return recognizer, {}
    
    print("Training face recognizer with available data...")
    
    # Loop through each person in the dataset
    for person_name in os.listdir(data_dir):
        if person_name.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(data_dir, person_name)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            face_rects = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Only process the largest face in training images
            if len(face_rects) > 0:
                x, y, w, h = max(face_rects, key=lambda rect: rect[2] * rect[3])
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (128, 128))  # Resize to expected input size
                
                # Extract name from filename (assuming format like "john_doe_1.jpg")
                name = "_".join(person_name.split("_")[:-1])
                if not name:  # Fallback if naming convention is different
                    name = os.path.splitext(person_name)[0]
                    
                if name not in label_dict:
                    label_dict[name] = current_label
                    current_label += 1
                    print(f"Added person: {name} with label {label_dict[name]}")
                
                faces.append(face_roi)
                labels.append(label_dict[name])
            else:
                print(f"Warning: No face detected in training image {image_path}")
    
    if faces:
        print(f"Training with {len(faces)} faces from {len(label_dict)} different people")
        recognizer.train(np.array(faces), np.array(labels))
        # Set a better threshold value
        recognizer.setThreshold(80)
    else:
        print("Error: No faces detected in training data")
    
    return recognizer, label_dict

def recognize_faces():
    """Perform face recognition using the trained model."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    face_detector = create_face_detector()
    recognizer, label_dict = train_face_recognizer()
    
    # Check if we have any training data
    if not label_dict:
        print("Error: No training data available. Please add labeled images to the 'data' directory.")
        return
        
    label_names = {v: k for k, v in label_dict.items()}
    
    # Define confidence threshold for recognition
    # Lower values are more strict (less false positives)
    CONFIDENCE_THRESHOLD = 70
    
    unknown_person_detected = False
    door_open = False
    alarm_thread = None
    alarm_duration = 10
    alarm_pause_duration = 60
    alarm_end_time = 0
    
    print("Starting face recognition. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing video.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            current_time = time.time()
            
            # If no faces detected
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_DUPLEX, 
                           0.7, (255, 255, 0), 1)
            else:
                # Process only the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (128, 128))
                
                # Predict label and confidence
                label, confidence = recognizer.predict(face_roi)
                
                # Initialize name and color variables
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown face
                
                # Lower confidence value means better match in OpenCV LBPH
                if confidence < CONFIDENCE_THRESHOLD and label in label_names:
                    name = label_names[label]
                    color = (0, 255, 0)  # Green for known face
                    
                    # Handle recognized person
                    if not door_open:
                        threading.Thread(target=simulate_door_control, args=("open",)).start()
                        door_open = True
                    unknown_person_detected = False
                else:
                    # Handle unknown person
                    if not unknown_person_detected and current_time > alarm_end_time:
                        unknown_person_detected = True
                        print(f"Unknown person detected! Confidence: {confidence}")

                        # Start the alarm in a separate thread
                        if not alarm_thread or not alarm_thread.is_alive():
                            alarm_thread = threading.Thread(target=play_alarm, args=(alarm_duration,))
                            alarm_thread.start()

                        if door_open:
                            threading.Thread(target=simulate_door_control, args=("close",)).start()
                            door_open = False

                        # Set alarm cooldown period
                        alarm_end_time = current_time + alarm_duration + alarm_pause_duration

                # Draw face box and name on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                confidence_text = f"{name} ({confidence:.1f})"
                cv2.putText(frame, confidence_text, 
                           (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 
                           0.6, (255, 255, 255), 1)

                if name == "Unknown":
                    cv2.putText(frame, "WARNING: Unknown Person Detected!", 
                               (10, 30), cv2.FONT_HERSHEY_DUPLEX, 
                               0.7, (0, 0, 255), 2)

            # Display door status
            status_text = "Door OPEN" if door_open else "Door CLOSED"
            cv2.putText(frame, f"Status: {status_text}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 
                       0.7, (255, 255, 255), 1)
            
            # Display confidence threshold
            cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD}", 
                       (frame.shape[1] - 200, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Recognition Security System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):  # Increase threshold (more lenient)
                CONFIDENCE_THRESHOLD = min(CONFIDENCE_THRESHOLD + 5, 100)
                print(f"Threshold increased to: {CONFIDENCE_THRESHOLD}")
            elif key == ord('-'):  # Decrease threshold (more strict)
                CONFIDENCE_THRESHOLD = max(CONFIDENCE_THRESHOLD - 5, 10)
                print(f"Threshold decreased to: {CONFIDENCE_THRESHOLD}")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pygame.mixer.get_init():
            pygame.mixer.quit()

if __name__ == "__main__":
    recognize_faces()