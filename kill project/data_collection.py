import cv2
import mediapipe as mp
import os
import time

# Create a 'data' folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Function to capture and save the cropped face images
def save_cropped_face(name, face, img_id):
    file_name = f"data/{name}_{img_id}.jpg"
    cv2.imwrite(file_name, face)
    print(f"Cropped face image saved: {file_name}")

def main():
    img_id = 0
    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            break

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        # If faces are detected, process each one
        if results.detections:
            largest_face = None
            largest_area = 0

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Calculate the area of the bounding box
                area = width * height

                # Check if this is the largest face so far (nearest face)
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, width, height)

            # If a nearest face is found, crop and display it
            if largest_face:
                x, y, width, height = largest_face

                # Add padding around the face bounding box (10% padding)
                padding = 0.1
                x_pad = int(width * padding)
                y_pad = int(height * padding)

                x = max(0, x - x_pad)
                y = max(0, y - y_pad)
                width = min(w, width + 2 * x_pad)
                height = min(h, height + 2 * y_pad)

                # Crop the nearest face
                cropped_face = frame[y:y+height, x:x+width]

                # Draw the expanded bounding box on the original frame
                cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            name = input("Enter the name to save the images: ")
            for i in range(20):  # Capture and save 10 cropped face images
                time.sleep(3)  # Wait for 3 seconds between captures

                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    largest_face = None
                    largest_area = 0

                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, c = frame.shape
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

                        cropped_face = frame[y:y+height, x:x+width]  # Crop the face region

                        img_id += 1
                        save_cropped_face(name, cropped_face, img_id)  # Save the face

                # Display the frame with the nearest face rectangle
                cv2.imshow('Face Recognition', frame)
                cv2.waitKey(500)  # Wait 500ms between captures to avoid duplicates

            print(f"Saved 20 cropped face images for {name}. Press 's' to save more or 'q' to quit.")

        elif key == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
