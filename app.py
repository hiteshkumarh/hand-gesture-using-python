from flask import Flask, render_template, Response
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.7)

# Load the image to be zoomed
image_path = "th.jpeg"
image = cv2.imread(image_path)

# Capture video from the default camera
cap = cv2.VideoCapture(0)

def generate_frames():
    zoom_scale = 1.0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect hands
            hands, img = detector.findHands(frame)

            if hands:
                # Example: Adjust zoom based on the distance between two fingers
                hand = hands[0]
                lmList = hand['lmList']
                if len(lmList) >= 8:
                    x1, y1 = lmList[4][0], lmList[4][1]  # Thumb tip
                    x2, y2 = lmList[8][0], lmList[8][1]  # Index finger tip
                    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    zoom_scale = max(0.5, min(2.0, distance / 100))  # Scale between 0.5 and 2.0

            # Resize image based on zoom scale
            h, w, _ = image.shape
            new_h, new_w = int(h * zoom_scale), int(w * zoom_scale)
            resized_image = cv2.resize(image, (new_w, new_h))

            # Center the resized image
            start_x, start_y = (new_w - w) // 2, (new_h - h) // 2
            end_x, end_y = start_x + w, start_y + h
            if new_w > w and new_h > h:
                cropped_image = resized_image[start_y:end_y, start_x:end_x]
            else:
                cropped_image = cv2.resize(image, (w, h))

            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', cropped_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
