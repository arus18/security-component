from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import base64
from firebase_admin import messaging
from collections import deque
import sys
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
import firebase_admin
from firebase_admin import credentials, messaging
import logging
import traceback
from flask import send_file
from io import BytesIO
from flask import Flask, Response
from flask_cors import CORS
from tensorflow import keras
from datetime import datetime
import uuid


logging.basicConfig(level=logging.INFO)

# Set up Ngrok authentication token
#ngrok.set_auth_token("2cwuKNigB4MOmhcqz2V7iaAsZqF_7f797rB1svrxrUAKGXmAK")

# Establish Ngrok tunnel with custom domain
#ngrok_tunnel = ngrok.connect(80, bind_tls=True, hostname="discrete-lately-killdeer.ngrok-free.app")

# Print Ngrok URL
#print("Ngrok Tunnel URL:", ngrok_tunnel.public_url)

app = Flask(__name__)
CORS(app)

harmful_object_class_index = 0
# Global variables
recent_object_detection_predictions = {}  # Dictionary to store recent object detection predictions for each camera
recent_video_classification_predictions = {}  # Dictionary to store recent video classification predictions for each camera
activities_to_detect = {}
client_token = None
max_recent_predictions = 10  # Maximum number of recent predictions to store
RECENT_PREDICTIONS_SIZE = 10

recent_predictions = []
predictions_list = []

# Load YOLO model for object detection
yolo_model_path = 'best.pt'
yolo_model = YOLO(yolo_model_path)

if not firebase_admin._apps:
    cred = credentials.Certificate("daycare-237d8-firebase-adminsdk-fza8w-ff3a089ab7.json")
    firebase_admin.initialize_app(cred)

# Load the pre-trained video classification model
video_classification_model_path = "movinet_saved_model"  # Update the path accordingly
video_classification_model = tf.keras.models.load_model(video_classification_model_path)

# Define class names for video classification
class_names = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword",
    "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug",
    "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup",
    "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit",
    "situp", "smile", "smoke", "somersault", "stand", "swing_baseball", "sword", "sword_exercise",
    "talk", "throw", "turn", "walk", "wave"
]

# Configuration parameters
batch_size = 16
object_detection_threshold = 0.48
video_classification_threshold = 0.5

# Function to perform object detection on a batch of frames
def perform_object_detection_batch(frames):
    results_batch = yolo_model(frames)
    return results_batch

# Function to preprocess a single frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    return frame

# Function to preprocess a batch of frames into clips
def preprocess_video(frames, clip_length=16):
    clips = []
    frames_buffer = []

    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        frames_buffer.append(preprocessed_frame)

        if len(frames_buffer) == clip_length:
            clips.append(frames_buffer)
            frames_buffer = []

    # If any remaining frames, pad the last clip
    if frames_buffer:
        while len(frames_buffer) < clip_length:
            frames_buffer.append(np.zeros_like(frames_buffer[0]))
        clips.append(frames_buffer)

    return np.stack(clips)

# Function to perform video classification on a batch of frames
def perform_video_classification_batch(camera_ip, frames):
    global predictions_list
    try:
        # Preprocess the batch of frames into clips
        preprocessed_clips = preprocess_video(frames)

        # Model prediction
        predictions_batch = video_classification_model.predict(preprocessed_clips)

        # Process video classification results
        for clip_index, clip_prediction in enumerate(predictions_batch):
            max_confidence_index = np.argmax(clip_prediction)
            predicted_class = class_names[max_confidence_index]
            confidence_score = clip_prediction[max_confidence_index]

            # Generate clip ID
            clip_id = f"{camera_ip}_clip_{clip_index}"

            # Save video clip with ID
            recent_video_classification_predictions.setdefault(camera_ip, []).append({'clip_id': clip_id, 'clip': frames[clip_index], 'prediction': {'predicted_class': predicted_class, 'confidence_score': float(confidence_score)}})

        return predictions_list

    except Exception as e:
        print(f"Error during video classification: {e}")
        return []

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2, _ = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Function to send Firebase push notification
#def send_push_notification(camera_ip, predictions):

@app.route('/save_token', methods=['POST'])
def save_token():
    global client_token
    data = request.form.get('token')
    client_token = data

    # Return a 200 OK response with the JSON message
    return jsonify(message="Token saved successfully."), 200


# Endpoint for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Read camera IP from the request
        camera_ip = request.form.get('camera_ip')

        # Read image files from the batch
        frames = [request.files[f'frame{i}'].read() for i in range(batch_size)]

        # Convert frames to NumPy arrays
        imgs = [cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in frames]

        # Perform object detection on the batch of frames
        results_batch = perform_object_detection_batch(imgs)

        # Process object detection results
        for frame_index, results in enumerate(results_batch):
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # confidence
                    confidence = box.conf[0].item()
                    print(confidence)

                    # class index
                    cls = int(box.cls[0])

                    # Check if a harmful object is found
                    if cls == harmful_object_class_index and confidence > object_detection_threshold:
                        harmful_object_found = True

                        # Draw boxes on the frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(imgs[frame_index], (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Add prediction to recent predictions list for the specific camera
                        recent_object_detection_predictions.setdefault(camera_ip, []).append({
                            'confidence_score': confidence,
                            'image': frames[frame_index]
                        })

        # Limit the recent predictions size for the specific camera
        if camera_ip is not None and camera_ip in recent_object_detection_predictions:
            recent_object_detection_predictions[camera_ip] = recent_object_detection_predictions[camera_ip][-RECENT_PREDICTIONS_SIZE:]

        return 'Object detection completed'

    except Exception as e:
        traceback.print_exc()  # Print the traceback
        error_message = f"Error during object detection: {e}"
        print(error_message)
        return jsonify(error=error_message), 500


@app.route('/save_activities', methods=['POST'])
def save_activities():
    data = request.json
    camera_ip = data.get('camera_ip')
    activities = data.get('activities')
    activities_to_detect[camera_ip] = activities
    print(activities_to_detect)
    return jsonify(message="Activities saved successfully.")



# Endpoint for video classification
@app.route('/classify_video', methods=['POST'])
def classify_video():
    try:
        # Read camera IP from the request
        camera_ip = request.form.get('camera_ip')

        # Read frames from the request
        frames = [request.files[f'frame{i}'].read() for i in range(batch_size)]

        # Convert frames to NumPy arrays
        imgs = [cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in frames]

        # Perform video classification on the batch of frames
        perform_video_classification_batch(camera_ip, imgs)

        # Limit the recent predictions size for the specific camera
        recent_video_classification_predictions[camera_ip] = recent_video_classification_predictions[camera_ip][-RECENT_PREDICTIONS_SIZE:]

        # Send push notification for harmful objects
        #send_push_notification(camera_ip, predictions_list)

        return jsonify(success=True)

    except Exception as e:
        print(f"Error during video classification: {e}")
        return jsonify(error=str(e)), 500

@app.route('/send-notification', methods=['POST'])
def send_notification():
    data = request.json
    registration_token = data.get('registration_token')
    title = data.get('title')
    body = data.get('body')

    # Create a message
    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        token=client_token,  # Specify the registration token of the device
    )

    # Send the message
    try:
        response = messaging.send(message)
        return jsonify({"message": "Notification sent successfully.", "response": response}), 200
    except Exception as e:
        return jsonify({"message": "Failed to send notification.", "error": str(e)}), 500


@app.route('/recent_object_detection_predictions', methods=['GET'])
def get_recent_object_detection_predictions():
  camera_ip = request.args.get('camera_ip')
  if camera_ip is None:
    return jsonify(message="Missing camera_ip parameter"), 400

  predictions = recent_object_detection_predictions[camera_ip]  # Replace with your data store logic

  if not predictions:
    return jsonify(message="No recent object detection predictions available for the specified camera IP"), 404

  # Add unique ID and format timestamp
  processed_predictions = []
  for prediction in predictions:
    unique_id = str(uuid.uuid4())
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Generate a unique ID using uuid library
    processed_prediction = {
      "id": unique_id,
      "confidence": prediction["confidence_score"],
      "timestamp": current_timestamp,  # Format timestamp for readability
      "image": base64.b64encode(prediction["image"]).decode('utf-8') if "image" in prediction else None,  # Handle optional image data
    }
    processed_predictions.append(processed_prediction)

  return jsonify(processed_predictions)

"""@app.route('/recent_object_detection_predictions', methods=['GET'])
def get_recent_object_detection_predictions():
    camera_ip = request.args.get('camera_ip')
    if camera_ip in recent_object_detection_predictions:
        predictions = recent_object_detection_predictions[camera_ip]

        # Encode the image data in base64 format
        encoded_predictions = []
        for prediction in predictions:
            encoded_prediction = {}
            for key, value in prediction.items():
                if key == 'image':
                    # Encode the image data in base64 format
                    encoded_prediction[key] = base64.b64encode(value).decode('utf-8')
                else:
                    encoded_prediction[key] = value
            encoded_predictions.append(encoded_prediction)

        return jsonify(encoded_predictions)
    else:
        return jsonify(message="No recent object detection predictions available for the specified camera IP")"""


# Endpoint for fetching recent video classification predictions
@app.route('/recent_predicted_clips', methods=['GET'])
def get_recent_predicted_clips():
    camera_ip = request.args.get('camera_ip')
    if camera_ip in recent_video_classification_predictions:
        predicted_clips = []
        for prediction in recent_video_classification_predictions[camera_ip]:
            if 'clip' in prediction:
                prediction_without_clip = prediction.copy()
                del prediction_without_clip['clip']
                predicted_clips.append(prediction_without_clip)
            else:
                predicted_clips.append(prediction)
        return jsonify(predicted_clips)
    else:
        return jsonify(message="No recent predicted clips available for the specified camera IP")

@app.route('/stream_clips', methods=['GET'])
def stream_clips():
    camera_ip = request.args.get('camera_ip')

    if camera_ip not in recent_video_classification_predictions:
        return jsonify(message=f"No clips available for camera IP: {camera_ip}")

    clips = recent_video_classification_predictions[camera_ip]

    def generate():
        for clip in clips:
            clip_data = clip['clip']
            if clip_data.any():
                # Stream clip data frame by frame (adjust based on frame format)
                for frame in clip_data:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_clip', methods=['GET'])
def stream_clip():
  camera_ip = request.args.get('camera_ip')
  clip_id = request.args.get('clip_id')

  if camera_ip not in recent_video_classification_predictions:
    return jsonify(message=f"No clips available for camera IP: {camera_ip}")

  clips = recent_video_classification_predictions[camera_ip]
  clip_data = None
  for clip in clips:
    if clip['clip_id'] == clip_id:
      clip_data = clip['clip']
      break

  if not clip_data.any():
    return jsonify(message=f"No clip found with ID {clip_id}")

  def generate():
    # Stream clip data frame by frame (adjust based on frame format)
    for frame in clip_data:
      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')

  return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(port=8000)
