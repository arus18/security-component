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
import time
from video_classifier import classify_video_tf



logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

harmful_object_class_index = 0
# Global variables
recent_object_detection_predictions = {}  # Dictionary to store recent object detection predictions for each camera
recent_video_classification_predictions = {}  # Dictionary to store recent video classification predictions for each camera
activities_to_detect = {}
client_token = None
max_recent_predictions = 30  # Maximum number of recent predictions to store
RECENT_PREDICTIONS_SIZE = 30
# Define a variable to store the notification sending status (True: enabled, False: disabled)
send_notifications_enabled = True

last_notification_time = None
delay_between_notifications = 10

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
object_detection_threshold = 0.5
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
            print(predicted_class)
            title = f"Activity detected"
            """if predicted_class in activities_to_detect:
                # Send notification
                title = f"Activity detected"
                send_notification(camera_ip, clip_id, title)"""
            # Generate clip ID
            clip_id = f"{camera_ip}_clip_{clip_index}"
            send_notification(camera_ip, clip_id, title)

            # Save video clip with ID
            recent_video_classification_predictions.setdefault(camera_ip, []).append({'clip_id': clip_id, 'clip': frames[clip_index], 'prediction': {'predicted_class': predicted_class, 'confidence_score': float(confidence_score)}})

        return predictions_list

    except Exception as e:
        print(f"Error during video classification: {e}")
        return []

# Function to send Firebase push notification
#def send_push_notification(camera_ip, predictions):

@app.route('/save_token', methods=['POST'])
def save_token():
    global client_token
    data = request.form.get('token')
    client_token = data

    # Return a 200 OK response with the JSON message
    return jsonify(message="Token saved successfully."), 200


@app.route('/get_annotated_image', methods=['POST'])
def get_annotated_image():
    try:
        # Get JSON data from the request body
        data = request.json
        prediction_id = data.get('prediction_id')
        camera_ip = data.get('camera_ip')

        print(camera_ip)
        print(prediction_id)

        if camera_ip in recent_object_detection_predictions:
            predictions = recent_object_detection_predictions[camera_ip]
            for prediction in predictions:
                if prediction['prediction_id'] == prediction_id:
                    annotated_image = prediction['image']
                    base64_image = base64.b64encode(annotated_image).decode('utf-8')
                    return jsonify({'annotated_image_base64': base64_image})

            return jsonify(error='Prediction ID not found for the specified camera IP'), 404
        else:
            return jsonify(error='No recent predictions found for the specified camera IP'), 404

    except Exception as e:
        traceback.print_exc()
        error_message = f"Error retrieving annotated image: {e}"
        print(error_message)
        return jsonify(error=error_message), 500


# Endpoint for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    """
    Performs object detection on a batch of images and stores predictions
    with bounding boxes for harmful objects.

    Returns:
        A JSON response indicating success or an error message with status code.
    """

    try:
        print("called")
        # Read camera IP from the request
        camera_ip = request.form.get('camera_ip')

        camera_name = request.form.get('camera_name')

        # Read image files from the batch
        frames = [request.files[f'frame{i}'].read() for i in range(batch_size)]

        # Convert frames to NumPy arrays
        imgs = [cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in frames]

        # Perform object detection on the batch of frames
        results_batch = perform_object_detection_batch(imgs)

        # Process object detection results
        for frame_index, results in enumerate(results_batch):
            annotated_frame = imgs[frame_index].copy()  # Copy original frame for annotation
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract confidence and class index
                    confidence = box.conf[0].item()
                    print(confidence)
                    cls = int(box.cls[0])
                    # Check for harmful object
                    if cls == harmful_object_class_index and confidence > object_detection_threshold:

                        prediction_id = str(uuid.uuid4())

                        title = f"Harmful object detetcted in, {camera_name}"

                        # Draw bounding box on the annotated frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        prediction_id = str(uuid.uuid4())
                        # Save annotated frame to recent predictions
                        encoded_frame = cv2.imencode('.jpg', annotated_frame)[1].tobytes()
                        recent_object_detection_predictions.setdefault(camera_ip, []).append({
                            'confidence_score': confidence,
                            'image': encoded_frame,
                            'prediction_id':prediction_id
                        })

                        send_notification(camera_ip,prediction_id,title)

        # Limit the recent predictions size for the specific camera
        if camera_ip is not None and camera_ip in recent_object_detection_predictions:
            recent_object_detection_predictions[camera_ip] = recent_object_detection_predictions[camera_ip][-RECENT_PREDICTIONS_SIZE:]

        return 'Object detection completed'

    except Exception as e:
        traceback.print_exc()
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


@app.route('/enable_notifications', methods=['POST'])
def enable_notifications():
  global send_notifications_enabled
  send_notifications_enabled = True
  return jsonify({'message': 'Notification sending enabled.'}), 200


@app.route('/disable_notifications', methods=['POST'])
def disable_notifications():
  global send_notifications_enabled
  send_notifications_enabled = False
  return jsonify({'message': 'Notification sending disabled.'}), 200



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
        #recent_video_classification_predictions[camera_ip] = recent_video_classification_predictions[camera_ip][-RECENT_PREDICTIONS_SIZE:]

        # Send push notification for harmful objects
        #send_push_notification(camera_ip, predictions_list)

        return jsonify(success=True)

    except Exception as e:
        print(f"Error during video classification: {e}")
        return jsonify(error=str(e)), 500

@app.route('/classify_video_tf', methods=['POST'])
def classify_video_endpoint():
    try:
        # Read camera IP from the request
        camera_ip = request.form.get('camera_ip')

        # Read frames from the request
        frames = [request.files[f'frame{i}'].read() for i in range(batch_size)]

        # Convert frames to NumPy arrays
        imgs = [cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) for frame in frames]

        # Perform video classification
        predictions = classify_video_tf(imgs)

        # Print predictions (for testing)
        for label, p in predictions:
            print(f'{label:20s}: {p:.3f}')

        # Send push notification for harmful objects
        # send_push_notification(camera_ip, predictions)

        return jsonify(success=True)

    except Exception as e:
        print(f"Error during video classification: {e}")
        return jsonify(error=str(e)), 500

def send_notification(camera_ip, prediction_id, title):
    global last_notification_time

    # Get the current time
    current_time = time.time()

    # Check if it's been enough time since the last notification
    if last_notification_time is None or current_time - last_notification_time >= delay_between_notifications:
        # Send the notification
        body = f"{camera_ip} | {prediction_id}"

        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=client_token,
        )
        response = messaging.send(message)

        # Update the last notification time
        last_notification_time = current_time


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
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Generate a unique ID using uuid library
    processed_prediction = {
      "id": unique_id,
      "confidence": prediction["confidence_score"],
      "timestamp": current_timestamp,  # Format timestamp for readability
      "image": base64.b64encode(prediction["image"]).decode('utf-8') if "image" in prediction else None,  # Handle optional image data
    }
    processed_predictions.append(processed_prediction)

  return jsonify(processed_predictions)


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
