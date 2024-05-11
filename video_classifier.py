import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pathlib

id = 'a2'
mode = 'stream'
version = '3'
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
# Load the TensorFlow model from the local directory
model_path = 'archive'
model = hub.load(hub_url)

# Get labels
labels_path = 'kinetics_600_labels.txt'  # Provide the path to the labels file
labels_path = pathlib.Path(labels_path)
lines = labels_path.read_text().splitlines()
KINETICS_600_LABELS = np.array([line.strip() for line in lines])

# Define constants
batch_size = 10
image_size = (224, 224)

# Function to perform video classification
def classify_video_tf(frames):
    frames_tensor = []
    for frame in frames:
        # Resize the frame
        frame = cv2.resize(frame, image_size)

        # Convert frame to BGR -> RGB (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Change dtype to float32 and normalize to [0, 1]
        frame = tf.cast(frame, tf.float32) / 255.

        frames_tensor.append(frame)

    # Convert list of frames to a tensor
    frames_tensor = tf.convert_to_tensor(frames_tensor)

    print("Shape of frames_tensor before adding batch axis:", frames_tensor.shape)  # Add this print statement

    # Add this print statement

    initial_state = model.init_states(frames_tensor[tf.newaxis, ...].shape)
    inputs = initial_state.copy()
    inputs['image'] = frames_tensor[tf.newaxis, 0:1, ...]
    logits, new_state = model(inputs)
    logits = logits[0]
    probs = tf.nn.softmax(logits, axis=-1)
    print(probs)
    # Get top predictions
    predictions = get_top_k(probs)

    return predictions



# Get top_k labels and probabilities
def get_top_k(probs, k=5, label_map=KINETICS_600_LABELS):
    """Outputs the top k model labels and probabilities on the given video.

    Args:
      probs: probability tensor of shape (num_frames, num_classes) that represents
        the probability of each class on each frame.
      k: the number of top predictions to select.
      label_map: a list of labels to map logit indices to label strings.

    Returns:
      a tuple of the top-k labels and probabilities.
    """
    print("Shape of probs tensor:", probs.shape)

    # Sort predictions to find top_k
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:, :k]
    # collect the labels of top_k predictions
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    # decode lablels
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    # top_k probabilities of the predictions
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return zip(top_labels, top_probs)

