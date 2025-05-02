import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.layers import StringLookup
import os

# Import the custom CTCLayer
from TextOCRModel import CTCLayer  # Ensure the CTCLayer is imported from your model file

# Load the saved model
model_path = "TextOCRModel.h5.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})

# Ensure the model is in inference mode
model.trainable = False
model.compile()  # Recompile to apply the changes

# Define constants
IMG_WIDTH = 200
IMG_HEIGHT = 50

# Define preprocessing function
def preprocess_image(image_path):
    """
    Preprocess the input image for prediction.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.transpose(image, perm=[1, 0, 2])  # Transpose to match training format
    return tf.expand_dims(image, axis=0)  # Add batch dimension

# Define prediction decoding function
def decode_prediction(pred_label, num_to_char):
    """
    Decode the predicted label into a readable string.
    """
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
    decoded = tf.keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0]
    decoded_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8")
    return decoded_text.replace("[UNK]", " ").strip()

# Load the character mappings
unique_characters = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))  # Adjust based on training
char_to_num = StringLookup(vocabulary=list(unique_characters), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Predict function
def predict_image(image_path):
    """
    Predict the text from the given image.
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return decode_prediction(prediction, num_to_char)

# Example usage
image_path = "handwriting-recognitionocr/test/TEST_0007.jpg"  # Replace with an actual image path
if os.path.exists(image_path):
    predicted_text = predict_image(image_path)
    print(f"Predicted text: {predicted_text}")
else:
    print(f"Image not found: {image_path}")
