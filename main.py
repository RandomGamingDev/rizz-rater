import PIL
from typing import Tuple
import PIL.Image
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.models import load_model
import cv2

def resize_with_pad(image, new_shape, padding_color = (255, 255, 255)) -> np.array:
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return image

def preprocess_image(image, target_size):
    return resize_with_pad(image, target_size) / .255

# Read the input image 
img = cv2.imread(input("Enter the path to the image you'd like to get rated: ")) 
if img is None:
    raise Exception("That isn't the path to a valid image!") 

# Convert into grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# Load the cascade 
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt2.xml") 

# Detect faces 
faces = face_cascade.detectMultiScale(gray, 1.2, 4)
if len(faces) == 0:
    raise Exception("There were no detected faces in the image") 
# We only need the first face
face_x, face_y, face_w, face_h = faces[0]

# Crop out the face
face = img[face_y:face_y + face_h, face_x:face_x + face_w] 
cv2.imwrite("face.jpg", face)

# Calculate the rizz score
model_path = "models/attractiveNet_mnv2.h5"
model = load_model(model_path)
face_tensor = np.array(face)
input_tensor = tf.expand_dims(preprocess_image(face_tensor, (350, 350)), axis=0)
prediction = model.predict(input_tensor)[0][0]
print(f"{ prediction }/5")

# Display the selection from the original image as a red rectangle
cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 0, 255), 2) 
cv2.imwrite("detected.jpg", img) 