import PIL
from typing import Tuple
import PIL.Image
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.models import load_model
import cv2
import rizz_rater as rr

# Read the input image 
img = cv2.imread(input("Enter the path to the image you'd like to get rated: ")) 
if img is None:
    raise Exception("That isn't the path to a valid image!") 

face_dims, face, rizz = rr.get_rizz(img)
face_x, face_y, face_w, face_h = face_dims
cv2.imwrite("face.jpg", face)
cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 0, 255), 2) 
cv2.imwrite("detected.jpg", img) 
print(f"{ rizz }/5")