# Importing some needed modules
import tensorflow as tf 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Loading our trained "HAMARA_PYARA_MODEL"
model = tf.keras.models.load_model('HAMARA_PYARA_MODEL.keras')


# In case you want to change the input image, It's ADVISABLE to change input image name from both testModel.py and this cell.

# Importing needed modules
import cv2 
import tensorflow as tf 
import numpy as np

# Reading image
myImage = cv2.imread('something.png')                                     # CHANGE IMAGE NAME HERE
# Converting image into gray from RGB
convertedImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

# Preparing it for kernel operation
# Resizing and adjusting pixels of the image
resizedImage = cv2.resize(convertedImage, (28,28), interpolation = cv2.INTER_AREA)
# Scaling from 0 to 1
newImage = tf.keras.utils.normalize(resizedImage, axis = 1)

# Kernal operation of CNN layer 
IMG_SIZE = 28
newImage = np.array(newImage).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Getting to know how the input image looks like(Not from any of test sets)
plt.imshow(myImage)

# Predicting from image input outside MNIST test sets
prediction = model.predict(newImage)
predictImage = np.argmax(prediction)
print("HAMARA_PYARA_MODEL say: It's a",predictImage)
