import os
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage import transform
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Désactiver l'utilisation de GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Charger le modèle préalablement entrainé
model = load_model('myFlower_model.h5')

# Charger et formater l'image à classifier
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
img = Image.open('fleur.jpg')
img = np.array(img).astype('float32')/255
img = transform.resize(img, (224, 224, 3))
img = np.expand_dims(img, axis=0)

# Autre manière de charger et formater l'image (diffère du preprocessing de l'entrainement)
"""
img = image.load_img('fleur.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = img.reshape((1,) + img.shape)
"""

# Effectuer la prédiction
pred_prob = model.predict(img)[0]
pred_class = list(pred_prob).index(max(pred_prob))

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulipe']

print(classes[pred_class])
