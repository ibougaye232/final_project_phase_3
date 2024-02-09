import numpy as np
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

from numpy import load

data=np.load("C:/Users/ass85/PycharmProjects/face_recognition_project/.venv/Scripts/images_dataset.npz")

images=data["images"]
labels=data["labels"]
print(labels)
images = preprocess_input(images)

# Charger le modèle VGG16 pré-entraîné
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

embeddings = vgg_model.predict(images)

#np.savez_compressed("embeddings_images_dataset.npz", images=embeddings, labels=labels)

print(embeddings.shape)