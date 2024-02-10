import numpy as np

import pickle
import cv2
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default .xml")
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def extract_face(img, output_size=(160, 160)):
    try:


        img_gray =img


        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi_gray = img_gray[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi_gray, output_size, interpolation=cv2.INTER_AREA)
            img_f = Image.fromarray(face_roi_resized)
            img_f.save("mba_f2.png")
            return face_roi_resized
        else:
            return None
    except Exception as e:
        print(f"Error in extract_face: {e}")
        return None

def facenet(images):
    try:
        # Charger le modèle VGG16 pré-entraîné une seule fois en dehors de la fonction
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

        # Ajuster la forme des images pour correspondre aux attentes du modèle VGG16
        images = np.expand_dims(images, axis=0)

        # Utiliser le modèle VGG16 pré-entraîné pour extraire les embeddings (caractéristiques)
        embeddings = vgg_model.predict(images)

        # Aplatir les embeddings en un vecteur 1D
        embeddings_flat = embeddings.reshape(1, -1)

        # Normaliser les embeddings
        embeddings_normalized = embeddings_flat

        return embeddings_normalized
    except:
        return None

def main(imgvb):
    with open("modeler3.pkl", "rb") as file:
        model = pickle.load(file)

    img = cv2.imdecode(np.frombuffer(imgvb.read(), np.uint8), cv2.IMREAD_COLOR)

    img2 = extract_face(img)

    img3 = facenet(img2)

    pred = model.predict(img3)

    return pred






