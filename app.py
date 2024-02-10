import numpy as np
import streamlit as st
import pickle
import cv2
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default .xml")
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tester import main

def extract_face(img, output_size=(160, 160)):
    try:
        # Utilisez OpenCV pour lire l'image correctement
        img2 = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

        img_gray =img2


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
    # Charger le modèle VGG16 pré-entraîné une seule fois en dehors de la fonction
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

    # Ajuster la forme des images pour correspondre aux attentes du modèle VGG16
    images = np.expand_dims(images, axis=0)

    # Utiliser le modèle VGG16 pré-entraîné pour extraire les embeddings (caractéristiques)
    embeddings = vgg_model.predict(images)

    # Aplatir les embeddings en un vecteur 1D
    embeddings_flat = embeddings.reshape(1,-1)

    # Normaliser les embeddings
    embeddings_normalized = embeddings_flat

    return embeddings_normalized


    return embeddings2



with open("modeler3.pkl", "rb") as file:
    model = pickle.load(file)


st.title("Welcome to my final project, a facial recognition app")
st.text("choose the player you want the AI to recognize by specifying his ID")


img=st.file_uploader("upload an image", type=[".png",".jpg"])


if img is not None:
    predicted=main(img)

    if st.button("click to validate") :
        st.write(predicted)
        if predicted == 0:
            name = "Kylian Mbappé"
            story = "Ayant grandi à Bondy en banlieue parisienne, Mbappé commence sa carrière professionnelle en 2015 à l’AS Monaco où il démontre sa précocité et remporte la Ligue 1 2016-2017. À l’issue de cette saison, il signe au Paris Saint-Germain pour un transfert de 180 millions d’euros, le deuxième montant le plus élevé de l’histoire derrière Neymar. Avec le PSG, l’attaquant remporte de nombreux trophées nationaux dont cinq championnats, atteint la finale de la Ligue des Champions en 2020 et devient le meilleur buteur du club. Meilleur scoreur du championnat de France à cinq reprises, il est le footballeur le plus récompensé aux trophées UNFP, en étant sacré quatre fois meilleur joueur de Ligue 1, et trois fois meilleur espoir."
        elif predicted == 1:
            name = "Lionel Messi"
            story = "Messi commence le football dans sa ville natale de Rosario en Argentine. Atteint d'un problème de croissance qui nécessite un traitement hormonal, il rejoint à treize ans le FC Barcelone, en Espagne, dont il devient un joueur emblématique. Il y remporte un succès exceptionnel pendant ses dix-sept saisons en équipe première, avant de poursuivre sa carrière au Paris Saint-Germain puis à l’Inter Miami."
        else:
            name = "Cristiano Ronaldo"
            story = "Considéré comme l'un des meilleurs footballeurs de l'histoire, il est avec Lionel Messi (avec qui une rivalité sportive est entretenue) l’un des deux seuls à avoir remporté le Ballon d'or au moins cinq fois. Auteur de plus de 870 buts en plus de 1 200 matchs en carrière, Cristiano Ronaldo est le meilleur buteur de l'histoire du football selon la Fédération internationale de football association (FIFA). Il est également le meilleur buteur de la Ligue des champions de l'UEFA, des coupes d'Europe, du Real Madrid, du derby madrilène, de la Coupe du monde des clubs de la FIFA et de la sélection portugaise, dont il est le capitaine officiel depuis 2008. Premier joueur à avoir remporté le Soulier d'or européen à quatre reprises, il est également le meilleur buteur de l'histoire du championnat d'Europe des nations (avec 14 buts) devant Michel Platini et détient le record de buts en équipe nationale, avec 128 réalisations."

        st.image(img, caption=f"his name is {name}")
        st.write(f"about the player :\n{story}")
else:
    st.write("try again")
