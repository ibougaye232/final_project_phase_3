import numpy as np
import streamlit as st
import joblib

data=np.load("C:/Users/ass85/PycharmProjects/face_recognition_project/.venv/Scripts/images_dataset.npz")

data2=np.load("C:/Users/ass85/PycharmProjects/face_app_2/final_embeddings_images_dataset.npz")

model = joblib.load('C:/Users/ass85/PycharmProjects/face_app_2/trained_model2.joblib')
images=data["images"]
images2=data2["images"]
print(images2)
labels=data["labels"]

st.title("Welcome to my final project, a facial recognition app")
st.text("choose the player you want the AI to recognize by specifying his ID")



for i in (0,36,69):
    a=i+35
    if labels[i]=="messi":
        a=68
    st.image(images[i],caption=f"{labels[i]}'s ID is between{i} and {a}",width=100)


ID=st.number_input("select the player by specifying the ID")

if st.button("click to validate"):
    if model.predict(images2[int(ID)].reshape(1, -1))==0:
        name="Kylian Mbappé"
        story="Ayant grandi à Bondy en banlieue parisienne, Mbappé commence sa carrière professionnelle en 2015 à l’AS Monaco où il démontre sa précocité et remporte la Ligue 1 2016-2017. À l’issue de cette saison, il signe au Paris Saint-Germain pour un transfert de 180 millions d’euros, le deuxième montant le plus élevé de l’histoire derrière Neymar. Avec le PSG, l’attaquant remporte de nombreux trophées nationaux dont cinq championnats, atteint la finale de la Ligue des Champions en 2020 et devient le meilleur buteur du club. Meilleur scoreur du championnat de France à cinq reprises, il est le footballeur le plus récompensé aux trophées UNFP, en étant sacré quatre fois meilleur joueur de Ligue 1, et trois fois meilleur espoir."
    elif model.predict(images2[int(ID)].reshape(1, -1))==1:
        name="Lionel Messi"
        story="Messi commence le football dans sa ville natale de Rosario en Argentine. Atteint d'un problème de croissance qui nécessite un traitement hormonal, il rejoint à treize ans le FC Barcelone, en Espagne, dont il devient un joueur emblématique. Il y remporte un succès exceptionnel pendant ses dix-sept saisons en équipe première, avant de poursuivre sa carrière au Paris Saint-Germain puis à l’Inter Miami."
    else:
        name="Cristiano Ronaldo"
        story="Considéré comme l'un des meilleurs footballeurs de l'histoire, il est avec Lionel Messi (avec qui une rivalité sportive est entretenue) l’un des deux seuls à avoir remporté le Ballon d'or au moins cinq fois. Auteur de plus de 870 buts en plus de 1 200 matchs en carrière, Cristiano Ronaldo est le meilleur buteur de l'histoire du football selon la Fédération internationale de football association (FIFA). Il est également le meilleur buteur de la Ligue des champions de l'UEFA, des coupes d'Europe, du Real Madrid, du derby madrilène, de la Coupe du monde des clubs de la FIFA et de la sélection portugaise, dont il est le capitaine officiel depuis 2008. Premier joueur à avoir remporté le Soulier d'or européen à quatre reprises, il est également le meilleur buteur de l'histoire du championnat d'Europe des nations (avec 14 buts) devant Michel Platini et détient le record de buts en équipe nationale, avec 128 réalisations."

    st.image(images[int(ID)],caption=f"his name is {name}")
    st.write(f"about the player :\n{story}")