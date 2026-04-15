import numpy as np
import cv2
import matplotlib.pyplot as plt

# Vai cair smile na prova SMILE

pathImg = r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\smile.jpg'
# pathImg2 = r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\sprint2\aula5\palmeiras.jpg'     1.3   60,60
path1 = cv2.data.haarcascades + 'haarcascade_smile.xml'

imagem = cv2.imread(pathImg)
imagemGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

face_frontal = cv2.CascadeClassifier(path1)
face_recon = face_frontal.detectMultiScale(
    imagemGray,
    scaleFactor=1.7,
    minNeighbors=20,
    minSize=(60, 60)
)


print('Sorrisos encontrados: ', len(face_recon))

for (x, y, w, h) in face_recon:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 4)


plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()