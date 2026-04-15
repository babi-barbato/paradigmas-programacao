import numpy as np
import cv2
import matplotlib.pyplot as plt

# Caminho da imagem
pathImg = r'C:/Users/babib/OneDrive/Documentos/paradigmas-programacao/smile.jpg'

# Carregar imagem
imagem = cv2.imread(pathImg)

# Converter para escala de cinza
imagemGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Carregar classificadores
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
smile_xml = cv2.data.haarcascades + 'haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_xml)
smile_cascade = cv2.CascadeClassifier(smile_xml)

# Detectar rostos
faces = face_cascade.detectMultiScale(
    imagemGray,
    scaleFactor=1.2,
    minNeighbors=2,
    minSize=(60, 60)
)

total_sorrisos = 0

# Para cada rosto encontrado
for (x, y, w, h) in faces:
    # Desenhar retângulo no rosto
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 5)

    roi_gray = imagemGray[y:y+h, x:x+w]
    roi_color = imagem[y:y+h, x:x+w]

    # PEGAR APENAS PARTE INFERIOR DO ROSTO (onde fica o sorriso)
    roi_gray_lower = roi_gray[int(h/2):h, :]
    roi_color_lower = roi_color[int(h/2):h, :]

    # Detectar sorriso
    smiles = smile_cascade.detectMultiScale(
        roi_gray_lower,
        scaleFactor=1.9,
        minNeighbors=20,
        minSize=(25, 25)
    )

    total_sorrisos += len(smiles)

    # Desenhar retângulos nos sorrisos
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color_lower, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 5)

print('Sorrisos encontrados:', total_sorrisos)

# Mostrar imagem
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()