import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas

imagem = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\bolasTest.png')

# Converte para escala de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica desfoque gaussiano
borrao = cv2.GaussianBlur(imagem_gray, (5, 5), 0)

# Calculando os limite inferior e superior para o canny usando mediana
v = np.median(borrao) 
lower = int(max(0, 0.66 * v))
upper = int(max(255, 1.33 * v))

# Aplicaçao do canny com dois conjuntos de thresholds
cannyA = cv2.Canny(borrao, lower, upper)
cannyB = cv2.Canny(borrao, 70, 200)

# Montagem do mosaico final
final = np.vstack([
    np.hstack([cannyA, cannyB]),
    np.hstack([imagem_gray, borrao])
])

# Exibição
plt.figure(figsize=(10, 8))
plt.imshow(final, cmap='gray')
plt.title("Detector de bordas canny")
plt.axis("off")
plt.show()

