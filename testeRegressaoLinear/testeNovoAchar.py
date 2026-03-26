import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas

# Ler imagem sonar

path = r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\testeRegressaoLinear\2015\0011_2015.jpg'

imagemOrig = cv2.imread(path)
imagemPb = cv2.cvtColor(imagemOrig, cv2.COLOR_BGR2GRAY)

# Melhorar contraste (muito importante para sonar)
imagemPb = cv2.equalizeHist(imagemPb)

# Blur
borrao = cv2.GaussianBlur(imagemPb, (25, 25), 0)

# Threshold Otsu
T = mahotas.thresholding.otsu(borrao)
bin = borrao.copy()
bin[bin > T] = 255
bin[bin <= T] = 0

# Inverter se necessário
bin = cv2.bitwise_not(bin)

# Canny
v = np.median(borrao)
lower = int(max(0, 0.66 * v))
upper = int(min(255, 1.33 * v))
bordas = cv2.Canny(bin, lower, upper)

# Morphology para juntar bordas
kernel = np.ones((5,5), np.uint8)
bordas = cv2.dilate(bordas, kernel, iterations=2)
bordas = cv2.erode(bordas, kernel, iterations=1)

# Contornos
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos pequenos
contornos_filtrados = []
for c in contornos:
    area = cv2.contourArea(c)
    if area > 200:  # ajuste esse valor
        contornos_filtrados.append(c)

# Desenhar
cv2.drawContours(imagemOrig, contornos_filtrados, -1, (0,255,0), 2)

plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(imagemOrig, cv2.COLOR_BGR2RGB))
plt.title("Objetos detectados: " + str(len(contornos_filtrados)))
plt.axis("off")
plt.show()