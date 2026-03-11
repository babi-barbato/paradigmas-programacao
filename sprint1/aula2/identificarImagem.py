import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas

# 1- Converter imagem em tons de cinza
imagemOrig = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\bolasDeNatal.jpg')
imagemPb = cv2.cvtColor(imagemOrig, cv2.COLOR_BGR2GRAY)

# 2- Aplica o blur na imagem
borrao = cv2.GaussianBlur(imagemPb, (185, 185), 0)

# 3- Binarizar Imagem
T = mahotas.thresholding.otsu(borrao)
print(T)
bin = borrao.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)

# 4-Calcula limite inferior e superior para o canny usando a medida dos pixels
v = np.median(borrao) 
lower = int(max(0, 0.66 * v))
upper = int(max(255, 1.33 * v))

# 5- Aplica o canny com dois conjuntos de thresholds
bordas = cv2.Canny(bin, lower, upper)

# 6- Identificar os contornos
contornos, hierarquia = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# RETR_EXTERNAL: Busca apenas o contorno de fora ignora o que tem dentro
# CHAIN_APPROX_SIMPLE: Cadeia smples aproximada, ele vai trazer só os valores 
# que precisa para trazer o contorno, trás só que é realmente necessário

# 7- Desenhar na imagem original
cv2.drawContours(imagemPb, contornos, -1, (255, 0, 0), 2) # -1 define que quero pegar todas as bordas | cor da linha | espessura da linha 

# Montagem do mosaico final
ladolado = np.vstack([
    np.hstack([imagemPb, borrao]),
    np.hstack([bin, bordas])
])

# Exibição
# plt.imshow("Passo a Passo", ladolado)
# plt.imshow("Objetos " + str(len(contornos)), ladolado)
# plt.imshow("Objetos " + str(len(contornos)), imagemOrig)
# cv2.waitKey(27)

plt.figure(figsize=(10,8))
plt.imshow(imagemPb)
plt.title("Objetos detectados: " + str(len(contornos)))
plt.axis("off")
plt.show()