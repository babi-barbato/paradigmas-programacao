import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas

imagem = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\moinhoFlorido.jpg')

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

imagem0 = imagem.copy()

# BINARIZAÇÃO COM THRESHOLD FIXO
# ret, bin = cv2.threshold(imagem, 160, 255, cv2.THRESH_BINARY) # Se quiser inverter: THRESH_BINARY_INV

# THRESHOLD OTSU
T = mahotas.thresholding.otsu(imagem0)
print(T)
bin0 = imagem.copy()
bin0[bin0 > T] = 255
bin0[bin0 < 255] = 0
bin0 = cv2.bitwise_not(bin0)

plt.figure(figsize=(8, 6))
# plt.imshow(bin, cmap='gray')
plt.imshow(bin, cmap='gray')
plt.title("Binarizaçao da imagem por threshold fixo")
plt.axis("off")
plt.show()