import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas

imagem = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\aula1\moinhoFlorido.jpg')

imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# 1- Gerar dois histogramas um em tons de cinza e outro RGB:
# Histograma RGB
cores = ('r', 'g', 'b')
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Histograma RGB")
plt.xlabel("Intensidade")
plt.ylabel("Pixels")

for i, cor in enumerate(cores):
    hist = cv2.calcHist([imagem], [i], None, [256], [0,256])
    plt.plot(hist, color=cor)
    plt.xlim([0,256])

# Histograma em tons de cinza
plt.subplot(1,2,2)
plt.title("Histograma Tons de Cinza")
plt.xlabel("Intensidade")
plt.ylabel("Pixels")

hist_gray = cv2.calcHist([imagem_gray], [0], None, [256], [0,256])
plt.plot(hist_gray, color='black')
plt.xlim([0,256])

plt.show()


# 2- Utilizar o mesmo histograma do item anterior e gerar o contraste:
contraste = cv2.equalizeHist(imagem_gray)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(contraste, cmap='gray')
plt.title("Imagem com Contraste")
plt.axis("off")

plt.subplot(1,2,2)
hist_contraste = cv2.calcHist([contraste], [0], None, [256], [0,256])
plt.plot(hist_contraste, color='black')
plt.title("Histograma após Contraste")
plt.xlim([0,256])

plt.show()


# 3- Gerar “blur” da foto pelo cálculo da mediana;
blur_mediana = cv2.medianBlur(imagem_gray, 7)

plt.figure(figsize=(6,6))
plt.imshow(blur_mediana, cmap='gray')
plt.title("Blur - Filtro da Mediana")
plt.axis("off")
plt.show()


# 4- Binarizar a foto utilizando o método Otsu;
T = mahotas.thresholding.otsu(imagem_gray)

bin = imagem_gray.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)

plt.figure(figsize=(6,6))
plt.imshow(bin, cmap='gray')
plt.title("Binarização - Método Otsu")
plt.axis("off")
plt.show()