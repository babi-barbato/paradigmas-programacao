from matplotlib import pyplot as plt
import numpy as np
import cv2

imgagem = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\moinhoFlorido.jpg')
cv2.imshow('Imagem RGB', img)

cores = ('b', 'g', 'r')

plt.figure()
plt.title('Histograma RGB')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Número de Pixels')

for (canal, cor) in enumerate(cores):
    hist = cv2.calcHist([img], [canal], None, [256], [0, 256])
    plt.plot(hist, color=cor)
    plt.xlim([0, 256])

plt.show()
cv2.waitKey(0)
