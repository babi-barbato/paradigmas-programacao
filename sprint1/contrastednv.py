import cv2
import numpy as np

# Ler a imagem
imagem = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\sprint1\images.jpg')

# Ajustar contraste e brilho
alpha = 1.8  # contraste (1.0 = normal)
beta = 0.2     # brilho

imagem_contraste = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Salvar ou visualizar
cv2.imwrite("imagem_contraste.jpg", imagem_contraste)

cv2.imshow("Original", imagem)
cv2.imshow("Com Contraste", imagem_contraste)
cv2.waitKey(0)
cv2.destroyAllWindows()