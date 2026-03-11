import numpy as np
import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\moinhoFlorido.jpg')

# Usa o 13 pois ele vai agrupar de 13 em 13 pegar o centro e borrar
# Quanto mais o numero aumenta mais porrado fica / é impar pq par não tem meio
imagem_suavizada = cv2.GaussianBlur(imagem, (13, 13), 0)

plt.figure(figsize=(8, 6))
cv2.imshow(cv2.cvtColor(imagem_suavizada))
cv2.waitKey(0)
