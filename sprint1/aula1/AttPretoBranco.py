from matplotlib import pyplot as plt
import cv2

# img = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\suave.png') # lê a imagem
img = cv2.imread(r'C:\Users\babib\OneDrive\Documentos\identificar_padroes\aula1\moinhoFlorido.jpg') # lê a imagem
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem P&B', img)

hist = cv2.calcHist([img], [0], None, [256], [0, 256]) # calcula o histograma da imagem
plt.figure()
plt.title('Histograma Preto e Branco')
plt.xlabel('Intensidade de Pixel')
plt.ylabel('Número de Pixels')
plt.plot(hist) # plota o histograma
plt.xlim([0, 256]) # define os limites do eixo x
plt.show() # exibe o histograma

cv2.waitKey(27) # espera até que uma tecla seja pressionada