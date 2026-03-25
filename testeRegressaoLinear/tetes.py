import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==============================
# CONFIG
# ==============================
path = r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\testeRegressaoLinear\2015'
IMG_SIZE = 64

X = []
y = []

# ==============================
# LEITURA DAS IMAGENS
# ==============================
for file in os.listdir(path):
    
    if not file.lower().endswith((".jpg", ".jpeg")):
        continue

    img_path = os.path.join(path, file)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0   # normalização
    img = img.flatten()

    X.append(img)

    # Label baseada no nome do arquivo
    if "bomba" in file.lower():
        y.append(1)
    else:
        y.append(0)

# ==============================
# CONVERSÃO
# ==============================
X = np.array(X)
y = np.array(y)

# ==============================
# DEBUG (MUITO IMPORTANTE)
# ==============================
print("\n===== INFO DATASET =====")
print("Total imagens:", len(X))
print("Shape X:", X.shape)
print("Labels únicas:", np.unique(y))
print("Qtd bomba:", np.sum(y))
print("Qtd não bomba:", len(y) - np.sum(y))

# ==============================
# TREINO / TESTE
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODELO
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# PREDIÇÃO
# ==============================
y_pred = model.predict(X_test)

# ==============================
# MÉTRICAS
# ==============================
mse = mean_squared_error(y_test, y_pred)
print("\n===== RESULTADOS =====")
print("MSE:", mse)

# ==============================
# DEBUG DAS PREDIÇÕES
# ==============================
print("\ny_test:", y_test[:10])
print("y_pred:", y_pred[:10])

# ==============================
# GRÁFICO
# ==============================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Valor Real")
plt.ylabel("Predição")
plt.title("Regressão Linear em Imagens")
plt.grid()
plt.show()