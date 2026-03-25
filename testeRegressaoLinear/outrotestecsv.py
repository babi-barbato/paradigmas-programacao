import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# =========================
# Ler dataset
# =========================
path1 = r'C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\testeRegressaoLinear\sonar_data.csv'
df = pd.read_csv(path1, header=None)

# Última coluna é a classe
X = df.iloc[:, 0:60].values
y = df.iloc[:, 60].values

# Converter R e M para números
y = np.where(y == 'M', 1, 0)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Regressão Linear
# =========================
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

print("\nRegressão Linear")
print("MSE:", mean_squared_error(y_test, y_pred_reg))

# =========================
# Regressão Logística
# =========================
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)

y_pred_log = log.predict(X_test)

print("\nRegressão Logística")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred_log))

# =========================
# Gráfico regressão
# =========================
plt.scatter(y_test, y_pred_reg)
plt.xlabel("Valor Real")
plt.ylabel("Predição")
plt.title("Regressão Linear - Sonar")
plt.show()