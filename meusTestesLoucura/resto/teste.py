import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Exemplo de dados (substitua pelos seus)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # exemplo: y = x²

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criando transformação polinomial
poly = PolynomialFeatures(degree=2)

# Aplicando nos dados de treino
X_train_poly = poly.fit_transform(X_train)

# Criando e treinando o modelo
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Transformando dados de teste
X_test_poly = poly.transform(X_test)

# Previsões
y_pred = model.predict(X_test_poly)

print("Previsões:", y_pred)