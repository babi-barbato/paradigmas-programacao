import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1 - Carregar a base de dados
path = r"C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\testeRegressaoLinear\sonar_data.csv"
dataset = pd.read_csv(path)

# 2 - Separar variáveis de entrada (X) e saída (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Converter:
# M = Mine = 1 (Bomba)
# R = Rock = 0 (Não bomba)
y = np.where(y == 'M', 1, 0)

# 3 - Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4 - Criar modelo Árvore de Decisão
modelo = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=None,
    random_state=42
)

# 5 - Treinar o modelo
modelo.fit(X_train, y_train)

# 6 - Fazer previsões
y_pred = modelo.predict(X_test)

# 7 - Avaliar o modelo
print("=== MATRIZ DE CONFUSÃO ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== ACURÁCIA ===")
print(accuracy_score(y_test, y_pred))

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred))

# 8 - Teste manual com uma amostra
print("\n=== TESTE COM VÁRIAS AMOSTRAS ===")

amostras = X_test[0:15]
valores_reais = y_test[0:15]
predicoes = modelo.predict(amostras)

for i in range(len(predicoes)):
    real = "Bomba" if valores_reais[i] == 1 else "Rocha"
    previsto = "Bomba" if predicoes[i] == 1 else "Rocha"

    print(f"Amostra {i} -> Real: {real} | Previsto: {previsto}")