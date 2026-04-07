import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# 1 - Carregar base
# ==========================================
path = r"C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\marise\sonar_features_2026_03_25.csv"
dataset = pd.read_csv(path)

print("Formato da base:", dataset.shape)

# ==========================================
# 2 - Remover colunas de texto
# ==========================================
# Remover image_path
dataset = dataset.drop(columns=["image_path"])

# ==========================================
# 3 - Tratar valores NaN
# ==========================================
dataset = dataset.fillna(0)

# ==========================================
# 4 - Separar X e y
# ==========================================
X = dataset.drop(columns=["label"]).values
y = dataset["label"].values

# Converter label texto para número
# negative = 0 | positive = 1
y = np.where(y == "positive", 1, 0)

# ==========================================
# 5 - Dividir treino e teste
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

# ==========================================
# 6 - Criar modelo
# ==========================================
modelo = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

# ==========================================
# 7 - Treinar modelo
# ==========================================
modelo.fit(X_train, y_train)

# ==========================================
# 8 - Previsões
# ==========================================
y_pred = modelo.predict(X_test)

# ==========================================
# 9 - Avaliação
# ==========================================
print("\n=== MATRIZ DE CONFUSÃO ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== ACURÁCIA ===")
print(accuracy_score(y_test, y_pred))

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred))

# ==========================================
# 10 - Testar várias amostras
# ==========================================
print("\n=== TESTE COM VÁRIAS AMOSTRAS ===")

amostras = X_test[0:10]
valores_reais = y_test[0:10]
predicoes = modelo.predict(amostras)

for i in range(len(predicoes)):
    real = "Bomba" if valores_reais[i] == 1 else "Não bomba"
    previsto = "Bomba" if predicoes[i] == 1 else "Não bomba"
    print(f"Amostra {i} -> Real: {real} | Previsto: {previsto}")