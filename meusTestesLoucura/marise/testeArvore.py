import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

path = r"C:\Users\babib\OneDrive\Documentos\paradigmas-programacao\testeRegressaoLinear\sonar_data.csv"
dataset = pd.read_csv(path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# M = Mine = 1 (Mina)
# R = Rock = 0 (Não Mina)
y = np.where(y == 'M', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modelo = DecisionTreeClassifier(
    criterion='entropy',
    max_depth= None,
    random_state=42
)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("=== MATRIZ DE CONFUSÃO ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== ACURÁCIA ===")
print(accuracy_score(y_test, y_pred))

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred))

# print("\n=== TESTE COM VÁRIAS AMOSTRAS ===")  
# amostras = X_test[0:15]
# valores_reais = y_test[0:15]
# predicoes = modelo.predict(amostras)
# for i in range(len(predicoes)):
#     real = "Bomba" if valores_reais[i] == 1 else "Rocha"
#     previsto = "Bomba" if predicoes[i] == 1 else "Rocha"
#     print(f"Amostra {i} -> Real: {real} | Previsto: {previsto}")

plt.figure(figsize=(12,6))
plot_tree(
    modelo,
    feature_names=dataset.columns[:-1],
    class_names=["Rocha", "Mina"],
    filled=True,
    rounded=True
)
plt.savefig("arvore_decisao.png")
plt.show()