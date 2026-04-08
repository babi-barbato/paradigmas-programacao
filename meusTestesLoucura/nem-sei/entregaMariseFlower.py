# ============================
# 1 - Importar bibliotecas
# ============================
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================
# 2 - Carregar base Iris
# ============================
iris = load_iris()

X = iris.data
y = iris.target

# ============================
# 3 - Tamanho e balanceamento
# ============================
print("Tamanho do dataset:", X.shape)
print("\nBalanceamento das classes:")
print(pd.Series(y).value_counts())

# ============================
# 4 - Árvore com parâmetros default
# ============================
modelo = DecisionTreeClassifier()

# ============================
# 5 - K-Fold com 3,5,7 e 10 folds
# ============================
folds = [3,5,7,10]

print("\nResultados K-Fold:")
for k in folds:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=kfold)
    print(f"K = {k}")
    print("Acurácia média:", scores.mean())
    print("------------------------")

# ============================
# 6 - GridSearch para melhores hiperparâmetros
# ============================
parametros = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    DecisionTreeClassifier(),
    parametros,
    cv=5,
    scoring='accuracy'
)

grid.fit(X, y)

print("\nMelhores parâmetros:")
print(grid.best_params_)
print("Melhor acurácia:", grid.best_score_)

# ============================
# 7 - Treinar modelo final com melhores parâmetros
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo_final = grid.best_estimator_
modelo_final.fit(X_train, y_train)

y_pred = modelo_final.predict(X_test)

# ============================
# 8 - Matriz de confusão
# ============================
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# ============================
# 9 - VP, VN, FP, FN
# ============================
print("\nVP, VN, FP, FN por classe:")

for i in range(len(cm)):
    VP = cm[i,i]
    FN = cm[i,:].sum() - VP
    FP = cm[:,i].sum() - VP
    VN = cm.sum() - (VP + FN + FP)

    print(f"\nClasse {i}:")
    print("VP:", VP)
    print("FN:", FN)
    print("FP:", FP)
    print("VN:", VN)