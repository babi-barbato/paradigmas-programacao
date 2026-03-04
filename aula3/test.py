from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# 1. Carregar os dados
data = load_iris()
X = data.data  # Características (comprimento da pétala, etc.)
y = data.target # O que queremos prever (espécie da flor)

# 2. Dividir em treino (80%) e teste (20%) com um novo random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11) # Changed random_state

# 3. Criar o modelo com um novo random_state para o classificador
modelo = RandomForestClassifier(random_state=0) # Changed random_state

# 4. O Coração do Processo: MODEL FIT
# Aqui o modelo "estuda" os dados de treino para aprender os padrões
modelo.fit(X_train, y_train)

# 5. Fazer previsões com o que ele aprendeu
previsoes = modelo.predict(X_test)

# 6. Calcular métricas
acuracia = accuracy_score(y_test, previsoes)
# Usamos average='macro' pois temos 3 tipos de flores (multiclasse)
precisao = precision_score(y_test, previsoes, average='macro', zero_division=0) # Added zero_division to handle potential warnings

print(f"Acurácia: {acuracia * 100:.2f}%")
print(f"Precisão: {precisao * 100:.2f}%")