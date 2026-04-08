import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ()
df = pd.read_csv("Afegan_raw.csv", index_col = 0, delimiter = ";")
df.head (10)
df.shape

# Passo a passo para auto-escalar os dados
# Calculando a media de cada variavel
dfm = df.mean()

# Calculando o desvio padrao de cada variavel
dfdp = df.std()

# Auto-escalando os dados
dfauto = (df-dfm)/dfdp
dfauto.head(10)


# OUTRO JEITO
# from sklearn.preprocessing import StandardScaler
# scale_obj = StandardScaler()
# dfauto = scale_obj.fit_transform(dfauto.astype(float))
# display(dfauto)

# Calculo da matriz de correlacoes
corr = dfauto.iloc[:,0:17].corr()
corr

# HEATMAP
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels= corr.columns,
            cmap = "YlGnBu"
            )


# DECONPONTO A MATRIZ DE VARIANCIAS E COMVARIANCIAS EM COMPONENTES PRINCIPAIS
X = np.asarray(dfauto.iloc[:,0:16])
S = np.cov(X)

# VAROAMCIAS DE CADA VARIAVEL
np.diagonal(X)



from sklearn.decomposition import PCA
pca = PCA(n_components=10)

# TREINA MODELO
pca.fit(X)
pca.components_

