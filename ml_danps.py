#importando as bibliotecas
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn 
import pandas.util.testing as tm
from sklearn import datasets 
from sklearn.model_selection import train_test_split,KFold,cross_val_score, cross_val_predict 
from sklearn.svm import SVC      #importa o algoritmo svm para ser utilizado 
from sklearn import tree         # importa o algoritmo arvore de decisão
from sklearn.linear_model import LogisticRegression #importa o algoritmo de regressão logística
from sklearn.metrics import mean_absolute_error #utilizada para o calculo do MAE
from sklearn.metrics import mean_squared_error #utilizada para o calculo do MSE
from sklearn.metrics import r2_score #utilizada para o calculo do R2
from sklearn import metrics  #utilizada para as métricas de comparação entre os métodos
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 

from sklearn import svm

def meus_imports():
  print ("import pandas as pd #biblioteca utilizada para o tratamento de dados via dataframes",
  "import numpy as np #biblioteca utilizada para o tratamento de valores numéricos (vetores e matrizes)",
  "import matplotlib.pyplot as plt #biblioteca utilizada para construir os gráficos",
  "import seaborn as sn #biblioteca utilizada para os plots mais bonitos ",
  "import pandas.util.testing as tm",
  "from sklearn import datasets #sklearn é uma das lib mais utilizadas em ML, ela contém, além dos datasets, várias outras funções úteis para a análise de dados",
  "from sklearn.model_selection import train_test_split,KFold,cross_val_score, cross_val_predict # esse método é utilizado para dividir o conjunto de dados em grupos de treinamento e test",
  "from sklearn.svm import SVC      #importa o algoritmo svm para ser utilizado ",
  "from sklearn import tree         # importa o algoritmo arvore de decisão",
  "from sklearn.linear_model import LogisticRegression #importa o algoritmo de regressão logística",
  "from sklearn.metrics import mean_absolute_error #utilizada para o calculo do MAE",
  "from sklearn.metrics import mean_squared_error #utilizada para o calculo do MSE",
  "from sklearn.metrics import r2_score #utilizada para o calculo do R2",
  "from sklearn import metrics  #utilizada para as métricas de comparação entre os métodos",
  "import matplotlib.pyplot as plt",
  "import seaborn as sns",
  "import numpy as np",
  "from sklearn.ensemble import RandomForestClassifier",
  "from sklearn.tree import DecisionTreeClassifier ",
  "from sklearn.neighbors import KNeighborsClassifier",
  "from sklearn import svm")

def ler_csv(name, status):
  df = pd.read_csv(name)
  if(status):
    print("#Tamanho do dataset\n", len(df), "\n")
    print("#Preview Dados:\n", df .head(), "\n")
    print("#Estatísticas do dataset\n", df.describe() , "\n")
    nans = df.isna().sum() #contando a quantidade de valores nulos
    print(nans[nans > 0], "\n")
    print("#Campos nulos\n", df.isnull().sum(), "\n")
  return df
   
def apagar_colunas(df, drops): 
  df = df.drop(drops, inplace=True, axis=1) 
  return df

#trabalhando com dados nulos
def nulo_media(df, col):
  df[col].fillna(df[col].mean(), inplace=True) 
  
def nuloToVazio(df, col):
  df[col].fillna("", inplace=True)

def nuloToValor(df, col, valor):
  df[col].fillna(value=valor, inplace=True)

def compararModeloClassificacao(x, y):
  # Separando o dataset entre entradas e saídas
  #x = got_dataset.iloc[:,1:].values
  #y = got_dataset.iloc[:, 0].values
  
  # divide o dataset entre 5 diferentes grupos
  kfold = KFold(n_splits=5, shuffle=True, random_state=42)

  # construindo os modelos de classificação
  modelos = [LogisticRegression(solver='liblinear'), 
    RandomForestClassifier(n_estimators=400, random_state=42), 
    DecisionTreeClassifier(random_state=42), svm.SVC(kernel='rbf', gamma='scale', random_state=42),
    KNeighborsClassifier()]
  
  #utilizando a validação cruzada
  mean=[]
  std=[]
  for model in modelos:
    result = cross_val_score(model, x, y, cv=kfold, scoring="accuracy", n_jobs=-1)
    mean.append(result)
    std.append(result)

  classificadores=['Regressão Logística', 'Random Forest', 'Árvore de Decisão', 'SVM', 'KNN']
  
  plt.figure(figsize=(10, 10))
  for i in range(len(mean)):
    sns.distplot(mean[i], hist=False, kde_kws={"shade": True})
  plt.title("Distribuição de cada um dos classificadores", fontsize=15)
  plt.legend(classificadores)
  plt.xlabel("Acurácia", labelpad=20)
  plt.yticks([])
  plt.show()
