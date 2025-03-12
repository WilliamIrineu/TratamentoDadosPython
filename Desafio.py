
# pip install scikit-learn

import sklearn 

from sklearn import datasets

# Etapa 0 - Importar o data set iris, e verificar o que é cada key presente no iris

iris = datasets.load_iris()
iris

print(iris)


print(iris.keys())

# conjunto de dados
iris.data

# resposta da variavel de classificacao
iris.target

# nome da variavel de classificacao
iris.target_names 

# nome das colunas
iris.feature_names

# praticando

# Etapa 1 - Dados iris para um Data Frame

dados=iris.data
dados

type(dados)

#dados=list(dados.flatten())

dados=list(dados)

import pandas as pd
dados=pd.DataFrame(dados,columns=['1','2','3','4'])
dados

type(dados)

# Etapa 2 - Target para um Data Frame

dados_target=iris.target
dados_target

# como não precisa de quebrar o array, não precisa usar o .flatten()
dados_target=list(dados_target)
dados_target

dados_target=pd.DataFrame(dados_target,columns=['Class'])
dados_target

# agora temos os dados target (as classes como um data frame)

# Etapa 3 - Mudar o nome das colunas no conjunto de dados

# nome das colunas
colunas=iris.feature_names
colunas
type(colunas)# ja está como lista, entao vamos tratar elas

# Para tratamento vou utilizar o conhecimento map e lambda

colunas # observe que tem espaço entre as palavras

# Tratamentos que vou fazer é trocar: " (cm)" por vazio, depois trocar espaço por -

# primeiro tratamento - " (cm)" por vazio
colunas=list(map(lambda x: x.replace(' (cm)',''),colunas))
colunas

# segundo tratamento - espaço por -
colunas=list(map(lambda x: x.replace(' ','-'),colunas))
colunas

# agora temo o nome das colunas do nosso data frame, e em formato de lista, que facilita
# a troca do nome das colunas

# Etapa 4 - Mudar o nome das colunas no conjunto de dados

dados.columns=colunas
dados

# Agora temos os dados com o nome das colunas que queriamos

# Etapa 5 - Adicionar o Class ao nosso conjunto de dados

target_nome=iris.target_names

target_nome

dados_target['Class'] = dados_target['Class'].replace([0,1,2],['setosa', 'versicolor', 'virginica'])
dados_target

# Etapa 6 - Adicionar o Class ao nosso conjunto de dados
# lembrando de usar o axis=1 estamos fazendo por linhas
dados

dados_target

dados_tratados=pd.concat([dados,dados_target],axis=1)
dados_tratados

# agora que tratamos vamos exportar o dados_tratados

dados_tratados.to_csv('iris_dados_tratados.csv',sep=';',encoding='utf-8',index=False)