from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":


    #cargamos los datos para el modelo
    dt_heart = pd.read_csv('./datasets/heart.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    #para usar el algoritmo PCA debemos normalizar nuestros features
    #La normalización que hace sklearn con StandardScaler es: z = x-u / s donde u:media y s:desviacion estandar
    dt_features = StandardScaler().fit_transform(dt_features)

    #dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size = 0.3, random_state = 1)
    print(X_train.shape)
    print(y_train.shape)


    #Creamos el algoritmo PCA
    #el parametro n_components es el numero de features que queremos seleccionar
    #Lo mandamos a llamar con los datos de entrenamiento
    pca = PCA(n_components=3)
    pca.fit(X_train)

    #Creamos el algoritmo IPCA.
    #IPCA no usa todos los datos a la vez sino que los divide en pequeñas partes llamadas batch y los combina al final
    #con el parametro batch_size se define la cantidad de batch.

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    #creamos la regresión logística
    logistic = LogisticRegression(solver='lbfgs')

    #Aplicamos el algoritmo PCA y IPCA tanto a los datos de prueba como de entrenamiento.
    #entrenamos la regresión logistica
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('Score PCA: ', logistic.score(dt_test, y_test))


    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('Score IPCA: ', logistic.score(dt_test, y_test))
