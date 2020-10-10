from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":


    #cargamos los datos para el modelo
    dt_heart = pd.read_csv('../datasets/heart.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    #para usar el algoritmo KernelPCA debemos normalizar nuestros features
    #La normalización que hace sklearn con StandardScaler es: z = x-u / s donde u:media y s:desviacion estandar
    dt_features = StandardScaler().fit_transform(dt_features)

    #dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size = 0.3, random_state = 1)
    
    #Creamos el algoritmo KPCA
    #el parametro n_components es el numero de features que queremos seleccionar
    #el parametro kernel puede ser: polinomial (poly), linear (linear), gausiano (rbf)
    #Lo mandamos a llamar con los datos de entrenamiento
    kpca = KernelPCA(n_components=3, kernel= 'poly')
    kpca.fit(X_train)

    #creamos la regresión logística
    logistic = LogisticRegression(solver='lbfgs')

    #Aplicamos el algoritmo PCA y IPCA tanto a los datos de prueba como de entrenamiento.
    #entrenamos la regresión logistica
    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('Score KPCA: ', logistic.score(dt_test, y_test))