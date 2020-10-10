import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == "__main__":

    dataset = pd.read_csv('./datasets/felicidad.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()
    #scoring acepta varios tipos de errores como: r2, accuracy, precision, neg_mean_absolute_error.
    score = cross_val_score(model, X,y, cv= 3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))