import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('datasets/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'generosity', 'corruption', 'dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    #Entre mayor sea el alpha mas se van a penalizar los features
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_Ridge = modelRidge.predict(X_test)

    #calculamos la pérdida con el mean_squared_error comparando los datos de prueba contra la predicción que hicimos
    #a menor pérdida mejor es el modelo para ese tipo de datos
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('Linear loss:', linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('Lasso loss:', lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_Ridge)
    print('Ridge loss:', ridge_loss)

    #Calculando el coef_ podemos ver cuales son los features que cada modelo considera más importantes. 
    #A mayor coef_ mayor importancia le da el modelo a ese feature

    print('='*32)
    print('Coef lasso')
    print(modelLasso.coef_)

    print('='*32)
    print('Coef Ridge')
    print(modelRidge.coef_)