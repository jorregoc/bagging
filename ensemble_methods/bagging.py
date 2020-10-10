import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    
    dt_heart = pd.read_csv('datasets/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35)

    #compararemos el score del modelo normal vs el ensemble

    #modelo Kneighbors
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('='*64)
    print(accuracy_score(knn_pred, y_test))

    #modelo de ensemble (bagging)
    #base_estimator = en qué estimador va a estar basado el método
    #n_estimators = cuantos modelos vamos a utilizar
    bag_class = BaggingClassifier(base_estimator = KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print('='*64)
    print(accuracy_score(bag_pred, y_test))

    #Si comparamos con diferentes modelos al tiempo podemos usar el ciclo for
    classifier = {
        'KNeighbors': KNeighborsClassifier(),
        'LinearSCV': LinearSVC(),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for name, estimator in classifier.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=5).fit(X_train, y_train)
        bag_pred = bag_class.predict(X_test)

        print('Accuracy Bagging with {}:'.format(name), accuracy_score(bag_pred, y_test))
        print('')

