import pandas as pd

# MiniBatchKMeans es un algoritmo que gasta menos recursos que K_means

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":

    dataset = pd.read_csv('datasets/candy.csv')
    print(dataset.head(10))

    X = dataset.drop('competitorname', axis=1)

#batch_size = de a cuantos datos irá formando los grupos
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))

    dataset['group'] = kmeans.predict(X)

    print(dataset)