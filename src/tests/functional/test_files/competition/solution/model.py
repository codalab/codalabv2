from sklearn.cluster import KMeans


class Model:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=3)

    def fit(self, X, y):
        self.kmeans.fit(X=X, y=y)

    def predict(self, X):
        return self.kmeans.predict(X)
