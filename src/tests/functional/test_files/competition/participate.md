# How to participate

Data has been normalized


#### Seed:
```
class Model:
    def fit(self, X_train, y_train):
        """
        This should handle the logic of training your model
        :param X_train: 7 dimensional np.array of training data
        :param y_train: 1 dimensional np.array of the same length as X_train. Contains classifications of X_train
        """
        pass

    def predict(self, X_test):
        """
        This should handle making predictions with a trained model
        :param X_test: 7 dimensional np.array of testing data
        :return: 1 dimensional np.array of the same length as X_test containing predictions to each point in X_test
        """
        pass

```