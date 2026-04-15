from sklearn.neural_network import MLPClassifier

def train_nn(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20)
    model.fit(X_train, y_train)
    return model