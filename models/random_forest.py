from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model