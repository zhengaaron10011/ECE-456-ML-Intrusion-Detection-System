from preprocessing.preprocess import load_and_preprocess
from models.logistic import train_logistic
from models.random_forest import train_rf
from models.neural_net import train_nn
from evaluation.metrics import evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("Training Logistic Regression...")
    log_model = train_logistic(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, "Logistic Regression")

    print("\nTraining Random Forest...")
    rf_model = train_rf(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    print("\nTraining Neural Network...")
    nn_model = train_nn(X_train, y_train)
    evaluate_model(nn_model, X_test, y_test, "Neural Network")

if __name__ == "__main__":
    main()