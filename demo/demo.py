from preprocessing.preprocess import load_and_preprocess
from models.random_forest import train_rf

def run_demo():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = train_rf(X_train, y_train)

    sample = X_test[0].reshape(1, -1)
    prediction = model.predict(sample)

    print("\n=== DEMO ===")
    print("Prediction:", "ATTACK 🚨" if prediction[0] == 1 else "NORMAL ✅")

if __name__ == "__main__":
    run_demo()