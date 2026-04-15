import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    df = pd.read_csv("data/raw/nsl_kdd.csv")

    # Encode categorical features
    for col in df.select_dtypes(include=['object']):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Binary classification (attack vs normal)
    df['label'] = df['label'].apply(lambda x: 0 if x == "normal" else 1)

    X = df.drop("label", axis=1)
    y = df["label"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)