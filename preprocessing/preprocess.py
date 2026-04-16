import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    # Column names for NSL-KDD
    columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label",
        "difficulty"
    ]

    # Load training and test data
    train_df = pd.read_csv("data/raw/KDDTrain+.txt", header=None)
    test_df = pd.read_csv("data/raw/KDDTest+.txt", header=None)

    # Assign column names
    train_df.columns = columns
    test_df.columns = columns

    # Convert labels to binary (0 = normal, 1 = attack)
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == "normal" else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == "normal" else 1)

    # Combine for consistent encoding
    combined = pd.concat([train_df, test_df], axis=0)

    # Encode categorical features
    for col in combined.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    # Split back into train and test
    train_df = combined.iloc[:len(train_df)]
    test_df = combined.iloc[len(train_df):]

    # Separate features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test