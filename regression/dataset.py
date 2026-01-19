import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    
def load_and_prepare_data():
    config = load_config()

    df = pd.read_csv(config["data"]["path"])

    df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors="ignore")

    df["date"] = pd.to_datetime(df["date"])
    df["year_sold"] = df["date"].dt.year
    df["month_sold"] = df["date"].dt.month
    df.drop(columns=["date"], inplace=True)

    current_year = 2024
    df["house_age"] = current_year - df["yr_built"]
    df["renovation_age"] = np.where(df["yr_renovated"] > 0, current_year - df["yr_renovated"], 0)

    df.dropna(inplace=True)

    df["log_price"] = np.log(df["price"])

    y = df["log_price"]
    X = df.drop(columns=["price", "log_price"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

