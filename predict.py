### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    X, y = prep_data(df)

    reg = load("reg.joblib")
    predictions = reg.predict(X)

    return predictions


if __name__ == "__main__":
    df = pd.read_csv("fish_holdout.csv")
    X, ho_truth = prep_data(df)

    pl = PolynomialFeatures(degree=2)
    X = pl.fit_transform(X)

    reg = load("reg.joblib")
    ho_predictions = reg.predict(X)

    print(ho_predictions)
    print(ho_truth)

    ho_mse = mean_squared_error(ho_truth, ho_predictions)
    print(ho_mse)