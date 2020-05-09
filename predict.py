### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
import pandas as pd
from sklearn.metrics import r2_score

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    X, y = prep_data(df)
 
    reg = load("reg_plr2.joblib")

    predictions = reg.predict(X)

    return predictions

if __name__ == "__main__":
    ho_predictions = predict_from_csv("fish_holdout_demo.csv")
    print(ho_predictions)


#####

## WE WRITE THIS ###

    ho_truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values
    # print(ho_truth)
    ho_mse = r2_score(ho_truth, ho_predictions)
    print(ho_mse)
#####

