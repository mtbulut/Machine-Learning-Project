import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump
from preprocess import prep_data
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv("fish_participant.csv")
X, y = prep_data(df)

polynomial_features = PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)

# lets use linearRegration, Dummy Classifier 
lr = LinearRegression() 

# following is the train part begins
lr.fit(X_poly, y)
# This is going to save the trained data

# Cross Validation 
cross_val_scores = cross_val_score(lr, X_poly, y, scoring="r2", cv=10)
print(cross_val_scores.mean())

# 0.845107255337558

dump(lr, "reg_plr2.joblib")

