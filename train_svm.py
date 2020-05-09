import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump
import pickle
from preprocess import prep_data
from sklearn.model_selection import cross_val_score
from sklearn import svm


df = pd.read_csv("fish_participant.csv")
X, y = prep_data(df)

# lets use SVM 
clf = svm.SVR()

# following is the train part begins
clf.fit(X, y)
# This is going to save the trained data

# Cross Validation 
cross_val_scores = cross_val_score(clf, X, y, scoring="r2", cv=10)
print(cross_val_scores)

# 0.845107255337558  lr three features
# 0.9292609639190234 lr all features 
# 0.9068366579170707   plr_2 all features 

pickle.dumps(clf)

# dump(lr, "reg.joblib")