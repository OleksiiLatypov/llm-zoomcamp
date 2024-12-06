#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from typing import List, Tuple, Dict
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pickle


with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)


print(dv)

client = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([client])
probability = model.predict_proba(X)[:, 1]
print(probability)

if __name__ == '__main__':
    print()