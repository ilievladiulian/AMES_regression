import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, learning_curve
import numpy as np
import matplotlib.pyplot as plt

import metrics

def run_regression(df_train_final, normalize=False):
    X = pd.DataFrame(df_train_final[df_train_final.columns.difference(['SalePrice'])])
    y = pd.DataFrame(df_train_final['SalePrice'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression(normalize=normalize)
    model.fit(X_train, y_train)

    metrics.measure_metrics(model, X_test, y_test)

    return model

def run_regression_with_cv(df_train_final):
    X = pd.DataFrame(df_train_final[df_train_final.columns.difference(['SalePrice'])])
    y = pd.DataFrame(df_train_final['SalePrice'])

    model = LinearRegression()

    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for _, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X.iloc[train,:], y.iloc[train,:])
        scores.append(model.score(X.iloc[test,:], y.iloc[test,:]))

    print('Average R^2 score: ' + str(np.mean(scores)))

def run_regression_with_learning_curve(df_train_final):
    X = pd.DataFrame(df_train_final[df_train_final.columns.difference(['SalePrice'])])
    y = pd.DataFrame(df_train_final['SalePrice'])

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator = LinearRegression(),
        X = X,
        y = y,
        cv = 5
    )
    
    plt.figure()
    plt.plot(train_sizes, train_scores, label = 'Training error')
    plt.plot(train_sizes, validation_scores, label = 'Validation error')
    plt.show(block=False)

