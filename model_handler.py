import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import metrics

def run_regression(df_train_final, normalize=False):
    X = pd.DataFrame(df_train_final[df_train_final.columns.difference(['SalePrice'])])
    y = pd.DataFrame(df_train_final['SalePrice'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression(normalize=normalize)
    model.fit(X_train, y_train)

    metrics.measure_metrics(model, X_test, y_test)


