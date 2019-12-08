from sklearn.metrics import mean_squared_error, accuracy_score
from math import sqrt

def measure_metrics(model, X, y):
    y_predicted = model.predict(X)

    measure_rmse(y, y_predicted)
    measure_mse(y, y_predicted)
    
    print('R^2: ' + str(model.score(X, y)))

def measure_rmse(y, y_predicted):
    rmse = sqrt(mean_squared_error(y, y_predicted))
    print('Root mean squared error: ' + str(rmse))

def measure_mse(y, y_predicted):
    print('Mean squared error: ' + str(mean_squared_error(y, y_predicted)))