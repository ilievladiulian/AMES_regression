import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LassoCV
import feature_generator

def extract_relevant_features_by_correlation(df_train, df_test):
    plt.figure()
    sns.heatmap(df_train.corr())
    plt.show(block=False)

    print(df_train.corr()['SalePrice'].sort_values(ascending=False).head(22))

    # sort descending by correlation with SalePrice
    # df_train.corr()['SalePrice'].sort_values(ascending=False)
    #
    # the first 20:
    #
    # exterqual, grlivarea, overallqual, kitchenqual, bsmtqual
    # garagecars, totalbsmtsf, 1stflrsf, yearbuild
    # fullbath, yearremodadd, foundation, totrmsabvgrd, garagefinish
    # garageyrblt, masvnrarea, fireplaces, heatingqc, neighborhood, saletype
    #
    # ignore GarageArea because it conveys similar information to GarageCars, and include SaleType

    interest_numerical_features = ['ExterQual', 'GrLivArea', 'OverallQual', 'KitchenQual', 'BsmtQual', 'GarageCars',
                        'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'FullBath', 'YearRemodAdd', 'TotRmsAbvGrd',
                        'GarageFinish', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'HeatingQC', 'SalePrice']
    interest_categorical_features = ['Neighborhood', 'Foundation', 'SaleType']

    df_train_corr = df_train[interest_numerical_features]
    df_test_corr = df_test[interest_numerical_features]

    for category in interest_categorical_features:
        dfDummies = pd.get_dummies(df_train[category], prefix = category)
        df_train_corr = pd.concat([df_train_corr, dfDummies], axis=1)
        
        dfDummies = pd.get_dummies(df_test[category], prefix = category)
        df_test_corr = pd.concat([df_test_corr, dfDummies], axis=1)

    plt.figure()
    sns.heatmap(df_train_corr.corr())
    plt.show(block=False)
    return df_train_corr, df_test_corr, interest_numerical_features, interest_categorical_features

def plot_distributions_for_numerical_features(df, numerical_features):
    df[numerical_features].hist()
    plt.show(block=False)

def lasso_regression_filtered_columns(df):
    alphas = [10000, 50000, 10e5, 5*10e5, 10e6]

    non_zero = []

    for alpha in alphas:
        X_train = pd.DataFrame(df[df.columns.difference(['SalePrice'] + feature_generator.categorical_features)])
        y_train = pd.DataFrame(df['SalePrice'])

        lasso = LassoCV(n_alphas=1, alphas=[alpha])
        lasso.fit(X_train,y_train)
        
        print(lasso.coef_)
        plt.figure()
        plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
        plt.show(block=False)

        # drop columns where lasso coefficient is 0
        non_zero_columns = df.drop(df.columns.difference(['SalePrice'] + feature_generator.categorical_features)[np.where(np.abs(lasso.coef_) == 0)[0]], axis=1)
        # drop redundant categorical and target columns
        non_zero_columns = non_zero_columns.drop(feature_generator.categorical_features + ['SalePrice'], axis=1)

        non_zero = []

    # as alpha increases, the number of zero valued lasso coefficients increases -> we are interested only in the first value of alpha (for most non-zero columns)
    return non_zero[0]

def select_features_of_interest(df_train_corr, df_train_lasso):
    print('a')

