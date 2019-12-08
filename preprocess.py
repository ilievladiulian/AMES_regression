import pandas as pd
import numpy as np
from scipy import stats
import dictionaries

def eliminate_duplicates(df):
    return df.drop_duplicates(subset='Id', keep=False)

def eliminate_outliers(df):
    # columns that intuitively influence the target variable and are also numeric so they can present outliers
    columns_of_interest = ['LotArea', 'YearBuilt', 'GrLivArea']

    # calculate the z-score of those columns along their respective column
    # and if the score is above a certain threshold, remove the rows
    threshold = 3

    z_train = np.abs(stats.zscore(df[columns_of_interest]))
    return df[(z_train < threshold).all(axis = 1)]

def replace_numerical_features_with_categories(df):
    return df.replace({'MoSold': dictionaries.mo_sold_dict, 'MSSubClass': dictionaries.ms_subclass_dict})

def replace_ordered_categories_with_numbers(df):
    return df.replace({
        'PoolQC': dictionaries.quality_dict, 
        'GarageCond': dictionaries.quality_dict, 
        'GarageQual': dictionaries.quality_dict, 
        'GarageFinish': dictionaries.quality_dict, 
        'FireplaceQu': dictionaries.quality_dict, 
        'Functional': dictionaries.quality_dict, 
        'KitchenQual': dictionaries.quality_dict, 
        'HeatingQC': dictionaries.quality_dict, 
        'BsmtFinType2': dictionaries.quality_dict, 
        'BsmtFinType1': dictionaries.quality_dict, 
        'BsmtExposure': dictionaries.quality_dict, 
        'BsmtCond': dictionaries.quality_dict,
        'BsmtQual': dictionaries.quality_dict, 
        'ExterCond': dictionaries.quality_dict, 
        'ExterQual': dictionaries.quality_dict
    })

def handle_missing_data(df):
    # finding the percentage of missing data in the columns that have missing data
    percentage = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    # remove all columns that have more than 40 percent missing data
    df.drop(percentage[percentage > 0.4].index, axis=1, inplace=True)

    # complete the rest with the global mean
    return df.fillna(df.mean())