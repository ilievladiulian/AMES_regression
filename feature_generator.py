import pandas as pd
import dictionaries

categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
                        'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'Electrical', 'PavedDrive', 'RoofMatl',
                        'GarageType', 'SaleType', 'SaleCondition', 'CentralAir', 'MasVnrType', 'MoSold']

def simplify_features(df):
    # add simple categories
    df = df.replace({
        'OverallQual': dictionaries.fuzzy_dict,
        'OverallCond': dictionaries.fuzzy_dict
    })

    # replace simple categories with ordered numbers
    df = df.replace({
        'OverallQual': dictionaries.fuzzy_dict_simple,
        'OverallCond': dictionaries.fuzzy_dict_simple
    })
    return df

def one_hot_encode_categorical_features(df):
    for category in categorical_features:
        df[category] = pd.Categorical(df[category])
        dfDummies = pd.get_dummies(df[category], prefix = category)
        df = pd.concat([df, dfDummies], axis=1)
    return df