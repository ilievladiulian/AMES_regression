import pandas as pd
import matplotlib.pyplot as plt

import preprocess
import feature_generator
import analyzer
import model_handler

def load_dataset():
    df_train = pd.read_csv('data/train.csv')

    # test.csv concatenated with benchmark.csv by id
    df_test = pd.read_csv('data/test_full.csv')

    return df_train, df_test

def main():
    df_train, df_test = load_dataset()
    

    # REQUIREMENT 1 - Investigate and preprocess the dataset

    # eliminate duplicates by id
    df_train = preprocess.eliminate_duplicates(df_train)
    df_test = preprocess.eliminate_duplicates(df_test)

    # eliminate outliers (only in train - to not influence training)
    df_train = preprocess.eliminate_outliers(df_train)

    # replace numerical features that are actually categories
    df_train = preprocess.replace_numerical_features_with_categories(df_train)
    df_test = preprocess.replace_numerical_features_with_categories(df_test)

    # replace ordered categories with ordered numbers
    df_train = preprocess.replace_ordered_categories_with_numbers(df_train)
    df_test = preprocess.replace_ordered_categories_with_numbers(df_test)

    # handle missing data - remove columns with more than 40 percent missing data and fill the rest with mean
    df_train = preprocess.handle_missing_data(df_train)
    df_test = df_test.fillna(df_test.mean())


    # REQUIREMENT 2 - Prepare by generating new features in the following ways
    
    # generate new features
    df_train = feature_generator.simplify_features(df_train)
    df_test = feature_generator.simplify_features(df_test)

    # one hot encoding for categorical data
    df_train_new = feature_generator.one_hot_encode_categorical_features(df_train)
    df_test_new = feature_generator.one_hot_encode_categorical_features(df_test)


    # REQUIREMENT 3 - Analyze, visualize and select features

    # extract top 20 relevant features using correlation plot
    df_train_corr, _, numerical_features, _ = analyzer.extract_relevant_features_by_correlation(df_train_new, df_test_new)

    # data distribution plots for top 20 most relevant features (only numerical features will be displayed)
    analyzer.plot_distributions_for_numerical_features(df_train_corr, numerical_features)

    # 5 lasso regression points for 5 different values of alpha
    df_train_lasso = analyzer.lasso_regression_filtered_columns(df_train_new)

    # filter final features
    df_train_final = analyzer.select_features_of_interest(df_train_corr, df_train_lasso)

    
    # REQUIREMENT 4 - Train a regression model able to predict the house prices

    # implementing a basic regression model and computing its accuracy
    model_handler.run_regression(df_train_final)

    # running the regression model using normalization
    model_handler.run_regression(df_train_final, normalize=True)

    # apply k-fold cross-validation
    model_handler.run_regression_with_cv(df_train_final)

    # run regression and plot train and test learning curves
    model_handler.run_regression_with_learning_curve(df_train_final)

    plt.show()
    return

if __name__ == '__main__':
    main()
