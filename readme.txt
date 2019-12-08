Regression based learning system that is able to predict house prices


REQUIREMENT 1 -  Investigate and preprocess the dataset

Before going into running a regression model over the dataset to be able to predict house prices, first we will take a look at the data available.

Seeing as duplicate data does not bring any new information for our model, we first eliminate the duplicates by id after we load the data from csv.
This is done in the preprocess.py utility tool.

Next we would like to handle the outliers, so that they do not affect our learning process. We look at the numerical features that could intuitively 
present outliers and also be of interest, and we select 'LotArea', 'YearBuilt' and 'GrLivArea'. The best decision would be that if they are too 'out
of our range of values', we eliminate the row. So we calculate the z-score of these columns, and if the score is above a certain threshold (which we
took as equal to 3) we eliminate the row.

The next step would be to transform numerical features that are actually categories to said categories, and the ordered categories to ordered numbers.
Last, but not least, we look into the missing data. We first analyze how much of the data is missing on each column, and remove the columns that have
more than 40% of the values missing. The rest are replaced by the mean of that column.


REQUIREMENT 2 - Prepare by generating new features

Further preprocessing of the dataset can be done by generating new features based on the ones we have. We do this in the feature_generator.py tool. Here
we transform some features represented through complicated ordered numbers into less ordered numbers. For example, looking at the columns 'OverallQual'
and 'OverallCond', we transform them from the space [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] to [1, 2, 3] which represent 'low', 'average' and 'high', respectively.

Another process we use to generate new features is by one hot encoding categorical data. Each column that represents categories also is transformed into
several columns, one for each possible value, that represents the one hot encoding of the said category.


REQUIREMENT 3 - Analyze, visualize and select features

This step is done using the analyzer.py script. The first thing to analyse is the correlation matrix heatmap on the full data frame, including the newly 
added columns that represent one hot encodings of categorical features. Here we can determine 21 columns that are strongly correlated (correlation > 0.4) 
with the target variable 'SalePrice': 

interest_numerical_features = ['ExterQual', 'GrLivArea', 'OverallQual', 'KitchenQual', 'BsmtQual', 'GarageCars',
                            'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'FullBath', 'YearRemodAdd', 'TotRmsAbvGrd',
                            'GarageFinish', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'HeatingQC', 'SalePrice']
interest_categorical_features = ['Neighborhood', 'Foundation', 'SaleType']

Another column included, but not taken into account is the 'GarageArea' feature, that seems to be very similar to 'GarageCars'. For later analysis, the
categorical features are not looked into 'as is', but as one hot encodings.

Another way to extract information about the features is by running a lasso regression. For high values of alpha, more features get a lasso coefficient equal
to 0. The alphas that we use are [10000, 50000, 10e5, 5*10e5, 10e6], high enough values so that the regression converges. For alpha = 10000, we get 22 features
with non-zero coefficient, while for alpha = 10e6 we get only 3. We only look at the features given by alpha = 10000, because the ones for higher values are
included in the ones for this value of alpha. Here we can see that we can also take into account the features '2ndFlrSF' and 'PoolArea', which, intuitively, 
should influence the 'SalePrice'.

The final features would be: 'ExterQual', 'GrLivArea', 'OverallQual', 'KitchenQual', 'BsmtQual', 'GarageCars', '2ndFlrSF',
'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'FullBath', 'YearRemodAdd', 'TotRmsAbvGrd', 'PoolArea',
'GarageFinish', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'HeatingQC', 'Neighborhood', 'Foundation', 'SaleType'


REQUIREMENT 4 - Train a regression model able to predict the house prices

After running a simple regression model over the training set comprised of the above columns, with a test split of 0.2, we get an R^2 score of around 0.86. This
shows that the features we chose are actually strongly connected to the target variable.

When running a k-fold cross validation with k = 5, we also consistently obtain an average R^2 score of 0.86.

By plotting the learning curve of the regression model, our conclusions are sustained, seeing that the validation error converges downward to around the 0.86 accuracy,
while the training error converges upward to the same value.


OTHER

The score obtained for predicting the 'SalePrice' of the test.csv data on Kaggle in the 'House Prices: Advanced Regression Techniques' challenge was:

RMSE = 0.16677

(Used the predicted target.csv values)

