import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# Feature selection types
def correlation_coefficent_select(predictor_df, outcome):
    '''Based on the pearson's correlation coefficents of certain variables with the outcome.'''


def feature_importance_select(predictor_df, outcome):
    ''' Is a recursive feature elimination method. Based on a random forest model
    Help from: https://chrisalbon.com/code/machine_learning/trees_and_forests/feature_selection_using_random_forest/
    '''
    # Hold the predictor names
    feat_labels = list(predictor_df.columns.values)
    print(type(feat_labels[0]))
    print(predictor_df[feat_labels[0]])
    # Replace all nans with the mean of the column

    # Scale the predictor values to be within 1 and 0
    scaler = MinMaxScaler()
    X = scaler.fit_transform(predictor_df)

    # Split the data (80_train/20_test)
    X_train, X_test, y_train, y_test = train_test_split(X, outcome, test_size=0.2, random_state=0)

    # Create a random forest classifier
    regressor = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)

    # Train the classifier
    regressor.fit(X_train, y_train)

    # Print the name and gini importance of each feature
    feature_importances = []
    for feature in zip(feat_labels, regressor.feature_importances_):
        feature_importances.append(feature)

    return feature_importances

    # # Create a selector object that will use the random forest classifier to identify
    # # features that have an importance of more than 0.15
    # sfm = SelectFromModel(regressor, threshold=0.15)
    #
    # # Train the selector
    # sfm.fit(X_train, y_train)
    #
    # # Print the names of the most important features
    # for feature_list_index in sfm.get_support(indices=True):
    #     print(feat_labels[feature_list_index])
    #

if __name__ == '__main__':
    # Just using th eti and jank csv (bysite)
    ej_df = pd.read_csv('/Users/etiennechenevert/Documents/CRMSresearch/projGit/EtiAndJank.csv')
    ej_df = ej_df.dropna(subset=['Elevation above/below NAVD 88\n(m)'])
    y = ej_df['Elevation above/below NAVD 88\n(m)']
    X = ej_df.drop(['Hydrologic Basin Type',
                    'CRMS Site',
                    'Geographic Region',
                    'Unnamed: 0', 'Simple site',
                    'Basins',
                    'Community',
                    'Binary Moisture Content',
                    'Binary Organic Fraction',
                    'Particle Size Standard Deviation (phi)',
                    'Year (yyyy)_y', 'Month (mm)_y',
                    'Year (yyyy)_x.1', 'Month (mm)_x.1',
                    'Rod depth below land surface \n(m)',
                    'Pleistocene depth below land surface \n(m)',
                    'Rod depth to Pleistocene \n(m)',
                    'Elevation above/below NAVD 88\n(m)'], axis=1)
    X = X.fillna(X.mean())
    imps = feature_importance_select(X, y)




