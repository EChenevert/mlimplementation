import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


# Feature selection types
def correlation_coefficent_select(corr_df, outcome):

    '''Based on the pearson's correlation coefficents of certain variables with the outcome.
    outcome = string'''

    corr_filt = corr_df.loc[:, outcome][corr_df.iloc[:, 11] > np.abs(0.3)]
    return corr_filt




def feature_importance_select(predictor_df, outcome):
    ''' Is a recursive feature elimination method. Based on a random forest model
    Help from: https://chrisalbon.com/code/machine_learning/trees_and_forests/feature_selection_using_random_forest/
    '''
    # Hold the predictor names
    feat_labels = list(predictor_df.columns.values)
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

    # return feature_importances

    # Create a selector object that will use the random forest classifier to identify
    # features that have an importance of more than 0.15
    sfm = SelectFromModel(regressor, threshold=0.05)

    # Train the selector
    sfm.fit(X_train, y_train)

    important_names = []
    # Print the names of the most important features
    for feature_list_index in sfm.get_support(indices=True):
        important_names.append(feat_labels[feature_list_index])

    return feature_importances, important_names

def genetic_algorithm():
    '''Creates a genetic algorithm model'''

def pca(X, categorical_var):
    ''''''
    Xpca = X
    Xpca = Xpca.drop([categorical_var])

    p = PCA(n_components=np.shape(Xpca)[1])
    Xpca = Xpca.fillna(value=Xpca.mean())
    p.fit(Xpca)
    print('explained variance ratio: %s'
          % str(p.explained_variance_ratio_))
    plt.plot(np.r_[[0], np.cumsum(p.explained_variance_ratio_)])
    plt.xlim(0, 5)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    # part2

    y = X[categorical_var]
    cat_labs = X[categorical_var].unique()
    y = y.map({'High': 0, 'Nuetral': 1, 'Low': 2})

    X_r = PCA.transform(Xpca)
    plt.scatter(X_r[y == 0, 0], X_r[y == 0, 1], alpha=.8, color='blue',
                label='High')
    plt.scatter(X_r[y == 1, 0], X_r[y == 1, 1], alpha=.8, color='green',
                label='Nuetral')
    plt.scatter(X_r[y == 2, 0], X_r[y == 2, 1], alpha=.8, color='orange',
                label='Low')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.title('PCA of IRIS dataset')


def K_means(ls_nclusters, ls_seed):
    ''''''

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
    importances, imp_names = feature_importance_select(X, y)

