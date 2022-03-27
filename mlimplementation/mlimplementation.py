import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from gplearn.genetic import SymbolicRegressor
from sympy import sympify

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import seaborn as sns



# Feature selection types
def correlation_coefficent_select(corr_df, outcome):

    '''Based on the pearson's correlation coefficents of certain variables with the outcome.
    outcome = string'''

    corr_filt = corr_df.loc[:, outcome][corr_df.iloc[:, 11] > np.abs(0.3)]
    return corr_filt

def plot_feat_imps(feats):
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale = 5)
    sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30,15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight = 'bold')
    plt.ylabel('Features', fontsize=25, weight = 'bold')
    plt.title('Feature Importance', fontsize=25, weight = 'bold')


def random_forest(X, y):
    '''This function will run a random forest machine learning algorithm on inputted
    data'''
    # X, y = self.set_up()
    scaler = MinMaxScaler()
    Xin = scaler.fit_transform(X)
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(Xin, y, test_size=0.2, random_state=0)
    # Create random forest regressor (fitting)
    regressor = RandomForestRegressor(n_estimators=2501, random_state=0, min_samples_leaf=10).fit(X_train, y_train)

    # Predict outcome from model
    y_pred = regressor.predict(X_test)

    feature_importance = pd.DataFrame(regressor.feature_importances_,
                                        columns=['importance']).sort_values('importance', ascending=False)

    r, _ = pearsonr(np.squeeze(y_test), np.squeeze(y_pred))
    # np.squeeze(y_test), np.squeeze(y_pred)
    model_performance = {
        'Mean Absolute Error (cm):': metrics.mean_absolute_error(y_test, y_pred),
        # 'Mean Squared Error:': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (cm):': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        'R squared:': metrics.r2_score(y_test, y_pred),
        'Pearson Correlation Coefficient': r
    }

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.show()

    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['RF'])


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
    sfm = SelectFromModel(regressor, threshold=0.03)

    # Train the selector
    sfm.fit(X_train, y_train)

    important_names = []
    # Print the names of the most important features
    for feature_list_index in sfm.get_support(indices=True):
        important_names.append(feat_labels[feature_list_index])

    # plot_feat_imps(feature)

    return feature_importances, important_names


def symbolic_regression(X, y):
    """ Creates a symbolic regression and returns the performance of the model AS WELL as the best fit equation"""
    # X, y = self.set_up()
    scaler = MinMaxScaler()
    Xin = scaler.fit_transform(X)

    X_Train, X_Test, y_train, y_test = train_test_split(Xin, y, test_size=0.2, random_state=0)

    # Create the SR, the inputs are from the tutoriol on gplearn documentation website
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_Train, y_train)
    y_pred = est_gp.predict(X_Test)

    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'sin': lambda x: np.sin(x),
        'cos': lambda x: np.cos(x),
        'inv': lambda x: 1 / x,
        'sqrt': lambda x: x ** 0.5,
        'pow3': lambda x: x ** 3
    }

    print(est_gp._program)
    equation = sympify((est_gp._program), locals=converter)
    r, _ = pearsonr(np.squeeze(y_test), np.squeeze(y_pred))
    model_performance = {
        'Mean Absolute Error (cm):': metrics.mean_absolute_error(y_test, y_pred),
        # No dividing done because I use a StandardScalar which should make units "equal"
        # 'Mean Squared Error:': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (cm):': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        # No dividing done because I use a StandardScalar which should make units "equal"
        'R squared:': metrics.r2_score(y_test, y_pred),
        'Pearson Correlation Coefficient': r
    }

    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['SR']), equation


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
    """"""


if __name__ == '__main__':
    # Just using th eti and jank csv (bysite)
    ej_df = pd.read_csv('/Users/etiennechenevert/Documents/CRMSresearch/projGit/seasonalavg.csv')
    ej_df = ej_df.dropna(subset=['Average Accretion (mm)'])
    y = ej_df['Average Accretion (mm)']
    X = ej_df.drop(['Simple site',
                    'Particle Size Standard Deviation (phi)',
                    'Year (yyyy)_y',
                    'Year (yyyy)_x.1',
                    'Measurement Depth (ft)',
                    'Observation Length \n(years)',
                    'Longitude', 'Season',
                    'Turbidity (FNU)', 'Chlorophyll a (ug/L)',
                    'Total Nitrogen (mg/L)', 'Total Kjeldahl Nitrogen (mg/L)',
                    'Nitrate as N (mg/L)', 'Nitrite as N (mg/L)',
                    'Nitrate+Nitrite as N (unfiltered; mg/L)',
                    'Nitrate+Nitrite as N (filtered; mg/L)',
                    'Ammonium as N (unfiltered; mg/L)', 'Ammonium as N (filtered; mg/L)',
                    'Total Phosphorus (mg/L)', 'Orthophosphate as P (unfiltered; mg/L)',
                    'Orthophosphate as P (filtered; mg/L)', 'Silica (unfiltered; mg/L)',
                    'Silica (filtered; mg/L)', 'Total Suspended Solids (mg/L)',
                    'Volatile Suspended Solids (mg/L)', 'Secchi (ft)',
                    'Fecal Coliform (MPN/100ml)', 'pH (pH units)', 'Velocity (ft/sec)',
                    'Radiometric Dating Method and Units',
                    'Isotope Concentration', 'Latitude',
                    'Accretion Measurement 1 (mm)',
                    'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
                    'Accretion Measurement 4 (mm)', 'Direction (Collar Number)',
                    'Direction (Compass Degrees)', 'Pin Number',
                    'Observed Pin Height (mm)', 'Verified Pin Height (mm)',
                    'Vertical Accretion Rate \n(mm/yr)', 'Vertical Accretion \nRMS Error ',
                    'Surface-Elevation Change Rate \n(mm/yr)', 'Year (yyyy)_y.1',
                    'Accretion Surplus/Deficit\n(mm/yr)', 'Average Flood Depth Jank (mm)',
                    'Elevation above/below NAVD 88\n(m)',
                    'Average Accretion (mm)'], axis=1)
    X = X.fillna(X.mean())
    # Test feature importances
    importances, imp_names = feature_importance_select(X, y)
    # plot ideal
# Longitude being highlighted as an important feature likely means we need to split the data between western coast and
# eastern LA coast
# 'Oberservation Length (ft) also being highlighted as an important feature may be highlighting a sampling bias in the data
    # Make dataset of selected important features
    Ximp = X[imp_names]
    # Run symbolic regression
    sr, eq = symbolic_regression(Ximp, y)
    # Run random forest model
    Ximp_rf = X[imp_names]
    rf = random_forest(Ximp_rf, y)
