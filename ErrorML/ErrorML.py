import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoLarsCV
from sklearn import ensemble
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import make_pipeline

from tpot.builtins import StackingEstimator

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from .BasicTransformer import BasicTransformer


def load_data(filename):
    df = pd.read_csv(filename, na_values='-9999.0')
    df = df.drop(['DoD', 'Z_diff_foc'], axis=1, errors='ignore')
    df.DepthRC_JD = df.DepthRC_JD.fillna(0)
    return df

# ['Z_diff', 'VEG_TREES', 'Slope', 'MaxSl_Foc', 'MinSl_Foc', 'StdSl_Foc',
#    'DepthRC_JD', 'Type', 'Pt_Density', 'CQ_Mean', 'CQmean_Foc', 'Rough40',
#    'Rough40_Foc', 'Precsn_m', 'Shadow', 'Blur', 'Reflection']


FOCAL_VARS = ['MaxSl_Foc', 'MinSl_Foc', 'StdSl_Foc',
              'CQ_Mean_Foc', 'Rough40_Foc']
NON_FOCAL_VARS = ['Slope', 'Rough40', 'CQ_Mean']
OTHER_VARS = ['Pt_Density', 'VEG_TREES', 'DepthRC_JD',
              'Shadow', 'Blur', 'Reflection', 'Type', 'Precsn_m']


def get_processed_data(df, categorised=False, focal=False,
                       scale=True, just_cols=False, exclude=None,
                       absolute=False, use_cols=None,
                       classes=[-2, -1, -0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5, 1, 6.5],
                       subset=None):
    if subset is not None:
        df = df[df['Type'] == subset]
    col = df.Z_diff

    if absolute:
        col = col.abs()

    if categorised:
        Z_diff_cat = pd.cut(col, classes)
        y = Z_diff_cat.cat.codes
    else:
        y = df.pop('Z_diff').values
    
    #### Get X matrix (explanatory variables)
    if focal is None:
        selected_columns = FOCAL_VARS + NON_FOCAL_VARS + OTHER_VARS
    elif focal is True:
        selected_columns = FOCAL_VARS + OTHER_VARS
    elif focal is False:
        selected_columns = NON_FOCAL_VARS + OTHER_VARS
    
    subdf = df[selected_columns]
    
    if exclude is not None:
        subdf = subdf.drop(exclude, axis=1)

    if use_cols is not None:
        subdf = subdf[use_cols]
    
    X = BasicTransformer(cat_threshold=3, return_df=True, scale_nums=scale).fit_transform(subdf)

    if just_cols:
        return X.columns

    return X, y


def get_train_and_test(df, categorised=False, oversample=True, focal=False,
                       scale=True, just_cols=False, exclude=None,
                       absolute=False,
                       classes=[-2, -1, -0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5, 1, 6.5],
                       subset=None):    
    
    X, y = get_processed_data(df, categorised, focal, scale, just_cols, exclude, absolute, classes, subset=subset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #### Do oversampling if necessary
    if categorised and oversample:
        ros = SMOTE()
        X_train, y_train = ros.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

def create_pipeline(kind, pca_n_elements=None):
    if kind == 'rf_classifier':
        return ensemble.RandomForestClassifier(n_estimators=100)
    elif kind == 'gaussian_nb':
        return GaussianNB()
    elif kind == 'gnb_pca':
        pipeline = make_pipeline(PCA(iterated_power=6, svd_solver='randomized'), GaussianNB())
        return pipeline
    elif kind == 'gnb_pca_default':
        if pca_n_elements is None:
            pipeline = make_pipeline(PCA(), GaussianNB())
        else:
            pipeline = make_pipeline(PCA(n_components=pca_n_elements), GaussianNB())
        return pipeline
    elif kind == 'bnb_pca':
        pipeline = make_pipeline(PCA(iterated_power=6, svd_solver='randomized'), BernoulliNB())
        return pipeline
    elif kind == 'rf_gnb':
        pipeline = make_pipeline(
                    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.5, min_samples_leaf=8, min_samples_split=18, n_estimators=100)),
                    GaussianNB()
                    )
        return pipeline
    elif kind == 'rf_regression':
        return RandomForestRegressor(n_estimators=100)
    elif kind == 'linear_regression':
        return LinearRegression()
    
def run_cross_validation(pipeline, X_train, y_train):
    if pipeline._estimator_type == 'regressor':
        scoring_functions = ('r2', 'neg_mean_absolute_error')
    elif pipeline._estimator_type == 'classifier':
        scoring_functions = ('accuracy')
    
    kf = KFold(n_splits=5, shuffle=True)
    results = cross_validate(pipeline, X_train, y_train, cv=kf,
                             scoring=scoring_functions)
    
    if pipeline._estimator_type == 'regressor':
        avg_r2 = results['test_r2'].mean()
        avg_mae = results['test_neg_mean_absolute_error'].mean() * -1
        print('CV R2: %.3f' % avg_r2)
        print('CV MAE: %.5f' % avg_mae)
        return {'r2': avg_r2,
                'mae': avg_mae}
    elif pipeline._estimator_type == 'classifier':
        avg_acc = results['test_score'].mean()
        print('CV Accuracy: %.3f' % avg_acc)
        return {'accuracy': avg_acc}

def get_feature_importances_rf_classifier(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    imps = pipeline.feature_importances_ / pipeline.feature_importances_.max()
    
    rf_importances = pd.Series(imps, index=X_train.columns).sort_values(ascending=False)
    return rf_importances.head(10)    

def get_feature_importances(pipeline, X_train, y_train):
    if type(pipeline) == ensemble.RandomForestClassifier:
        return get_feature_importances_rf_classifier(pipeline, X_train, y_train)
    elif type(pipeline) == ensemble.RandomForestRegressor:
        return get_feature_importances_rf_regression(pipeline, X_train, y_train)
    elif type(pipeline) == LinearRegression:
        return get_feature_importances_lr(pipeline, X_train, y_train)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
