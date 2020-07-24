# -*- coding: utf-8 -*-
"""
-Suman

"""

# Installing required library
!pip install -q optuna

Importing required metadata
import pandas as pd 
from sklearn import ensemble
from sklearn import linear_model
from tqdm import tqdm
import optuna
import lightgbm as lgbm
from optuna.samplers import TPESampler
from sklearn import model_selection
from sklearn import metrics
import numpy as np

# Google colab specific
from google.colab import drive
drive.mount('/content/drive')

# Google Colab Specific
from google.colab import files
src = list(files.upload().values())[0]
open('cross_val.py','wb').write(src)
from cross_val import cross_val

# Objective function Hyperparameter Tuning
def objective(trial):    
    param = {
        'objective': 'regression',
        'metric': 'mean_absolute_error',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'learning_rate': 0.01,
        'n_estimators': trial.suggest_int('n_estimators', 700, 3000),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    lgbm_regr = lgbm.LGBMRegressor(**param)
    gbm_2 = lgbm_regr.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return metrics.mean_absolute_error(np.expm1(y_valid), np.expm1(gbm_2.predict(X_valid)))

if __name__=='__main__':
    # Read Data Here
    df = pd.read_csv('/content/drive/My Drive/CrossValidation_And_HyperParam/train.csv')
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Split the data into dependent and independent variable
    y = df.price_range.values
    X = df.drop('price_range',axis='columns').values

    ############### Declare the estimatiors

    X_train,X_valid, y_train,y_valid = model_selection.train_test_split(X,y,test_size=0.2)
    # for reproducibility
    sampler = TPESampler(seed=10) 
    # Optimize the model hyperparameter
    study = optuna.create_study(direction='minimize', sampler=sampler)
    # Change the number of trial as per need 
    study.optimize(objective, n_trials=10,n_jobs=1,show_progress_bar=True)



    best_param = study.best_params
    #Calling Cross Validation Funcrion
    reg = lgbm.LGBMRegressor(**best_param)
    cross_val(reg,5,X,y,'regression')

