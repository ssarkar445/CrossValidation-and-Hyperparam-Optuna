################################################### Usage ########################################
'''
cross_val(param1,param2,param3,param4,param5)

param1 : model
param2 : Number of folds 
param3 : Independependent / Features
param4 : Dependent /Target
param5 : regression/classification

Note :
param2 should be >1
'''
################################################### Usage ########################################


import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
import numpy as np
from sklearn import svm
from tqdm import tqdm
from sklearn import pipeline

def cross_val(mod,folds,X,y,type='regression'):
    if type=='regression':
        fold = model_selection.KFold(n_splits=folds)
        outcomes = []
        error = []
        for fold,(t_idx,v_id) in tqdm(enumerate(fold.split(X=X,y=y))):
            Xtrain, Xtest = X[t_idx], X[v_id]
            ytrain, ytest = y[t_idx], y[v_id]
            mod.fit(Xtrain, ytrain)
            predictions = mod.predict(Xtest)
            r2_score = metrics.r2_score(ytest,predictions)
            mean_error = np.sqrt(metrics.mean_squared_error(ytest,predictions))
            outcomes.append(r2_score)
            error.append(mean_error)
        mean_outcome = np.mean(outcomes)
        mean_error = np.mean(error)
        print("R2 Score: {0}".format(mean_outcome)) 
        print('Mean Error :{}'.format(mean_error))
    else:
        fold = model_selection.StratifiedKFold(n_splits=folds)
        outcomes = []
        error = []
        for fold,(t_idx,v_id) in tqdm(enumerate(fold.split(X=X,y=y))):
            Xtrain, Xtest = X[t_idx], X[v_id]
            ytrain, ytest = y[t_idx], y[v_id]
            mod.fit(Xtrain, ytrain)
            predictions = mod.predict(Xtest)
            accuracy = metrics.accuracy_score(ytest,predictions)
            mean_error = metrics.log_loss(ytest,predictions)
            outcomes.append(accuracy)
            error.append(mean_error)
        mean_outcome = np.mean(outcomes)
        mean_error = np.mean(error)
        print("Accuracy Score: {0}".format(mean_outcome)) 
        print('Log Loss :{}'.format(mean_error))

