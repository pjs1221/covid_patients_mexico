# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:38:41 2020

@author: shinp
"""
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(),3600)
        tmin, tsec = divmod(temp_sec,60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour,tmin,round(tsec,2)))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("covid_data_cleaned.csv")

# choose relevant columns
df.columns

df_model = df[['deceased','sex','pneumonia', 'age', 'diabetes', 'copd',
       'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular',
       'obesity', 'renal_chronic', 'tobacco','days_to_entry']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('deceased', axis=1)
y = df_dum.deceased.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression(0.9315672256591835)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lgr = LogisticRegression()
lgr.fit(X_train,y_train)

lgr_scores = cross_val_score(lgr,X_train, y_train, cv =10)
lgr_avg = np.mean(lgr_scores)

print('Cross Validation Accuracy Scores:',lgr_scores)
print('Cross Validation Accuracy Mean:',lgr_avg)



#Decision Tree (0.9215302506106834)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

dtree_scores = cross_val_score(dtree,X_train, y_train, cv =10)
dtree_avg = np.mean(dtree_scores)

print('Cross Validation Accuracy Scores:',dtree_scores)
print('Cross Validation Accuracy Mean:',dtree_avg)

# Random forest (0.9244418539014261)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)

rf_scores = cross_val_score(rf,X_train, y_train, cv =10)
rf_avg = np.mean(rf_scores)

print('Cross Validation Accuracy Scores:',rf_scores)
print('Cross Validation Accuracy Mean:',rf_avg)

# XGboost (0.9330885146182804)
import xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train,y_train)

xgb_scores = cross_val_score(xgb,X_train,y_train, cv=10)
xgb_avg = np.mean(xgb_scores)

print('Cross Validation Accuracy Scores:',xgb_scores)
print('Cross Validation Accuracy Mean:',xgb_avg)

# Gradient boosted classifier (0.9329802122995764)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc = gbc.fit(X_train,y_train)

gbc_scores = cross_val_score(gbc,X_train, y_train, cv =10)
gbc_avg = np.mean(gbc_scores)

print('Cross Validation Accuracy Scores:',gbc_scores)
print('Cross Validation Accuracy Mean:',gbc_avg)


# tune models using GridsearchCV and RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid={
    "learning_rate":[0.05,0.10,0.15,0.20],
    "max_depth": [3,4,5,6],
    "min_child_weight": [1,3],
    "gamma": [0.2,0.3,0.4],
    "colsample_bytree": [0.5,0.7]
    
}

    #RandomizedSearch
random_search = RandomizedSearchCV(xgb,param_distributions=param_grid,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

start_time = timer(None)
random_search.fit(X_train,y_train)np
timer(start_time)

random_search.best_estimator_
random_search.best_params_

    
    #Gridsearch
grid_search = GridSearchCV(xgb,param_grid=param_grid, cv = 5, n_jobs = -1, verbose = 10)

start_time = timer(None)
grid_search.fit(X_train,y_train)
timer(start_time)

grid_search.best_params_
best_grid = grid_search.best_estimator_




# test ensembles
from sklearn import metrics
tpred_lgr = lgr.predict(X_test)
tpred_dtree = dtree.predict(X_test)
tpred_rf = rf.predict(X_test)
tpred_gbc = gbc.predict(X_test)
tpred_xgb = grid_search.best_estimator_.predict(X_test)

lgr_accuracy = metrics.accuracy_score(y_test,tpred_lgr)
dtree_accuracy = metrics.accuracy_score(y_test,tpred_dtree)
rf_accuracy = metrics.accuracy_score(y_test,tpred_rf)
gbc_accuracy = metrics.accuracy_score(y_test,tpred_gbc)
xgb_accuracy = metrics.accuracy_score(y_test,tpred_xgb)



#Utilize Differing Thresholds for the Model
from sklearn.metrics import confusion_matrix
print(metrics.confusion_matrix(y_test,y_test))
print(metrics.confusion_matrix(y_test,tpred_lgr))
print(metrics.confusion_matrix(y_test,tpred_dtree))
print(metrics.confusion_matrix(y_test,tpred_rf))
print(metrics.confusion_matrix(y_test,tpred_gbc))
print(metrics.confusion_matrix(y_test,tpred_xgb))


tpred_lgr_prob = lgr.predict_proba(X_test)[:,1]
tpred_dtree_prob = dtree.predict_proba(X_test)[:,1]
tpred_rf_prob = rf.predict_proba(X_test)[:,1]
tpred_gbc_prob = gbc.predict_proba(X_test)[:,1]
tpred_xgb_prob = grid_search.best_estimator_.predict_proba(X_test)[:,1]

model_probs = [tpred_lgr_prob, tpred_dtree_prob, tpred_rf_prob, tpred_gbc_prob, tpred_xgb_prob]

#Test for the maximum ratio between sensitivity and specificity for a range of thresholds for all models
max_j = [0] * len(model_probs)
max_ratio = [0] * len(model_probs)
x_scatter = np.empty([len(model_probs),500])
y_sens_scatter = np.empty([len(model_probs),500])
y_spec_scatter = np.empty([len(model_probs),500])
for i in range(len(model_probs)):
    for j in range(0,500):
        threshold = j/1000
        x_scatter[i][j] = threshold
        tpred_xgb_prob_test = np.array([1 if t > threshold else 0 for t in model_probs[i]])
        cm = metrics.confusion_matrix(y_test,tpred_xgb_prob_test)
        TP = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]
        TN = cm[0,0]
        y_sens_scatter[i][j] = TP / float(TP + FN)
        y_spec_scatter[i][j] = TN / float(TN + FP)
        
        if(y_sens_scatter[i][j] * y_spec_scatter[i][j] > max_ratio[i]):
            max_ratio[i] = y_sens_scatter[i][j] * y_spec_scatter[i][j]
            max_j[i] = j

#Save the model using pickle 
import pickle
pickl = {'model': gbc}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    
test_x = X_test[X_train.columns]

model.predict(test_x)

list(test_x.iloc[1,:])
            
            
