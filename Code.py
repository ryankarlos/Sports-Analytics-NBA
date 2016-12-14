# -*- coding: utf-8 -*--
"""
Created on Sat Nov 05 19:02:43 2016

BASKETBALL SHOT PREDICTION KAGGLE EDUCATIONAL COMPETITION - KOBE BRYANT 

@author: ryank
"""

#import sys

#sys.modules[__name__].__dict__.clear()  # removes all variables from the current workspace before running the rest of the script

import numpy as np
import pandas as pd
from sklearn import linear_model,svm,neighbors,decomposition, tree, ensemble
from sklearn.decomposition.pca import PCA
from sklearn import preprocessing 
from sklearn.preprocessing import scale 
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score 
import scipy as sp 
import collections
from time import time
import matplotlib.pyplot as plt


data = pd.read_csv("data.csv")

#submission = pd.read_csv("sample_submission.csv")
#submission = submission.drop('shot_made_flag', axis =1) # dropping the shot_made_flag which were set as 0.5 for all the shot IDs 

print(data.head())

Missing_values = data.shape[0] - data.count() # these 5000 missing values correspond to the shotmade flag which need to be predicted

#print(Missing_values) # 5000 missing values correspond to the shotmade flag attributes which are stored in NBA_test and need to be predicted so we will drop them from the training set 


#print(NBA_train['shot_made_flag'].isnull().sum()) # check to see that new training data set contains no missing values now


data.drop(data.columns[[2,3,10,-3,-4,-5,-6]],axis=1, inplace=True) #Droping columns we are not interested e.g. categorical descriptive

lon = data['lon']
lat = data['lat']
loc_x = data['loc_x']
loc_y = data['loc_y']

## Visualising what locx locy lat and lon represent
plt.figure(1)
plt.scatter(lat,lon, c =(1,1,0,),marker ='o', alpha=0.04 )  # scatter plot of court 
plt.title('Lat vs lon scatter plot showing positions where shot was released on the court')
plt.xlabel('latitude')
plt.ylabel('longitude')

plt.figure(2)
plt.scatter(loc_y,loc_x,  c = (0,1,1), marker ='o',alpha =0.04)
plt.title('Loc_x  vs loc_y scatter plot showing positions where shot was released on the court')
plt.xlabel('loc_x')
plt.ylabel('loc_y')

# This represents a semi circle- the boundary of which represents the 3 point line. Correct shots made wthin this region are worth 2 points, and outside are worth 3 points.so locx loc y lat and lon correspond to the coordinates on the court from where the ball was shot by Kobe. Looks like locx,locy give the same information as lat and lon 

label = data[['shot_made_flag']]
label = label.ix[:,0].dropna()
label_names = ['Missed','Scored']
label_dict = collections.Counter(label)
label_count = label_dict.values()     # do label.key() if you need the classes 
label_scored_percent = 100*float(label_count[1])/sum(label_count)

print ''
print 'Percent of shots scored in labelled set is %0.1f %% and percent of shots missed is %0.1f %%' % (label_scored_percent,100-label_scored_percent)
print ''
coordinates = data[['lat','lon','shot_made_flag']] #lets append the locx and locy data along with the shots made flag to a new array called coordinates 
plt.figure(3)
#color =[]
#for row in range(len(coordinates["shot_made_flag"])):
   # if row == 1:
   #     color.append((1,0,0)) 
   # else:
     #   color.append((0,1,0)) 

coordinates_scored = coordinates[coordinates.shot_made_flag == 1]

plt.scatter(coordinates_scored['lat'],coordinates_scored['lon'], c = (1,0,0), marker ='o', alpha=0.04,)  # scatter plot of court 
plt.title('Lat vs lon scatter plot showing positions where Kobe Bryant scored')
plt.xlabel('latitude')
plt.ylabel('longitude')

data['season'] = data['season'].map(lambda x: x[0:4]).astype(np.int64) # Converting the year range in the season column to a single year by stripping character following hyphen 

submission = data[pd.isnull(data).any(axis=1)]
submission = submission.drop('shot_made_flag', axis =1) 
data = data.dropna(subset =['shot_made_flag'])

#NBA_train['season'] = (np.int64((NBA_train['season']) + 1)


data = data.drop(['shot_made_flag','combined_shot_type','lat','loc_x', 'loc_y', 'lon','shot_id'], axis =1)
categories = ['shot_type','opponent','shot_zone_area','shot_zone_basic','shot_zone_range','season','period', 'action_type']
one_hot = pd.get_dummies(data[categories])
data_noncategory = data.drop(categories, axis=1)
data_one_hot = data_noncategory.join(one_hot)
scaler = preprocessing.StandardScaler().fit(data_one_hot)
data_scaled = scaler.transform(data_one_hot)


submission = submission.drop(['lat','loc_x', 'loc_y', 'lon','shot_id', 'combined_shot_type'], axis =1)# lets drop action type here and we will merge it later 
categories = ['shot_type','opponent','shot_zone_area','shot_zone_basic','shot_zone_range','season','period', 'action_type']
one_hot = pd.get_dummies(submission[categories])
submission_noncategory = submission.drop(categories, axis=1)
submission_one_hot = submission_noncategory.join(one_hot)
scaler_submission = preprocessing.StandardScaler().fit(submission_one_hot)
submission_scaled = scaler_submission.transform(submission_one_hot)

data_PCA = data.values  
train_PCA = data_PCA[:,[1,4,5]] # using only the first five columns for PCA i.e. non  
pca = PCA(n_components=3) # the components are mins remaining, period, season,seconds remaining and shot distance 

 # scaling the columns so they all have the same range then normalisaing by z-score transformation befor PCA
scaler_PCA = preprocessing.StandardScaler().fit(train_PCA)
PCA_scaled = scaler_PCA.transform(train_PCA)
pca.fit(PCA_scaled) 
projectedAxes = pca.transform(PCA_scaled)

varianceratio = pca.explained_variance_ratio_  #scores 
print(' ')
print(pca.explained_variance_ratio_) # prints array with componets with highest variance ratio in descending order
print(' ')


sum(varianceratio[0:-1]); #the first 4 components account for only 83% of the total variance so dimensionality reduction may results in loss of information and affect mdoel performance
train_PCA_cumsum_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(' ')
print(train_PCA_cumsum_var) # prints array with cummulative variance for the 5 components
print(' ')

with plt.style.context('seaborn-whitegrid'):
    plt.figure(4)

plt.bar(range(len(varianceratio)), (100*varianceratio), alpha=0.5, align='center',
label='individual explained variance')
plt.step(range(len(train_PCA_cumsum_var)), (train_PCA_cumsum_var), where='mid',
label='cumulative explained variance')
plt.title('Bar chart of explained variance ratio for individual components and Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
#plt.tight_layout()


# we see that each attribute contributes roughly 20% of the variance in the data are are almost equally important. So we will use all 5 features for the subsequent modelling steps 

plt.figure(5)
plt.scatter(projectedAxes[:,0], projectedAxes[:,1],marker = 'o', c =(1,0,0),s =50, alpha =0.4)
plt.title("Scatter plot PC1 vs PC2")
plt.xlabel("PC 1 ")
plt.ylabel("PC 2")

#### Modelling 


X_train, X_test, y_train, y_test = train_test_split(data_scaled, label, test_size=0.3, random_state=42)


# Utility function to report best scores

forest = ensemble.ExtraTreesClassifier()

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# use a full grid over all parameters
param_grid = {"n_estimators":[90],
              "max_depth": [1,3,10],
              "max_features": [113],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(forest, param_grid=param_grid, cv= 10)  #specify 10 fold cross validation otherwise default of 3 folds is used 
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
print ''
print 'A summary of the grid search:'
print grid_search.cv_results_
print ''
print 'The best estimator from the grid search was %s and corresponding score is %.2f' %(grid_search.best_estimator_ , grid_search.best_score_)
print grid_search.best_params_



forest = ensemble.ExtraTreesClassifier(bootstrap= True, min_samples_leaf=  10, n_estimators=  90, min_samples_split= 10, criterion = 'gini', max_features= 113, max_depth= 10, random_state =42)

#kfold = StratifiedKFold(n_splits = 10, shuffle = False)
#forest = ensemble.ExtraTreesClassifier(bootstrap= True, min_samples_leaf=  10, n_estimators=  100, min_samples_split= 3, criterion = 'entropy', max_features= 113, max_depth= 10, random_state =42)
forest.fit(X_train,y_train)
#results= cross_val_score(forest, X_train, y_train, cv=kfold)
#print(results.mean())

y_pred = forest.predict(X_test)
result = forest.score(X_test, y_test)
print ''
print "The test accuracy is %.2f" % result
print ''
print "AUC-ROC:",roc_auc_score(y_test,y_pred)
print ''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix without mormalisation.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test,y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes = label_names, title='Confusion matrix, without normalization')

f1score = f1_score(y_test, y_pred, average = 'weighted')

print "the F1 score is %.2f:" % f1score
print ''
# Computing log loss (funciton taken from kaggle)

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

print(logloss(y_test,y_pred))
print ''

print "Feature importance: %.2f:" % forest.feature_importances_

importances = forest.feature_importances_
std = np.std([trees.feature_importances_ for trees in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_test.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_test.shape[1]), indices)
plt.xlim([-1, X_test.shape[1]])
plt.show()


#preds = forest.predict_proba(X_test)
#preds = preds[:,0]                         # calculating predictions on submssion set using trained model. Only select the first column. 