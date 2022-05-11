#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

submission_sample = pd.read_csv('data/sampleSubmission.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#first/
train.head()
test.head()
train.info()
train['Elevation'].plot(kind='hist')
plt.hist(train['Elevation'])
plt.show()

train.isna().sum()

sns.pairplot(train, hue='Cover_Type', vars=train.columns[1:11])

pd.crosstab(train.Soil_Type31, train.Cover_Type)
# Create dummies variable for Mutivariate EDA
df_viz = train.iloc[:, 0:15]
df_viz = df_viz.drop(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
                      'Wilderness_Area4'], axis = 1)
df_viz.head()

corr = df_viz.corr()

# plot the heatmap
plt.figure(figsize=(14,12))
colormap = plt.cm.RdBu
sns.heatmap(corr,linewidths=0.1, 
            square=False, cmap=colormap, linecolor='white', annot=True)
plt.title('Pearson Correlation of Numeric Features', size=14)

X=train.drop(labels=['Id','Cover_Type'],axis=1)

y=train['Cover_Type']



X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=40)

print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=70)
rfc.fit(X_train,y_train)
rfc.score(X_val,y_val)

pred=rfc.predict(test.drop(labels=['Id'], axis=1))


submission=pd.DataFrame(data=pred,columns=['Cover_Type'])
submission.head()

submission['Id']=test['Id']
submission.head()

submission.set_index('Id',inplace=True)

submission


submission.to_csv('submission.csv')


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# Create an instance of Pipeline
#
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))


# In[18]:


# Create an instance of StratifiedKFold which can be used to get indices of different training and test folds
#
strtfdKFold = StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X_train, y_train)
scores = []
#
#
#
for k, (train, test) in enumerate(kfold):
    pipeline.fit(X_train.iloc[train, :], y_train.iloc[train])
    score = pipeline.score(X_train.iloc[test, :], y_train.iloc[test])
    scores.append(score)
    print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))
 
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# Create an instance of Pipeline
#
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
#
# Pass instance of pipeline and training and test data set
# cv=10 represents the StratifiedKFold with 10 folds
#
scores = cross_val_score(pipeline, X=X_train, y=y_train, cv=10, n_jobs=1)
 
print('Cross Validation accuracy scores: %s' % scores)
 
print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))


from sklearn import preprocessing
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_val)
clf.score(X_test_transformed, y_val)


from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, X, y, cv=10)


from sklearn.model_selection import RepeatedKFold

random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
     print("%s %s" % (train, test))

#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#from mlxtend.data import mnist_data
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
np.random.seed(1)
random.seed(1)


X=train.drop(labels=['Id','Cover_Type'],axis=1)
y=train['Cover_Type']


X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=1,
                                                    stratify=y)



print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)


# Initializing Classifiers
clf1 = LogisticRegression(multi_class='multinomial',
                          solver='newton-cg',
                          random_state=1)
clf2 = KNeighborsClassifier(algorithm='ball_tree',
                            leaf_size=50)
clf3 = DecisionTreeClassifier(random_state=1)
clf4 = SVC(random_state=1)





# Building the pipelines
pipe1 = Pipeline([('std', StandardScaler()),
                  ('clf1', clf1)])

pipe2 = Pipeline([('std', StandardScaler()),
                  ('clf2', clf2)])

pipe4 = Pipeline([('std', StandardScaler()),
                  ('clf4', clf4)])



# Setting up the parameter grids
param_grid1 = [{'clf1__penalty': ['l2'],
                'clf1__C': np.power(10., np.arange(-4, 4))}]

param_grid2 = [{'clf2__n_neighbors': list(range(1, 10)),
                'clf2__p': [1, 2]}]

param_grid3 = [{'max_depth': list(range(1, 10)) + [None],
                'criterion': ['gini', 'entropy']}]

param_grid4 = [{'clf4__kernel': ['rbf'],
                'clf4__C': np.power(10., np.arange(-4, 4)),
                'clf4__gamma': np.power(10., np.arange(-5, 0))},
               {'clf4__kernel': ['linear'],
                'clf4__C': np.power(10., np.arange(-4, 4))}]



gcv = GridSearchCV(estimator=est,
                      param_grid=pgrid,
                      scoring='accuracy',
                      n_jobs=1,
                      cv=2,
                      verbose=0,
                      refit=True)
gridcvs['Softmax'] = gcv   
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
nested_score = cross_val_score(gs_est, 
                                  X=X_train, 
                                  y=y_train, 
                                  cv=outer_cv,
                                  n_jobs=1)
nested_score
print('%s | outer ACC %.2f%% +/- %.2f' % 
         ('Softmax', nested_score.mean() * 100, nested_score.std() * 100))


# Setting up multiple GridSearchCV objects, 1 for each algorithm
gridcvs = {}

for pgrid, est, name in zip((param_grid1, param_grid2,
                             param_grid3),
                            (pipe1, pipe2, clf3),
                            ('Softmax', 'KNN', 'DTree')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='accuracy',
                       n_jobs=1,
                       cv=2,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv


outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for name, gs_est in sorted(gridcvs.items()):
    nested_score = cross_val_score(gs_est, 
                                   X=X_train, 
                                   y=y_train, 
                                   cv=outer_cv,
                                   n_jobs=1)
    print('%s | outer ACC %.2f%% +/- %.2f' % 
          (name, nested_score.mean() * 100, nested_score.std() * 100))


# Fitting a model to the whole training set
# using the "best" algorithm
best_algo = gridcvs['KNN']

best_algo.fit(X_train, y_train)
train_acc = accuracy_score(y_true=y_train, y_pred=best_algo.predict(X_train))
test_acc = accuracy_score(y_true=y_test, y_pred=best_algo.predict(X_test))

print('Accuracy %.2f%% (average over CV test folds)' %
      (100 * best_algo.best_score_))
print('Best Parameters: %s' % gridcvs['KNN'].best_params_)
print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))

pred=best_algo.predict(test.drop(labels=['Id'], axis=1))


submission=pd.DataFrame(data=pred,columns=['Cover_Type'])
submission.head()

submission.to_csv('submission.csv')




