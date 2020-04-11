---
layout: post
title: machine learning_ensemble
categories:
  - Programming
tags:
  - Python
  - ensemble
last_modified_at: 2020-04-11
use_math: true
---
source & copyright: lecture note in DataScienceLab in Yonsei university  

### Ensemble 

```python
%matplotlib inline

import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)
```


```python
iris = datasets.load_iris()
X, y = iris.data[:, 0:2], iris.target
```


```python
iris
```




    {'data': array([[5.1, 3.5, 1.4, 0.2],
            [4.9, 3. , 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5. , 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5. , 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.4, 3.7, 1.5, 0.2],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3. , 1.4, 0.1],
            [4.3, 3. , 1.1, 0.1],
            [5.8, 4. , 1.2, 0.2],
            [5.7, 4.4, 1.5, 0.4],
            [5.4, 3.9, 1.3, 0.4],
            [5.1, 3.5, 1.4, 0.3],
            [5.7, 3.8, 1.7, 0.3],
            [5.1, 3.8, 1.5, 0.3],
            [5.4, 3.4, 1.7, 0.2],
            [5.1, 3.7, 1.5, 0.4],
            [4.6, 3.6, 1. , 0.2],
            [5.1, 3.3, 1.7, 0.5],
            [4.8, 3.4, 1.9, 0.2],
            [5. , 3. , 1.6, 0.2],
            [5. , 3.4, 1.6, 0.4],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 3.4, 1.4, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.4, 3.4, 1.5, 0.4],
            [5.2, 4.1, 1.5, 0.1],
            [5.5, 4.2, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.2],
            [5. , 3.2, 1.2, 0.2],
            [5.5, 3.5, 1.3, 0.2],
            [4.9, 3.6, 1.4, 0.1],
            [4.4, 3. , 1.3, 0.2],
            [5.1, 3.4, 1.5, 0.2],
            [5. , 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [4.4, 3.2, 1.3, 0.2],
            [5. , 3.5, 1.6, 0.6],
            [5.1, 3.8, 1.9, 0.4],
            [4.8, 3. , 1.4, 0.3],
            [5.1, 3.8, 1.6, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [5.3, 3.7, 1.5, 0.2],
            [5. , 3.3, 1.4, 0.2],
            [7. , 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4. , 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1. ],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
            [5. , 2. , 3.5, 1. ],
            [5.9, 3. , 4.2, 1.5],
            [6. , 2.2, 4. , 1. ],
            [6.1, 2.9, 4.7, 1.4],
            [5.6, 2.9, 3.6, 1.3],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3. , 4.5, 1.5],
            [5.8, 2.7, 4.1, 1. ],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1],
            [5.9, 3.2, 4.8, 1.8],
            [6.1, 2.8, 4. , 1.3],
            [6.3, 2.5, 4.9, 1.5],
            [6.1, 2.8, 4.7, 1.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.6, 3. , 4.4, 1.4],
            [6.8, 2.8, 4.8, 1.4],
            [6.7, 3. , 5. , 1.7],
            [6. , 2.9, 4.5, 1.5],
            [5.7, 2.6, 3.5, 1. ],
            [5.5, 2.4, 3.8, 1.1],
            [5.5, 2.4, 3.7, 1. ],
            [5.8, 2.7, 3.9, 1.2],
            [6. , 2.7, 5.1, 1.6],
            [5.4, 3. , 4.5, 1.5],
            [6. , 3.4, 4.5, 1.6],
            [6.7, 3.1, 4.7, 1.5],
            [6.3, 2.3, 4.4, 1.3],
            [5.6, 3. , 4.1, 1.3],
            [5.5, 2.5, 4. , 1.3],
            [5.5, 2.6, 4.4, 1.2],
            [6.1, 3. , 4.6, 1.4],
            [5.8, 2.6, 4. , 1.2],
            [5. , 2.3, 3.3, 1. ],
            [5.6, 2.7, 4.2, 1.3],
            [5.7, 3. , 4.2, 1.2],
            [5.7, 2.9, 4.2, 1.3],
            [6.2, 2.9, 4.3, 1.3],
            [5.1, 2.5, 3. , 1.1],
            [5.7, 2.8, 4.1, 1.3],
            [6.3, 3.3, 6. , 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [7.1, 3. , 5.9, 2.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.5, 3. , 5.8, 2.2],
            [7.6, 3. , 6.6, 2.1],
            [4.9, 2.5, 4.5, 1.7],
            [7.3, 2.9, 6.3, 1.8],
            [6.7, 2.5, 5.8, 1.8],
            [7.2, 3.6, 6.1, 2.5],
            [6.5, 3.2, 5.1, 2. ],
            [6.4, 2.7, 5.3, 1.9],
            [6.8, 3. , 5.5, 2.1],
            [5.7, 2.5, 5. , 2. ],
            [5.8, 2.8, 5.1, 2.4],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 3. , 5.5, 1.8],
            [7.7, 3.8, 6.7, 2.2],
            [7.7, 2.6, 6.9, 2.3],
            [6. , 2.2, 5. , 1.5],
            [6.9, 3.2, 5.7, 2.3],
            [5.6, 2.8, 4.9, 2. ],
            [7.7, 2.8, 6.7, 2. ],
            [6.3, 2.7, 4.9, 1.8],
            [6.7, 3.3, 5.7, 2.1],
            [7.2, 3.2, 6. , 1.8],
            [6.2, 2.8, 4.8, 1.8],
            [6.1, 3. , 4.9, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [7.2, 3. , 5.8, 1.6],
            [7.4, 2.8, 6.1, 1.9],
            [7.9, 3.8, 6.4, 2. ],
            [6.4, 2.8, 5.6, 2.2],
            [6.3, 2.8, 5.1, 1.5],
            [6.1, 2.6, 5.6, 1.4],
            [7.7, 3. , 6.1, 2.3],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 3.1, 5.5, 1.8],
            [6. , 3. , 4.8, 1.8],
            [6.9, 3.1, 5.4, 2.1],
            [6.7, 3.1, 5.6, 2.4],
            [6.9, 3.1, 5.1, 2.3],
            [5.8, 2.7, 5.1, 1.9],
            [6.8, 3.2, 5.9, 2.3],
            [6.7, 3.3, 5.7, 2.5],
            [6.7, 3. , 5.2, 2.3],
            [6.3, 2.5, 5. , 1.9],
            [6.5, 3. , 5.2, 2. ],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3. , 5.1, 1.8]]),
     'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
     'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
     'DESCR': '.. _iris_dataset:\n\nIris plants dataset\n--------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n                \n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThe famous Iris database, first used by Sir R.A. Fisher. The dataset is taken\nfrom Fisher\'s paper. Note that it\'s the same as in R, but not as in the UCI\nMachine Learning Repository, which has two wrong data points.\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\n.. topic:: References\n\n   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...',
     'feature_names': ['sepal length (cm)',
      'sepal width (cm)',
      'petal length (cm)',
      'petal width (cm)'],
     'filename': 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv'}




```python
iris_plot = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns = iris['feature_names']+['target'])
iris_plot['target'] = iris_plot['target'].map({0:"setosa", 1:"versicolor",2:"virginia"})

sns.pairplot(iris_plot,x_vars=["sepal length (cm)"],y_vars=["sepal width (cm)"],hue='target', size = 5)
```




    <seaborn.axisgrid.PairGrid at 0x1ebe15a7630>




![png](https://drive.google.com/uc?export=view&id=133dhpGkUDh1sg46lpJ4ycG2-Kw1Hu2aP)


# Parallel Ensemble

### 1. 단일 트리모형, 배깅 트리모형, 랜덤포레스트 모형 


```python
clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
clf2 = BaggingClassifier(base_estimator=clf1, n_estimators=10, max_samples=0.8, max_features=0.8)
clf3 = RandomForestClassifier(random_state=0, n_estimators=10, max_features=0.8, criterion ='entropy')
```


```python
label = ['Decision Tree', 'Bagging Tree', 'Random Forest']
clf_list = [clf1, clf2, clf3]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 3)
grid = itertools.product([0,1,2],repeat=2)

for clf, label, grd in zip(clf_list, label, grid):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
        
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.63 (+/- 0.02) [Decision Tree]
    Accuracy: 0.67 (+/- 0.03) [Bagging Tree]
    Accuracy: 0.69 (+/- 0.02) [Random Forest]
    


![png](https://drive.google.com/uc?export=view&id=1Eal9Oq_DSY54Zecq6_ivkL4l3lkbvrKF)


#### 단일 tree모형보다 배깅 트리가, 배깅트리보단 랜덤포레스트가 더 높은 정확도를 보임

### 2. Parameter tuning한 랜덤포레스트 모형


```python
#Ensemble Size
num_est = np.linspace(1,100,20).astype(int)
bg_clf_cv_mean = []
bg_clf_cv_std = []
for n_est in num_est:    
    bg_clf = RandomForestClassifier(n_estimators=n_est, max_features=0.8)
    scores = cross_val_score(bg_clf, X, y, cv=3, scoring='accuracy')
    bg_clf_cv_mean.append(scores.mean())
    bg_clf_cv_std.append(scores.std())
    
plt.figure()
(_, caps, _) = plt.errorbar(num_est, bg_clf_cv_mean, yerr=bg_clf_cv_std, c='blue', fmt='-o', capsize=5)
for cap in caps:
    cap.set_markeredgewidth(1)                                                                                                                                
plt.ylabel('Accuracy'); plt.xlabel('Ensemble Size'); plt.title('Random Forest');
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1YwTIu_qQ-T5g60RXgcKT7qtnfk8VRrok)



```python
clf4 = RandomForestClassifier(random_state=0, n_estimators=60, max_features=0.8, criterion ='entropy')
```


```python
label = ['Random Forest_tuning']
clf_list = [clf4]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 1)

for clf, label in zip(clf_list, label):        
    scores = cross_val_score(clf, X, y, cv=2, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
        
    clf.fit(X, y)
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.73 (+/- 0.03) [Random Forest_tuning]
    


![png](https://drive.google.com/uc?export=view&id=1n6BacX9886IlzEZAFjc0lxmhI68RRrzQ)


#### 일부 parameter를 tuning한 랜덤포레스트. 예측력이 상승됨

### 3. Randomized Search를 통한 랜덤포레스트 모형


```python
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=3)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
est = RandomForestClassifier(n_jobs=-1)

rf_p_dist={'max_depth':[3,5,10,None],
               'criterion':['entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,10)
              }

rf_parameters = hypertuning_rscv(est, rf_p_dist, 40, X, y)
print(rf_parameters)
```

    ({'bootstrap': True, 'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 5}, 0.7733333333333333)
    


```python
clf5 = RandomForestClassifier(max_depth = 3, min_samples_leaf = 7,bootstrap=True,criterion='entropy', n_estimators = 500)

clf_list = [clf5]
label = ['RandomForest_RandomnizedCV']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 1)

for clf, label in zip(clf_list, label):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    
    clf.fit(X, y)
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.79 (+/- 0.03) [RandomForest_RandomnizedCV]
    


![png](https://drive.google.com/uc?export=view&id=1elrjBQ2qXJrE68EZmJSCQN54ZgaFDfrH)


#### Randomized Search를 거친 랜덤포레스트. 예측력이 제일 높음.

# Sequential Ensemble : Boosting


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm
```

### 1. 간단한 AdaBoost, GBM 모형


```python
clf6 = AdaBoostClassifier(n_estimators=1)
clf7 = GradientBoostingClassifier(random_state=0, n_estimators=1,  max_features=0.8)
```


```python
clf_list = [clf6, clf7]
label = ['AdaBoost (n_est=1)', 'GradientBoost (n_est=1)']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 2)
grid = itertools.product([0,1],repeat=2)

for clf, label, grd in zip(clf_list, label, grid):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.63 (+/- 0.02) [AdaBoost (n_est=1)]
    Accuracy: 0.65 (+/- 0.03) [GradientBoost (n_est=1)]
    


![png](https://drive.google.com/uc?export=view&id=1YixDeBtWYQKekRXKA5fJv0-fgz5exmbq)


#### 역시 예측력이 좋지 않다

### 2. 일부 parameter tuning한 GBM


```python
#Ensemble Size
num_est = np.linspace(1,100,20).astype(int)
bg_clf_cv_mean = []
bg_clf_cv_std = []
for n_est in num_est:    
    bg_clf = RandomForestClassifier(n_estimators=n_est, max_features=0.8)
    scores = cross_val_score(bg_clf, X, y, cv=3, scoring='accuracy')
    bg_clf_cv_mean.append(scores.mean())
    bg_clf_cv_std.append(scores.std())
    
plt.figure()
(_, caps, _) = plt.errorbar(num_est, bg_clf_cv_mean, yerr=bg_clf_cv_std, c='blue', fmt='-o', capsize=5)
for cap in caps:
    cap.set_markeredgewidth(1)                                                                                                                                
plt.ylabel('Accuracy'); plt.xlabel('Ensemble Size'); plt.title('Gradient Boosting');
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1utnhml3c1vQmQKxNrxc12PZ4kHEQgwAe)



```python
clf8 = GradientBoostingClassifier(random_state=0, n_estimators=90, max_features=0.8)
```


```python
clf_list = [clf8]
label = ['GradientBoost_tuning']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 1)

for clf, label in zip(clf_list, label):        
    scores = cross_val_score(clf, X, y, cv=2, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    
    clf.fit(X, y)
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.72 (+/- 0.03) [GradientBoost_tuning]
    


![png](https://drive.google.com/uc?export=view&id=1gek1odVZdKKVfk7a1TrmoNlG5diiaQ3M)


#### 일부 parameter만 tuning. 예측력이 상승했다!

### 3. Randomized Search를 통한 GBM


```python
est = GradientBoostingClassifier()
gb_p_dist={'n_estimators':[100,250,500,750],
           'max_depth':[3,5,10,None],
           'min_samples_leaf':randint(1,10),
           }

gb_parameters = hypertuning_rscv(est, gb_p_dist, 40, X, y)
print(gb_parameters)
```

    ({'max_depth': None, 'min_samples_leaf': 9, 'n_estimators': 100}, 0.7466666666666667)
    


```python
clf9 = GradientBoostingClassifier(max_depth = None, min_samples_leaf = 9, n_estimators = 100)

clf_list = [clf9]
label = ['GradientBoost_RandomnizedCV']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 1)

for clf, label in zip(clf_list, label):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    
    clf.fit(X, y)
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.75 (+/- 0.02) [GradientBoost_RandomnizedCV]
    


![png](https://drive.google.com/uc?export=view&id=1gif5UfJ2mvAherpflWWzornbSabU7apU)


#### 역시 RandomSearch를 진행하였을때가 가장 높은 예측력!

### 4. XGBoost, LightGBM


```python
clf10 = XGBClassifier(n_estimators=100,seed=1)
clf11 = lgbm.LGBMClassifier(n_estimators=100,seed=1)
```


```python
clf_list = [clf10, clf11]
label = ['XGBoost', 'LightGBM']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 2)
grid = itertools.product([0,1],repeat=2)

for clf, label, grd in zip(clf_list, label, grid):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.73 (+/- 0.01) [XGBoost]
    Accuracy: 0.75 (+/- 0.03) [LightGBM]
    


![png](https://drive.google.com/uc?export=view&id=1M2DIVmzGAOGvGl5Yxa4q8yvw_fWaeFsg)


#### Simple한 모델임에도 예측력이 높은 것을 확인할 수 있다!

# Multi-Class Classification


```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

clf12 = OneVsRestClassifier(LogisticRegression())
clf13 = OneVsRestClassifier(estimator = SGDClassifier(random_state=42))
```


```python
clf_list = [clf12, clf13]
label = ['OvR Classification_Logistic','OvR Classification_SGD']

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 2)
grid = itertools.product([0,1],repeat=2)

for clf, label, grd in zip(clf_list, label, grid):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()
```

    Accuracy: 0.75 (+/- 0.07) [OvR Classification_Logistic]
    Accuracy: 0.57 (+/- 0.06) [OvR Classification_SGD]
    


![png](https://drive.google.com/uc?export=view&id=12Xa7nos2FqciVPjJD3HUDhkJ1T6kFfNK)


#### OvR 에 적용되는 구역을 나눠주는 알고리즘에 따라 성능이 크게 차이남!

# Stacking


```python
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
```

### 1. 현재까지의 모델


```python
label = ['Decision Tree', 'Bagging Tree' , 'Random Forest', 'Random Forest_tuning', 'RandomForest_RandomnizedCV', 'AdaBoost (n_est=1)',
         'GradientBoost (n_est=1)', 'GradientBoost_tuning','GradientBoost_RandomnizedCV','XGBoost','LightGBM','OvR Classification_Logistic', 'OvR Classification_SGD']

clf_list = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13]

for clf, label in zip(clf_list, label):        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
```

    Accuracy: 0.63 (+/- 0.02) [Decision Tree]
    Accuracy: 0.71 (+/- 0.06) [Bagging Tree]
    Accuracy: 0.69 (+/- 0.02) [Random Forest]
    Accuracy: 0.71 (+/- 0.03) [Random Forest_tuning]
    Accuracy: 0.79 (+/- 0.02) [RandomForest_RandomnizedCV]
    Accuracy: 0.63 (+/- 0.02) [AdaBoost (n_est=1)]
    Accuracy: 0.65 (+/- 0.03) [GradientBoost (n_est=1)]
    Accuracy: 0.70 (+/- 0.04) [GradientBoost_tuning]
    Accuracy: 0.75 (+/- 0.02) [GradientBoost_RandomnizedCV]
    Accuracy: 0.73 (+/- 0.01) [XGBoost]
    Accuracy: 0.75 (+/- 0.03) [LightGBM]
    Accuracy: 0.75 (+/- 0.07) [OvR Classification_Logistic]
    Accuracy: 0.57 (+/- 0.06) [OvR Classification_SGD]
    

### 2. 2-level Stacking; with Logistic & with NN


```python
lr = LogisticRegression()
nn = MLPClassifier(random_state=1)

sclf1 = StackingCVClassifier(classifiers=clf_list, 
                          meta_classifier=lr)
sclf2 = StackingClassifier(classifiers=clf_list, 
                          meta_classifier=nn)
```


```python
label = ['Stacking Classifier_Logistic', 'Stacking Classifier_NN']
clf_list = [sclf1, sclf2]
    
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(1, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []

for clf, label, grd in zip(clf_list, label, grid):
        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())
        
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)

plt.show()
```

    Accuracy: 0.73 (+/- 0.08) [Stacking Classifier_Logistic]
    Accuracy: 0.71 (+/- 0.03) [Stacking Classifier_NN]
    


![png](https://drive.google.com/uc?export=view&id=1qM8TtVtA61S31VvCMmCYnxzcIixWo3_L)


#### 그다지 만족스럽지 못한 예측력

### 3. 3-level Stacking


```python
xg = XGBClassifier(n_estimators = 100)
ada = AdaBoostClassifier(n_estimators = 100)    
nn = MLPClassifier(hidden_layer_sizes=(100,50))

sclf_xgb = StackingClassifier(classifiers=clf_list, 
                            use_probas=True,
                            average_probas=False,
                          meta_classifier=xg)
sclf_ada = StackingClassifier(classifiers=clf_list,
                            use_probas=True,
                          average_probas=False,
                          meta_classifier=ada)
sclf_nn = StackingClassifier(classifiers=clf_list,
                            use_probas=True,
                          average_probas=False,
                          meta_classifier=nn)

lr = LogisticRegression(penalty = 'l2')

sclf_fin = StackingClassifier(classifiers=[sclf_xgb, sclf_ada, sclf_nn],
                               use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
```


```python
label = ['Stacking Classifier_XGB', 'Stacking Classifier_ADA', 'Stacking Classifier_NN', 'Stacking Classifier_FIN']
clf_list = [sclf_xgb, sclf_ada, sclf_nn, sclf_fin]
    
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
for clf, label, grd in zip(clf_list, label, grid):
        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())
        
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)

plt.show()
```

    Accuracy: 0.71 (+/- 0.03) [Stacking Classifier_XGB]
    Accuracy: 0.72 (+/- 0.01) [Stacking Classifier_ADA]
    Accuracy: 0.71 (+/- 0.02) [Stacking Classifier_NN]
    Accuracy: 0.72 (+/- 0.03) [Stacking Classifier_FIN]
    


![png](https://drive.google.com/uc?export=view&id=15OKBe5tHncvI2WfKYOvKIk4i0t_41uGR)


#### 그렇게 만족스럽진 않다..

### 4. 3-level Stacking with CV 


```python
xg = XGBClassifier(n_estimators = 100)
ada = AdaBoostClassifier(n_estimators = 100)    
nn = MLPClassifier(hidden_layer_sizes=(100,50))

sclf_xgb = StackingCVClassifier(classifiers=clf_list, cv=3,
                          meta_classifier=xg, random_state=1)
sclf_ada = StackingCVClassifier(classifiers=clf_list,cv=3,
                          meta_classifier=ada, random_state=1)
sclf_nn = StackingCVClassifier(classifiers=clf_list,cv=3,
                          meta_classifier=nn, random_state=1)

lr = LogisticRegression(penalty = 'l2')

sclf_fin = StackingCVClassifier(classifiers=[sclf_xgb, sclf_ada, sclf_nn],
                          meta_classifier=lr, random_state=1)
```


```python
label = ['StackingCV Classifier_XGB', 'StackingCV Classifier_ADA', 'StackingCV Classifier_NN', 'StackingCV Classifier_FIN']
clf_list = [sclf_xgb, sclf_ada, sclf_nn, sclf_fin]
    
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
for clf, label, grd in zip(clf_list, label, grid):
        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())
        
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)

plt.show()
```
