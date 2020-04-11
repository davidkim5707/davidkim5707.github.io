---
layout: post
title: machine learning_decisiontree,randomforest
categories:
  - Programming
tags:
  - Python
  - Decision Tree
  - Random Forest
last_modified_at: 2020-04-11
use_math: true
---
source& copyright: lecture note in DataScienceLab in Yonsei university  

### Decision Tree and RandomForest


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
```

# Visualization DT through Iris Dataset


```python
import matplotlib
from sklearn.tree import export_graphviz
import io
import pydot
from IPython.core.display import Image 
from sklearn.metrics import confusion_matrix
```


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
y = data.target
X = data.data[:, 2:]
feature_names = data.feature_names[2:]
```


```python
def draw_decision_tree(model):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
    return Image(image)


def plot_decision_regions(X, y, model, title):
    resolution = 0.01
    markers = ('s', '^', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = matplotlib.colors.ListedColormap(colors)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)

    plt.contour(xx1, xx2, Z, cmap=matplotlib.colors.ListedColormap(['k']))
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=[cmap(idx)], marker=markers[idx], s=80, label=cl)

    plt.xlabel(data.feature_names[2])
    plt.ylabel(data.feature_names[3])
    plt.legend(loc='upper left')
    plt.title(title)
    plt.grid(True)

    return Z
```


```python
Iris = load_iris()

Iris_Data = pd.DataFrame(data=np.c_[Iris['data'], Iris['target']], columns=Iris['feature_names']+['target'])
Iris_Data['target']=Iris_Data['target'].map({0:"setosa", 1:"versicolor", 2:"vriginica"})

X_Data=Iris_Data.iloc[:,:-1]
Y_Data=Iris_Data.iloc[:,[-1]]

Iris_Data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
Iris_Data.target.unique()
```




    array(['setosa', 'versicolor', 'vriginica'], dtype=object)




```python
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0).fit(X, y)
```


```python
draw_decision_tree(tree1)
```




![png](https://drive.google.com/uc?export=view&id=1NwWFyipEHDbpEZjaPT3HPPbmHnWgzZNQ)




```python
plot_decision_regions(X, y, tree1, "Depth 1")
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1IRvZp4dwPq6A8DY6A8Vwlvk6UH0R7q_P)



```python
confusion_matrix(y, tree1.predict(X))
```




    array([[50,  0,  0],
           [ 0, 50,  0],
           [ 0, 50,  0]])




```python
tree2 = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0).fit(X, y)
```


```python
draw_decision_tree(tree2)
```




![png](https://drive.google.com/open?id=1Rvf_JL2ZLwT1ARaCy_OmaCU6RO4xk7ol)




```python
plot_decision_regions(X, y, tree2, "Depth 2")
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1j5BJCQvp_TXWBRIl_tY9zM130OtlpAFD)



```python
confusion_matrix(y, tree2.predict(X))
```




    array([[50,  0,  0],
           [ 0, 49,  1],
           [ 0,  5, 45]])




```python
tree3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(X, y)
draw_decision_tree(tree3)
```




![png](https://drive.google.com/open?id=15b2v9HRd81jdwQwUa9PdjNS15N5b18Xy)




```python
plot_decision_regions(X, y, tree3, "Depth 3")
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1qucVwd1mS2dY5wDYhhieeKsZR6zAxRLz)



```python
confusion_matrix(y, tree3.predict(X))
```




    array([[50,  0,  0],
           [ 0, 47,  3],
           [ 0,  1, 49]])




```python
tree4 = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0).fit(X, y)
draw_decision_tree(tree4)
```




![png](https://drive.google.com/uc?export=view&id=1V-J4fHHAV7Nk4HN6KAeQBEASyFMvYkH8)




```python
plot_decision_regions(X, y, tree4, "Depth 4")
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1N9Krbf5PB_dT002JVVqmRo_0B7th-zZy)



```python
confusion_matrix(y, tree4.predict(X))
```




    array([[50,  0,  0],
           [ 0, 49,  1],
           [ 0,  1, 49]])




```python
tree5 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0).fit(X, y)
draw_decision_tree(tree5)
```




![png](https://drive.google.com/uc?export=view&id=17JD1_lXveY4IEetsuIXjdk51euhAguzM)




```python
plot_decision_regions(X, y, tree5, "Depth 5")
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=11WiPCd0y7qHCiT3yHQ2rIOCPm9y00yKp)



```python
confusion_matrix(y, tree5.predict(X))
```




    array([[50,  0,  0],
           [ 0, 49,  1],
           [ 0,  0, 50]])



## BreastCancerData from Kaggle


```python
BC = pd.read_csv("BreastCancer.txt",sep=',')
```


```python
BC.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>CT</th>
      <th>UCS</th>
      <th>UCSP</th>
      <th>MA</th>
      <th>SECS</th>
      <th>BN</th>
      <th>BC</th>
      <th>NN</th>
      <th>M</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Response is CLASS => Categorical Variable



```python
BC['CLASS'].value_counts() 
```




    2    458
    4    241
    Name: CLASS, dtype: int64



## Preprocessing data


```python
BC = BC.loc[lambda x:x['BN']!='?'] #BN column의 값이 ?이 아닌 것들만 가져옴
```


```python
BC_Response = pd.get_dummies(BC['CLASS']).iloc[:,-1] #CLASS column을 dummy 변수로 만들고 그 부분만 추출 -> (2,4)에서 (0,1)로 바뀜 
BC_X = BC.iloc[:,1:-1] #CLASS 제외 나머지 부분을 추출
```


```python
BC_Response.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: 4, dtype: uint8



## Calling 'DT_Classifier' function


```python
DT_1 = DecisionTreeClassifier(random_state=0)
```


```python
DT_1
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=0,
                splitter='best')



- criterion : 'gini'가 default ; 'entropy'도 가능
- max_depth : overfitting을 막기 위함, max-depth 어느 깊이까지 decision tree를 만들 것인가?
- min_samples_split : node안의 데이터 갯수가 최소 몇개 이상이어야만 split을 할 것인가?

## Cross Validation


```python
from sklearn import metrics
from sklearn.model_selection import cross_validate
```

## It is easy to take CV using Sklearn

- scoring에 f1_score, precision, recall을 넣고 cross_validation을 통해 각각의 값을 얻어냄


```python
np.random.seed(0)
scoring = ['f1','precision','recall'] 
scores_1 = cross_validate(DT_1, BC_X, BC_Response, scoring=scoring, cv=5,return_train_score=False)
```


```python
print("parameter가 tuning 되지 않았을 때의 F1 Score : {:.3f}".format(np.mean(scores_1['test_f1'])))
```

    parameter가 tuning 되지 않았을 때의 F1 Score : 0.908
    

## Manipulating hyperparameters of DT
- max_depth
- min_sample_split
- scikit-learn 사전 가지치기만 지원한다. (max_depth parameter tuning을 통해 트리 생성을 일찍 중단)


```python
from sklearn.model_selection import GridSearchCV
```


```python
np.random.seed(0)
hyperparamters = {'max_depth':[2,3,4,5,6,7,8,9], 
                  'min_samples_split':[2,3,4,5,6,7]} #hyperparameter를 dict type으로 넣기
GridCV = GridSearchCV(estimator=DT_1, param_grid=hyperparamters, cv=5, verbose=1) 
GridCV.fit(BC_X, BC_Response)
```

    Fitting 5 folds for each of 48 candidates, totalling 240 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 240 out of 240 | elapsed:    0.9s finished
    




    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=0,
                splitter='best'),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'max_depth': [2, 3, 4, 5, 6, 7, 8, 9], 'min_samples_split': [2, 3, 4, 5, 6, 7]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)




```python
GridCV.best_params_
```




    {'max_depth': 4, 'min_samples_split': 6}




```python
DT_2 = DecisionTreeClassifier(max_depth=4, min_samples_split=6,random_state=0)
```


```python
np.random.seed(0)
scoring = ['f1','precision','recall'] 
scores_2 = cross_validate(DT_2, BC_X, BC_Response, scoring=scoring, cv=5, return_train_score = False)
```


```python
print("parameter가 tuning 되지 않았을 때의 F1 Score : {:.3f}".format(np.mean(scores_1['test_f1'])))
print('parameter가 tuning 되었을 때의 F1 Score : {:.3f} '.format(np.mean(scores_2['test_f1'])))
```

    parameter가 tuning 되지 않았을 때의 F1 Score : 0.908
    parameter가 tuning 되었을 때의 F1 Score : 0.928 
    

## Bagging


```python
from sklearn.ensemble import BaggingClassifier
```

    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    


```python
Ens_1=BaggingClassifier(DT_1,oob_score=True,random_state=0)
```


```python
BaggingClassifier()
```




    BaggingClassifier(base_estimator=None, bootstrap=True,
             bootstrap_features=False, max_features=1.0, max_samples=1.0,
             n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
             verbose=0, warm_start=False)



- base_estimator : Decision Tree 등... 
- n_estimators : bootstrap을 몇개할 것인가?
- max_samples : 0~1사이의 값, 1이면 데이터를 활용
- max_feature : 변수를 몇개 사용할 것인가?
- oob_score : Bagging의 성능을 측정하기 위해 OOB를 이용하여 비교


```python
np.random.seed(0)
scores_3=cross_validate(Ens_1,BC_X, BC_Response, scoring=scoring, cv=5,return_train_score=False)
print("parameter가 tuning 되지 않았을 때의 F1 Score : {:.3f}".format(np.mean(scores_1['test_f1'])))
print('parameter가 tuning 되었을 때의 F1 Score : {:.3f} '.format(np.mean(scores_2['test_f1'])))
print('Bagging 했을 때의 F1 Score : {:.3f}'.format(np.mean(scores_3['test_f1'])))
```

    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:605: UserWarning: Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.
      warn("Some inputs do not have OOB scores. "
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:610: RuntimeWarning: invalid value encountered in true_divide
      predictions.sum(axis=1)[:, np.newaxis])
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:605: UserWarning: Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.
      warn("Some inputs do not have OOB scores. "
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:610: RuntimeWarning: invalid value encountered in true_divide
      predictions.sum(axis=1)[:, np.newaxis])
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:605: UserWarning: Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.
      warn("Some inputs do not have OOB scores. "
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:610: RuntimeWarning: invalid value encountered in true_divide
      predictions.sum(axis=1)[:, np.newaxis])
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:605: UserWarning: Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.
      warn("Some inputs do not have OOB scores. "
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:610: RuntimeWarning: invalid value encountered in true_divide
      predictions.sum(axis=1)[:, np.newaxis])
    

    parameter가 tuning 되지 않았을 때의 F1 Score : 0.908
    parameter가 tuning 되었을 때의 F1 Score : 0.928 
    Bagging 했을 때의 F1 Score : 0.941
    

    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:605: UserWarning: Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.
      warn("Some inputs do not have OOB scores. "
    C:\Users\user\Anaconda3\lib\site-packages\sklearn\ensemble\bagging.py:610: RuntimeWarning: invalid value encountered in true_divide
      predictions.sum(axis=1)[:, np.newaxis])
    

- parameter tuning 진행
- n_estimators 변동
- max_samples 변동


```python
np.random.seed(0)
hyperparamters = {'n_estimators':[35,45,55,65,75,85,95], 
                  'max_samples':[0.5,0.6,0.7,0.8,0.9,1]} #hyperparameter를 dict type으로 넣기
GridCV = GridSearchCV(estimator=Ens_1, param_grid=hyperparamters, cv=5, verbose=1) 
GridCV.fit(BC_X, BC_Response)
```

    Fitting 5 folds for each of 42 candidates, totalling 210 fits
    

    [Parallel(n_jobs=1)]: Done 210 out of 210 | elapsed:  1.2min finished
    




    GridSearchCV(cv=5, error_score='raise',
           estimator=BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                ....0, n_estimators=10, n_jobs=1, oob_score=True,
             random_state=0, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'n_estimators': [35, 45, 55, 65, 75, 85, 95], 'max_samples': [0.5, 0.6, 0.7, 0.8, 0.9, 1]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)




```python
GridCV.best_params_
```




    {'max_samples': 0.7, 'n_estimators': 35}




```python
Ens_2=BaggingClassifier(DT_1,max_samples=0.7,n_estimators=35,oob_score=True,random_state=0)
```


```python
np.random.seed(0)
scores_4 = cross_validate(Ens_2,BC_X,BC_Response, scoring=scoring, cv=5,return_train_score=False)
```


```python
print("parameter가 tuning 되지 않았을 때의 F1 Score : {:.3f}".format(np.mean(scores_1['test_f1'])))
print('parameter가 tuning 되었을 때의 F1 Score : {:.3f} '.format(np.mean(scores_2['test_f1'])))
print('Bagging 했을 때의 F1 Score : {:.3f}'.format(np.mean(scores_3['test_f1'])))
print("Bagging을 parameter tuning 했을 때의 F1 Score: {:.3f}".format(np.mean(scores_4['test_f1'])))
```

    parameter가 tuning 되지 않았을 때의 F1 Score : 0.908
    parameter가 tuning 되었을 때의 F1 Score : 0.928 
    Bagging 했을 때의 F1 Score : 0.941
    Bagging을 parameter tuning 했을 때의 F1 Score: 0.948
    

# RandomForest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
RandomForestClassifier()
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



- max_depth : default=None
- min_samples_split : default=2
- min_samples_leaf  : default=1
- 3가지 parameter는 동일하게 유지, tree를 deep하게 향상시키는 것이 좋기 때문

- n_estimators : bagging과 동일
- max_features : 각 split할 때, 선택할 변수의 개수 
- 만약 'auto' 라면, max_features=sqrt(n_features)
- 만약 'None' 이라면, max_features=n_features (변수개수 그 자체)
- 직접 숫자를 넣어줄 수 있다.


```python
RF_1=RandomForestClassifier(random_state=0)
```


```python
np.random.seed(0)
scores_5=cross_validate(RF_1,BC_X, BC_Response, scoring=scoring, cv=5,return_train_score=False)
```


```python
print("parameter가 tuning 되지 않았을 때의 F1 Score : {:.3f}".format(np.mean(scores_1['test_f1'])))
print('parameter가 tuning 되었을 때의 F1 Score : {:.3f} '.format(np.mean(scores_2['test_f1'])))
print('Bagging 했을 때의 F1 Score : {:.3f}'.format(np.mean(scores_3['test_f1'])))
print("Bagging을 parameter tuning 했을 때의 F1 Score: {:.3f}".format(np.mean(scores_4['test_f1'])))
print("RandomForest를 했을 때의 F1 Score : {:.3f} ".format(np.mean(scores_5['test_f1'])))
```

    parameter가 tuning 되지 않았을 때의 F1 Score : 0.908
    parameter가 tuning 되었을 때의 F1 Score : 0.928 
    Bagging 했을 때의 F1 Score : 0.941
    Bagging을 parameter tuning 했을 때의 F1 Score: 0.948
    RandomForest를 했을 때의 F1 Score : 0.942 
    


```python
np.random.seed(0)
hyperparamters = {'n_estimators':[35,45,55,65,75,85,95], 
                  'max_features':[1,2,3,4,5,6]} #hyperparameter를 dict type으로 넣기
GridCV = GridSearchCV(estimator=RF_1, param_grid=hyperparamters, cv=5, verbose=1) 
GridCV.fit(BC_X, BC_Response)
```

    Fitting 5 folds for each of 42 candidates, totalling 210 fits
    

    [Parallel(n_jobs=1)]: Done 210 out of 210 | elapsed:   42.5s finished
    




    GridSearchCV(cv=5, error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'n_estimators': [35, 45, 55, 65, 75, 85, 95], 'max_features': [1, 2, 3, 4, 5, 6]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)




```python
GridCV.best_params_
```




    {'max_features': 1, 'n_estimators': 35}




```python
np.random.seed(0)
RF_2 = RandomForestClassifier(n_estimators=35, max_features=1,random_state=0)
scores_6 = cross_validate(RF_2, BC_X, BC_Response, scoring=scoring, cv=5, return_train_score = False)
```


```python
print("parameter가 tuning 되지 않았을 때의 F1 Score : {:.3f}".format(np.mean(scores_1['test_f1'])))
print('parameter가 tuning 되었을 때의 F1 Score : {:.3f} '.format(np.mean(scores_2['test_f1'])))
print('Bagging 했을 때의 F1 Score : {:.3f}'.format(np.mean(scores_3['test_f1'])))
print("Bagging을 parameter tuning 했을 때의 F1 Score: {:.3f}".format(np.mean(scores_4['test_f1'])))
print("RandomForest를 했을 때의 F1 Score : {:.3f} ".format(np.mean(scores_5['test_f1'])))
print("RandomForest를 parameter tuning 했을 때의 F1 Score : {:.3f}".format(np.mean(scores_6['test_f1'])))
```

    parameter가 tuning 되지 않았을 때의 F1 Score : 0.908
    parameter가 tuning 되었을 때의 F1 Score : 0.928 
    Bagging 했을 때의 F1 Score : 0.941
    Bagging을 parameter tuning 했을 때의 F1 Score: 0.948
    RandomForest를 했을 때의 F1 Score : 0.942 
    RandomForest를 parameter tuning 했을 때의 F1 Score : 0.959
    

## Variable Importance plot (Feature Importance plot)
- random_state 지정 안해보고 비교
- 선택하는 변수의 갯수(m) 변화에 따른 비교


```python
RF_3=RandomForestClassifier()
RF_3.fit(BC_X,BC_Response)
importances =RF_3.feature_importances_
indices = np.argsort(importances)
feat_importances = pd.Series(RF_3.feature_importances_, index=BC_X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1n3p7ByQySmzFdinPt4Os-nEKNnyNfWsV)



```python
RF_4=RandomForestClassifier()
RF_4.fit(BC_X,BC_Response)
importances =RF_4.feature_importances_
indices = np.argsort(importances)
feat_importances = pd.Series(RF_4.feature_importances_, index=BC_X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=17TtP7XGFZ56YkuRKUaJQQMkMzSEEtOik)



```python
RF_5=RandomForestClassifier(max_features=3,random_state=0)
RF_5.fit(BC_X,BC_Response)
importances =RF_5.feature_importances_
indices = np.argsort(importances)
feat_importances = pd.Series(RF_5.feature_importances_, index=BC_X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1ceZ5cVLW0yc5FNzcEfCz8AxCyDp5piCk)



```python
RF_5=RandomForestClassifier(max_features=4,random_state=0)
RF_5.fit(BC_X,BC_Response)
importances =RF_5.feature_importances_
indices = np.argsort(importances)
feat_importances = pd.Series(RF_5.feature_importances_, index=BC_X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()
```


![png](https://drive.google.com/uc?export=view&id=1wIHJfs7UIeo1xwdbNASPgkCujcO1Fifz)


- Ensemble 모형은 Black Box지만, Feature Importance는 부분적으로 해석 가능한 부분이다.
- Random Forest는 이름 그대로 random하다. random_state option을 지정하지 않으면 전혀 다른 모델이 새롭게 만들어진다.
- 보는 바와 같이 Feature Importance는 변동될 수 있다
- EDA를 위한 과정 수준에서 적용하면 좋다.


```python

```
