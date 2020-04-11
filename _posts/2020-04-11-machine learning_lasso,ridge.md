---
layout: post
title: machine learning_lasso,ridge
categories:
  - Programming
tags:
  - Python
  - Lasso
  - Ridge
last_modified_at: 2020-04-11
use_math: true
---
source& copyright: lecture note in DataScienceLab in Yonsei university  

### Lasso & Ridge & Group Lasso


```python
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py 
import plotly.tools as tls 
import matplotlib.pyplot as plt 
```


```python
flu= pd.read_csv('flu_99_sample.csv', encoding='euc-kr')#cp949
```


```python
flu.head()
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
      <th>DT</th>
      <th>WK_CD</th>
      <th>CNT</th>
      <th>T_CNT</th>
      <th>N_CNT</th>
      <th>TEMP</th>
      <th>HIGH_TEMP</th>
      <th>LW_TEMP</th>
      <th>HM</th>
      <th>LW_HM</th>
      <th>DR</th>
      <th>WS</th>
      <th>PRESS</th>
      <th>SO2</th>
      <th>PM10</th>
      <th>NO2</th>
      <th>O2</th>
      <th>CO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20120101</td>
      <td>Hol</td>
      <td>43397</td>
      <td>4676</td>
      <td>14</td>
      <td>-0.12</td>
      <td>3.78</td>
      <td>-4.40</td>
      <td>65.62</td>
      <td>48.67</td>
      <td>0.52</td>
      <td>2.61</td>
      <td>1024.01</td>
      <td>0.009</td>
      <td>66.27</td>
      <td>0.021</td>
      <td>0.025</td>
      <td>0.971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20120102</td>
      <td>Hon</td>
      <td>517123</td>
      <td>7405</td>
      <td>37</td>
      <td>-2.36</td>
      <td>2.64</td>
      <td>-6.87</td>
      <td>62.62</td>
      <td>43.90</td>
      <td>0.22</td>
      <td>1.90</td>
      <td>1026.00</td>
      <td>0.007</td>
      <td>67.37</td>
      <td>0.024</td>
      <td>0.022</td>
      <td>0.871</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20120103</td>
      <td>TUE</td>
      <td>361808</td>
      <td>8926</td>
      <td>33</td>
      <td>-1.62</td>
      <td>2.82</td>
      <td>-5.09</td>
      <td>66.45</td>
      <td>43.49</td>
      <td>0.77</td>
      <td>2.30</td>
      <td>1021.68</td>
      <td>0.007</td>
      <td>68.88</td>
      <td>0.025</td>
      <td>0.022</td>
      <td>0.900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20120104</td>
      <td>WED</td>
      <td>336775</td>
      <td>10793</td>
      <td>32</td>
      <td>-3.79</td>
      <td>-0.94</td>
      <td>-6.08</td>
      <td>59.85</td>
      <td>42.66</td>
      <td>0.95</td>
      <td>3.25</td>
      <td>1025.20</td>
      <td>0.008</td>
      <td>52.58</td>
      <td>0.017</td>
      <td>0.027</td>
      <td>0.591</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20120105</td>
      <td>THU</td>
      <td>348399</td>
      <td>12729</td>
      <td>43</td>
      <td>-3.56</td>
      <td>1.78</td>
      <td>-7.86</td>
      <td>57.26</td>
      <td>35.96</td>
      <td>0.75</td>
      <td>2.08</td>
      <td>1028.96</td>
      <td>0.005</td>
      <td>42.04</td>
      <td>0.023</td>
      <td>0.026</td>
      <td>0.626</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_flu = flu.drop(['CNT','DT','WK_CD','PRESS'], axis=1)
y_flu = flu['CNT']
one_hot_day = pd.get_dummies(flu['WK_CD'])
X_flu = X_flu.join(one_hot_day)
```


```python
X_flu.head()
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
      <th>T_CNT</th>
      <th>N_CNT</th>
      <th>TEMP</th>
      <th>HIGH_TEMP</th>
      <th>LW_TEMP</th>
      <th>HM</th>
      <th>LW_HM</th>
      <th>DR</th>
      <th>WS</th>
      <th>SO2</th>
      <th>...</th>
      <th>CO2</th>
      <th>FRI</th>
      <th>Hol</th>
      <th>Hon</th>
      <th>MON</th>
      <th>SAT</th>
      <th>SUN</th>
      <th>THU</th>
      <th>TUE</th>
      <th>WED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4676</td>
      <td>14</td>
      <td>-0.12</td>
      <td>3.78</td>
      <td>-4.40</td>
      <td>65.62</td>
      <td>48.67</td>
      <td>0.52</td>
      <td>2.61</td>
      <td>0.009</td>
      <td>...</td>
      <td>0.971</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7405</td>
      <td>37</td>
      <td>-2.36</td>
      <td>2.64</td>
      <td>-6.87</td>
      <td>62.62</td>
      <td>43.90</td>
      <td>0.22</td>
      <td>1.90</td>
      <td>0.007</td>
      <td>...</td>
      <td>0.871</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8926</td>
      <td>33</td>
      <td>-1.62</td>
      <td>2.82</td>
      <td>-5.09</td>
      <td>66.45</td>
      <td>43.49</td>
      <td>0.77</td>
      <td>2.30</td>
      <td>0.007</td>
      <td>...</td>
      <td>0.900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10793</td>
      <td>32</td>
      <td>-3.79</td>
      <td>-0.94</td>
      <td>-6.08</td>
      <td>59.85</td>
      <td>42.66</td>
      <td>0.95</td>
      <td>3.25</td>
      <td>0.008</td>
      <td>...</td>
      <td>0.591</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12729</td>
      <td>43</td>
      <td>-3.56</td>
      <td>1.78</td>
      <td>-7.86</td>
      <td>57.26</td>
      <td>35.96</td>
      <td>0.75</td>
      <td>2.08</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.626</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



#### Ridge


```python
def ridge(X_train, y_train, X_test, y_test, alpha=0.1):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    clf = Ridge(alpha, max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.pskredict(X_test)
    y_test = y_test.ravel()
    mse = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    rsquare = r2_score(y_test,y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    print("Ridge's Rsquare : {}".format(rsquare))
    print("Ridge's RMSE : {}".format(rmse))
    print("Ridge's MAPE : {}".format(MAPE))
    return rsquare, rmse, MAPE
```


```python
X_train, X_test, y_train, y_test = train_test_split(X_flu, y_flu, test_size=0.25, random_state=0)
```


```python
alpha = 10
ridge(X_train, y_train, X_test, y_test, alpha)
print("Finish!")
```

    Ridge's Rsquare : 0.8157769447815234
    Ridge's RMSE : 222.56727662129884
    Ridge's MAPE : 33.15318408708642
    Finish!
    

### Lasso


```python
def lasso(X_train, y_train, X_test, y_test, alpha=0.1):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    clf = Lasso(alpha, max_iter=2000 ,normalize=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_test = y_test.ravel()
    mse = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    rsquare = r2_score(y_test,y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    print("Lasso's Rsquare : {}".format(rsquare))
    print("Lasso's RMSE : {}".format(rmse))
    print("Lasso's MAPE : {}".format(MAPE))
    return clf, rsquare, rmse, MAPE
```


```python
alpha = 1000 #15
clf, rsquare, rmse, MAPE = lasso(X_train, y_train, X_test, y_test, alpha)
```

    Lasso's Rsquare : 0.684262051290402
    Lasso's RMSE : 254.27354545910518
    Lasso's MAPE : 42.96796660492176
    


```python
coef_dict = dict(zip(clf.coef_, X_test.columns.tolist()))
for k in coef_dict:
    print("{} : {}".format(coef_dict[k], k))
```

    WED : 0.0
    LW_TEMP : -2310.921741187636
    LW_HM : -194.4017015542876
    NO2 : 378610.983151995
    Hol : -103228.96103009084
    MON : 71150.0891991327
    SUN : -187570.5811133673
    


```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

    C:\Users\bottl\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:625: DataConversionWarning:
    
    Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
    
    C:\Users\bottl\Anaconda3\lib\site-packages\sklearn\base.py:462: DataConversionWarning:
    
    Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
    
    C:\Users\bottl\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:625: DataConversionWarning:
    
    Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
    
    C:\Users\bottl\Anaconda3\lib\site-packages\sklearn\base.py:462: DataConversionWarning:
    
    Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
    
    


```python
alpha = 15
clf, rsquare, rmse, MAPE = lasso(X_train_scaled, y_train, X_test_scaled, y_test, alpha)
```

    Lasso's Rsquare : 0.8367584588260846
    Lasso's RMSE : 217.87259329914642
    Lasso's MAPE : 30.92048141400395
    


```python
np.std(y_test)
```




    145782.82392499596



### *Tuning hyperparameter 

Lasso 


```python
alpha_space = np.logspace(0.5, 2, 50)
Lasso_scores = []
Lasso_scores_std = []

Lasso_cv = Lasso(normalize=True)
lasso_results = dict()
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    Lasso_cv.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    Lasso_cv_scores = cross_val_score(Lasso_cv, X_train_scaled, y_train, cv=5)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    lasso_cv_avg = np.mean(Lasso_cv_scores)
    Lasso_scores.append(lasso_cv_avg)
    lasso_results[alpha] = lasso_cv_avg
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    Lasso_scores_std.append(np.std(Lasso_cv_scores))
    

# Use this function to create a plot    
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Display the plot
display_plot(Lasso_scores, Lasso_scores_std)
lasso_lambda = max(lasso_results, key=lasso_results.get)
print(lasso_lambda)
```


![png](https://drive.google.com/uc?export=view&id=1o5RfF5Gy8PBKONRynwki_GpXsRcGL4kF)


    3.1622776601683795
    


```python
alpha = 3.162
clf, rsquare, rmse, MAPE = lasso(X_train, y_train, X_test, y_test, alpha)
```

    Lasso's Rsquare : 0.8339772342347973
    Lasso's RMSE : 217.23948600933207
    Lasso's MAPE : 31.268360835814928
    

Ridge


```python
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge_cv = Ridge(normalize=True)
ridge_results = dict()
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge_cv.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge_cv, X_flu, y_flu, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_cv_avg = np.mean(ridge_cv_scores)
    ridge_scores.append(ridge_cv_avg)
    ridge_results[alpha] = ridge_cv_avg
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Use this function to create a plot    
def display_plot2(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Display the plot
display_plot2(ridge_scores, ridge_scores_std)
ridge_lambda = max(ridge_results, key=ridge_results.get)
print(ridge_lambda)
```


![png](https://drive.google.com/uc?export=view&id=1upMoXqx-mfwj9iZzUC9DRay_6LxUezNI)


    0.07196856730011514
    


```python
alpha = 0.071968
ridge(X_train, y_train, X_test, y_test, alpha)
print("Finish!")
```

    Ridge's Rsquare : 0.8283349570617056
    Ridge's RMSE : 220.4131142620761
    Ridge's MAPE : 32.111631673397696
    Finish!
    

#### Group Lasso 


```python
from grouplasso import GroupLassoRegressor
import numpy as np
```


```python
np.random.seed(0)
X = np.random.randn(10, 3)
# target variable is strongly correlated with 0th feature.
y = X[:, 0] + np.random.randn(10) * 0.1
```


```python
group_ids = np.array([0, 0, 1])
```


```python
model = GroupLassoRegressor(group_ids=group_ids, random_state=42, verbose=False, alpha=1e-1)
model.fit(X, y)
```


```python
model.coef_
```




    array([ 0.84795902, -0.01193463, -0.        ])




```python
X_train.head()
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
      <th>T_CNT</th>
      <th>N_CNT</th>
      <th>TEMP</th>
      <th>HIGH_TEMP</th>
      <th>LW_TEMP</th>
      <th>HM</th>
      <th>LW_HM</th>
      <th>DR</th>
      <th>WS</th>
      <th>SO2</th>
      <th>...</th>
      <th>CO2</th>
      <th>FRI</th>
      <th>Hol</th>
      <th>Hon</th>
      <th>MON</th>
      <th>SAT</th>
      <th>SUN</th>
      <th>THU</th>
      <th>TUE</th>
      <th>WED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>452</th>
      <td>7148</td>
      <td>22</td>
      <td>9.86</td>
      <td>17.13</td>
      <td>3.23</td>
      <td>49.93</td>
      <td>23.51</td>
      <td>0.02</td>
      <td>2.17</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.582</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>337</th>
      <td>15239</td>
      <td>37</td>
      <td>3.73</td>
      <td>8.82</td>
      <td>-0.68</td>
      <td>75.11</td>
      <td>49.17</td>
      <td>3.71</td>
      <td>2.82</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.789</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>509</th>
      <td>3023</td>
      <td>31</td>
      <td>21.69</td>
      <td>29.61</td>
      <td>14.63</td>
      <td>62.68</td>
      <td>30.19</td>
      <td>0.00</td>
      <td>1.64</td>
      <td>0.006</td>
      <td>...</td>
      <td>0.640</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>8435</td>
      <td>29</td>
      <td>7.74</td>
      <td>12.81</td>
      <td>3.31</td>
      <td>75.12</td>
      <td>52.47</td>
      <td>0.77</td>
      <td>2.54</td>
      <td>0.004</td>
      <td>...</td>
      <td>0.652</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>451</th>
      <td>6926</td>
      <td>48</td>
      <td>9.13</td>
      <td>16.82</td>
      <td>2.17</td>
      <td>52.41</td>
      <td>25.13</td>
      <td>0.07</td>
      <td>1.84</td>
      <td>0.005</td>
      <td>...</td>
      <td>0.691</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
group_ids = np.array([1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5,5,5])
model = GroupLassoRegressor(group_ids=group_ids, alpha=0.0001, max_iter=10,verbose=True)
model.fit(X_train, y_train)

model.coef_
y_pred = model.predict(X_test)
mean_absolute_error(y_test, y_pred)
```

    training loss: 100636716171.073
    training loss: 4.003322108137971e+24
    training loss: 2.1606407615828856e+38
    training loss: 1.166123628177799e+52
    training loss: 6.293708516349248e+65
    training loss: 3.396789665488846e+79
    training loss: 1.8332879575847778e+93
    training loss: 9.89447409585685e+106
    training loss: 5.340165860389934e+120
    training loss: 2.8821513038692322e+134
    

    C:\Users\bottl\Anaconda3\lib\site-packages\grouplasso\model.py:138: UserWarning:
    
    Failed to converge. Increase the number of iterations.
    
    




    1.1088491646308607e+74


