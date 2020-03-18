---
layout: post
title: Technical Note for 2nd paper.
categories:
  - Technical Notes
tags:
  - Technical Notes
last_modified_at: 2020-03-15
use_math: true
---

### Technical note
Prediction of industrial accident workers' return to the original work, using Machine Learning.

# 1.Data

### 1)load data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
```


```python
df = pd.read_csv('C:/Users/Dawis Kim/Dropbox/산재보험패널/1st_cohort/STATA/1st_cohort_treated_variable.csv')
df=df[["pid", "order_id", "rtor_dummy", "js_dummy", "med_dummy", "psy_dummy", "sex_dummy", "age_dummy", "spouse_dummy", 
       "edu_dummy", "work_dummy", "con_dummy", "dis_dummy", "dis_level", "dis_range", "emplev_dummy", "emplev_cont", "labor_dummy", 
       "labortime_dummy", "laborchange_dummy", "laborcontract_dummy", "satisfaction_dummy", "ralation_dummy", "income", "smoke_dummy", 
       "alcohol_dummy", "workday_dummy", "workhour_dummy", "doctorex_dummy", "recoveryex_dummy", "cureperiod_dummy", "jobdoctorex_dummy", 
       "jobreturnopinion_dummy"]]
df=df[df["dis_range"]>=2]
df=df[df["order_id"]==1]

#df1 = df[df["emplev_dummy"]==1] #상용직
#df2 = df[df["emplev_dummy"]!=1] #비상용직
```

### 2)classify the data


```python
common_variables = ["rtor_dummy", "sex_dummy", "age_dummy", "spouse_dummy", "edu_dummy", "work_dummy", "con_dummy", "dis_dummy",
                    "dis_level", "dis_range", "emplev_cont", "labor_dummy", "labortime_dummy", "laborchange_dummy", 
                    "laborcontract_dummy", "satisfaction_dummy", "ralation_dummy", "income", "smoke_dummy", "alcohol_dummy", 
                    "workday_dummy", "workhour_dummy", "doctorex_dummy", "recoveryex_dummy", "cureperiod_dummy", "jobdoctorex_dummy", 
                    "jobreturnopinion_dummy"]


common_sample = df[[*common_variables]]  #산재 서비스는 없음.
jobcondition_features =  ['emplev_dummy', 'income', 'labor_dummy', 'emplev_cont', 'workhour_dummy']
label_features = ["rtor_dummy"]
```

### 3)split train, test data


```python
from sklearn.model_selection import train_test_split

cx = common_sample[sorted(list(set(common_sample.columns) - set(jobcondition_features) - set(label_features)))]
cy = common_sample[label_features]
cx_train, cx_test, cy_train, cy_test = train_test_split(cx, cy, test_size=0.3, random_state=42)
```


```python
cx.columns
```




    Index(['age_dummy', 'alcohol_dummy', 'con_dummy', 'cureperiod_dummy',
           'dis_dummy', 'dis_level', 'dis_range', 'doctorex_dummy', 'edu_dummy',
           'jobdoctorex_dummy', 'jobreturnopinion_dummy', 'laborchange_dummy',
           'laborcontract_dummy', 'labortime_dummy', 'ralation_dummy',
           'recoveryex_dummy', 'satisfaction_dummy', 'sex_dummy', 'smoke_dummy',
           'spouse_dummy', 'work_dummy', 'workday_dummy'],
          dtype='object')



# 2. Modeling


```python
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb 
import xgboost as xgb 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
```


```python
def plot_feature_importances(model):
    n_features = cx.columns
    model.fit(cx_train, cy_train)
    plt.barh(n_features, model.feature_importances_, align='center')
    plt.yticks(np.arange(len(n_features)))
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    
    crucial_features = pd.DataFrame({'feature':n_features, 'importance':model.feature_importances_})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features = crucial_features[:7]
    return crucial_features

   
```

## 1) Tree-based Model
- Decision Tree classifier
- Random Forest classifie
- support vector machine
- Ligth Gradient Boost Model
- XGboost

### Decision Tree Classifier

#### before parmeter tuning


```python
dt_class = DecisionTreeClassifier(random_state = 42)
```


```python
temp_dt = plot_feature_importances(dt_class)
temp_dt
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>work_dummy</td>
      <td>0.201468</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ralation_dummy</td>
      <td>0.172911</td>
    </tr>
    <tr>
      <th>21</th>
      <td>workday_dummy</td>
      <td>0.095238</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dis_level</td>
      <td>0.057660</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age_dummy</td>
      <td>0.052661</td>
    </tr>
    <tr>
      <th>16</th>
      <td>satisfaction_dummy</td>
      <td>0.051530</td>
    </tr>
    <tr>
      <th>8</th>
      <td>edu_dummy</td>
      <td>0.046915</td>
    </tr>
  </tbody>
</table>
</div>




![png](https://drive.google.com/uc?export=view&id=195LXebp9EuafxdpmYTz0joQ_QcrUrmfu)



```python
selected_variables_dt = sorted(list(set(common_sample.columns) - set(jobcondition_features)))
```


```python
sample1_dt = df[["js_dummy", "psy_dummy", "med_dummy", *selected_variables_dt]] # 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample2_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", *selected_variables_dt]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample3_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", *selected_variables_dt]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample4_dt = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", *selected_variables_dt]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample5_dt = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample6_dt = df[["js_dummy", "psy_dummy", "med_dummy", "income", *selected_variables_dt]] # 임금/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample7_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", *selected_variables_dt]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample8_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", *selected_variables_dt]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample9_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", *selected_variables_dt]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample10_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample11_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample12_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample13_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample14_dt = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample15_dt = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample16_dt = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample17_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", *selected_variables_dt]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample18_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", *selected_variables_dt]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample19_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "income", *selected_variables_dt]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample20_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample21_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample22_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample23_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample24_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample25_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample26_dt = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample27_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_dt]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample28_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_dt]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample29_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_dt]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample30_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample31_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample32_dt = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_dt]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스
```


```python
samples_dt = [sample1_dt, sample2_dt, sample3_dt, sample4_dt, sample5_dt, sample6_dt, sample7_dt, sample8_dt, sample9_dt, sample10_dt
             , sample11_dt, sample12_dt, sample13_dt, sample14_dt, sample15_dt, sample16_dt, sample17_dt, sample18_dt, sample19_dt, sample20_dt, 
              sample21_dt, sample22_dt, sample23_dt, sample24_dt, sample25_dt, sample26_dt
             , sample27_dt, sample28_dt, sample29_dt, sample30_dt, sample31_dt, sample32_dt]
features_dt=[]
X_dt = []
y_dt = []
X_train_dt = []
X_test_dt = []
y_train_dt = []
y_test_dt = []

for i,sample in enumerate(samples_dt):
    features_dt.append(sorted(list(set(sample.columns) - set(label_features))))
    X_dt.append(sample[features_dt[i]])
    y_dt.append(sample[label_features])
    X_train_dt.append(train_test_split(X_dt[i], y_dt[i], test_size=0.3, random_state=42)[0])
    X_test_dt.append(train_test_split(X_dt[i], y_dt[i], test_size=0.3, random_state=42)[1])
    y_train_dt.append(train_test_split(X_dt[i], y_dt[i], test_size=0.3, random_state=42)[2])
    y_test_dt.append(train_test_split(X_dt[i], y_dt[i], test_size=0.3, random_state=42)[3])
    X_train_dt[i] = X_train_dt[i].reset_index(drop=True)
    X_test_dt[i] = X_test_dt[i].reset_index(drop=True)                   
    y_train_dt[i] = y_train_dt[i].reset_index(drop=True)
    y_test_dt[i] = y_test_dt[i].reset_index(drop=True)
```


```python
for i in range(2):
    print('---- sample{} ----'.format(i+1))
    accuracy = cross_val_score(dt_class, X_train_dt[i], y_train_dt[i], scoring='accuracy', cv = 5).mean() * 100
    print("Accuracy of Decision Tree is: " , accuracy)
```

    ---- sample1 ----
    Accuracy of Decision Tree is:  66.5909090909091
    ---- sample2 ----
    Accuracy of Decision Tree is:  66.66666666666666
    

#### randomsearch parameter tuning


```python
dt_params = {'max_depth':np.arange(2, 30), 
            'min_samples_split':np.arange(2, 10)}
params_list = [dt_params]

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=5, scoring='accuracy', random_state=42)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
best_param_dict_dt = dict()
samples_dt = ['sample1_dt', 'sample2_dt', 'sample3_dt', 'sample4_dt', 'sample5_dt', 'sample6_dt', 'sample7_dt', 'sample8_dt',
              'sample9_dt', 'sample10_dt', 'sample11_dt', 'sample12_dt', 'sample13_dt', 'sample14_dt', 'sample15_dt', 'sample16_dt',
             'sample17_dt', 'sample18_dt', 'sample19_dt', 'sample20_dt', 'sample21_dt', 'sample22_dt', 'sample23_dt', 'sample24_dt', 
              'sample25_dt', 'sample26_dt', 'sample27_dt', 'sample28_dt', 'sample29_dt', 'sample30_dt', 'sample31_dt', 'sample32_dt']

for i,sample in enumerate(samples_dt):
    print('---sample{}_dt---'.format(i+1))
    best_params = hypertuning_rscv(dt_class, dt_params, 30, X_train_dt[i], y_train_dt[i])
    best_param_dict_dt[sample] = best_params[0]
    print('best_params : ', best_params[0])
```

    ---sample1_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample2_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample3_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample4_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample5_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample6_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample7_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample8_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 5}
    ---sample9_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample10_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample11_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample12_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample13_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample14_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample15_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample16_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample17_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 5}
    ---sample18_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample19_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample20_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample21_dt---
    best_params :  {'min_samples_split': 8, 'max_depth': 5}
    ---sample22_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample23_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample24_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample25_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample26_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample27_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample28_dt---
    best_params :  {'min_samples_split': 8, 'max_depth': 5}
    ---sample29_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample30_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 5}
    ---sample31_dt---
    best_params :  {'min_samples_split': 3, 'max_depth': 3}
    ---sample32_dt---
    best_params :  {'min_samples_split': 8, 'max_depth': 5}
    

#### score test


```python
samples_dt = ['sample1_dt', 'sample2_dt', 'sample3_dt', 'sample4_dt', 'sample5_dt', 'sample6_dt', 'sample7_dt', 'sample8_dt',
              'sample9_dt', 'sample10_dt', 'sample11_dt', 'sample12_dt', 'sample13_dt', 'sample14_dt', 'sample15_dt', 'sample16_dt',
             'sample17_dt', 'sample18_dt', 'sample19_dt', 'sample20_dt', 'sample21_dt', 'sample22_dt', 'sample23_dt', 'sample24_dt', 
              'sample25_dt', 'sample26_dt', 'sample27_dt', 'sample28_dt', 'sample29_dt', 'sample30_dt', 'sample31_dt', 'sample32_dt']
accuracy = []
precision = []
sensitivity = []
Auc = []
for i,sample in enumerate(samples_dt):
    print('--------------sample{}_dt--------------'.format(i+1))
    clf = DecisionTreeClassifier(random_state=42, **best_param_dict_dt[sample])
    clf.fit(X_train_dt[i], y_train_dt[i])
       
    y_pred_dt = clf.predict(X_test_dt[i])
    y_pred_proba_dt = clf.predict_proba(X_test_dt[i])
    print("accuracy_score: {}".format( cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='accuracy', cv = 5).mean() * 100))
    accuracy.append(cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='accuracy', cv = 5).mean() * 100)
    print("precision_score: {}".format( cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='precision', cv = 5).mean() * 100))
    precision.append(cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='precision', cv = 5).mean() * 100)
    print("sensitivity_score: {}".format( cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='recall', cv = 5).mean() * 100))
    sensitivity.append(cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='recall', cv = 5).mean() * 100)
    try:
        print("AUC: Area Under Curve: {}".format(cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='roc_auc', cv = 5).mean() * 100))
        Auc.append(cross_val_score(dt_class, y_test_dt[i], y_pred_dt, scoring='roc_auc', cv = 5).mean() * 100)
    except ValueError:
        pass
    
score_dt = pd.DataFrame({'accuracy':accuracy, 'precision':precision, 'sensitivity': sensitivity, 'Auc': Auc})
    #y_score = clf.decision_path(X_test[i])
    #precision, recall, _ = precision_recall_curve(y_test[i], y_score[1])
    #fpr, tpr, _ = roc_curve(y_test[i], y_score)
    
    #f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.set_size_inches((8, 4)) 
    #axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #axes[0].set_title('Recall-Precision Curve')
    
    #axes[1].plot(fpr, tpr)
    #axes[1].plot([0, 1], [0, 1], linestyle='--')
    #axes[1].set_title('ROC curve')
```

    --------------sample1_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample2_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample3_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample4_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample5_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample6_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample7_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample8_dt--------------
    accuracy_score: 77.93976090669152
    precision_score: 61.369175627240146
    sensitivity_score: 71.92513368983958
    AUC: Area Under Curve: 76.18092127529954
    --------------sample9_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample10_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample11_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample12_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample13_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample14_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample15_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample16_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample17_dt--------------
    accuracy_score: 77.93976090669152
    precision_score: 61.16845878136201
    sensitivity_score: 71.92513368983958
    AUC: Area Under Curve: 76.18092127529954
    --------------sample18_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample19_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample20_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample21_dt--------------
    accuracy_score: 78.29374320757645
    precision_score: 62.68288854003139
    sensitivity_score: 72.0672268907563
    AUC: Area Under Curve: 76.5399425593022
    --------------sample22_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample23_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample24_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample25_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample26_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample27_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample28_dt--------------
    accuracy_score: 78.11830461108524
    precision_score: 62.127332984475835
    sensitivity_score: 71.89915966386555
    AUC: Area Under Curve: 76.34831400914797
    --------------sample29_dt--------------
    accuracy_score: 76.18848004968173
    precision_score: 76.61407511407512
    sensitivity_score: 63.33333333333334
    AUC: Area Under Curve: 74.47552447552448
    --------------sample30_dt--------------
    accuracy_score: 78.11985716503649
    precision_score: 62.74712643678162
    sensitivity_score: 71.5798319327731
    AUC: Area Under Curve: 76.26378808582828
    --------------sample31_dt--------------
    accuracy_score: 75.82673497904054
    precision_score: 79.18380153243855
    sensitivity_score: 62.4
    AUC: Area Under Curve: 74.40436507936509
    --------------sample32_dt--------------
    accuracy_score: 77.5904362676603
    precision_score: 62.73918992884511
    sensitivity_score: 70.42857142857143
    AUC: Area Under Curve: 75.61675244586637
    


```python
score_dt
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
      <th>accuracy</th>
      <th>precision</th>
      <th>sensitivity</th>
      <th>Auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>6</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77.939761</td>
      <td>61.369176</td>
      <td>71.925134</td>
      <td>76.180921</td>
    </tr>
    <tr>
      <th>8</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>9</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>10</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>11</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>12</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>13</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>14</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>15</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>16</th>
      <td>77.939761</td>
      <td>61.168459</td>
      <td>71.925134</td>
      <td>76.180921</td>
    </tr>
    <tr>
      <th>17</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>18</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>19</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>20</th>
      <td>78.293743</td>
      <td>62.682889</td>
      <td>72.067227</td>
      <td>76.539943</td>
    </tr>
    <tr>
      <th>21</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>22</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>23</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>24</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>25</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>26</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>27</th>
      <td>78.118305</td>
      <td>62.127333</td>
      <td>71.899160</td>
      <td>76.348314</td>
    </tr>
    <tr>
      <th>28</th>
      <td>76.188480</td>
      <td>76.614075</td>
      <td>63.333333</td>
      <td>74.475524</td>
    </tr>
    <tr>
      <th>29</th>
      <td>78.119857</td>
      <td>62.747126</td>
      <td>71.579832</td>
      <td>76.263788</td>
    </tr>
    <tr>
      <th>30</th>
      <td>75.826735</td>
      <td>79.183802</td>
      <td>62.400000</td>
      <td>74.404365</td>
    </tr>
    <tr>
      <th>31</th>
      <td>77.590436</td>
      <td>62.739190</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
  </tbody>
</table>
</div>




```python
samples_dt = ['sample1_dt', 'sample2_dt', 'sample3_dt', 'sample4_dt', 'sample5_dt', 'sample6_dt', 'sample7_dt', 'sample8_dt',
              'sample9_dt', 'sample10_dt', 'sample11_dt', 'sample12_dt', 'sample13_dt', 'sample14_dt', 'sample15_dt', 'sample16_dt',
             'sample17_dt', 'sample18_dt', 'sample19_dt', 'sample20_dt', 'sample21_dt', 'sample22_dt', 'sample23_dt', 'sample24_dt', 
              'sample25_dt', 'sample26_dt', 'sample27_dt', 'sample28_dt', 'sample29_dt', 'sample30_dt', 'sample31_dt', 'sample32_dt']

samples_dt1 = [sample1_dt, sample2_dt, sample3_dt, sample4_dt, sample5_dt, sample6_dt, sample7_dt, sample8_dt, sample9_dt, sample10_dt
             , sample11_dt, sample12_dt, sample13_dt, sample14_dt, sample15_dt, sample16_dt,sample17_dt, sample18_dt, sample19_dt, sample20_dt, 
              sample21_dt, sample22_dt, sample23_dt, sample24_dt, sample25_dt, sample26_dt
             , sample27_dt, sample28_dt, sample29_dt, sample30_dt, sample31_dt, sample32_dt]

feature_order = []
psyservice_order = []
psyservice_score = []
jobservice_order = []
jobservice_score = []
medservice_order = []
medservice_score = []

for i,(sample,sample1) in enumerate(zip(samples_dt, samples_dt1)):
    print('--------------sample{}_dt--------------'.format(i+1))
    clf = DecisionTreeClassifier(random_state=42, **best_param_dict_dt[sample])
    clf.fit(X_train_dt[i], y_train_dt[i])
    
    crucial_features = pd.DataFrame({'feature':list(set(sample1.columns) - set(label_features)), 'importance':clf.feature_importances_})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features.reset_index(drop = True, inplace=True)
    
    feature = pd.DataFrame({sample: crucial_features['feature']})
    feature_order.append(feature)
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')])
    psyservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].index.to_list())
    psyservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')])
    jobservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].index.to_list())
    jobservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')])
    medservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].index.to_list())
    medservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].to_list())

feature_order_dt = pd.concat(feature_order, axis =1 )
service_dt = pd.DataFrame({'psy_order': psyservice_order, 'psy_score': psyservice_score, 'job_order': jobservice_order,
                          'job_score': jobservice_score, 'med_order': medservice_order, 'med_score': medservice_score}, index = samples_dt)
for col in service_dt.columns:
       service_dt[col] = service_dt[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
```

    --------------sample1_dt--------------
    16    0.0
    Name: importance, dtype: float64
    21    0.0
    Name: importance, dtype: float64
    9    0.0
    Name: importance, dtype: float64
    --------------sample2_dt--------------
    1    0.291926
    Name: importance, dtype: float64
    22    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample3_dt--------------
    17    0.0
    Name: importance, dtype: float64
    22    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample4_dt--------------
    17    0.0
    Name: importance, dtype: float64
    22    0.0
    Name: importance, dtype: float64
    9    0.0
    Name: importance, dtype: float64
    --------------sample5_dt--------------
    16    0.0
    Name: importance, dtype: float64
    22    0.0
    Name: importance, dtype: float64
    0    0.513574
    Name: importance, dtype: float64
    --------------sample6_dt--------------
    17    0.0
    Name: importance, dtype: float64
    22    0.0
    Name: importance, dtype: float64
    9    0.0
    Name: importance, dtype: float64
    --------------sample7_dt--------------
    18    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    11    0.0
    Name: importance, dtype: float64
    --------------sample8_dt--------------
    1    0.238084
    Name: importance, dtype: float64
    17    0.0
    Name: importance, dtype: float64
    13    0.0
    Name: importance, dtype: float64
    --------------sample9_dt--------------
    18    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    0    0.488671
    Name: importance, dtype: float64
    --------------sample10_dt--------------
    1    0.291926
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample11_dt--------------
    18    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample12_dt--------------
    17    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample13_dt--------------
    18    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample14_dt--------------
    17    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    0    0.513574
    Name: importance, dtype: float64
    --------------sample15_dt--------------
    18    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    9    0.0
    Name: importance, dtype: float64
    --------------sample16_dt--------------
    17    0.0
    Name: importance, dtype: float64
    23    0.0
    Name: importance, dtype: float64
    0    0.513574
    Name: importance, dtype: float64
    --------------sample17_dt--------------
    26    0.0
    Name: importance, dtype: float64
    18    0.0
    Name: importance, dtype: float64
    20    0.0
    Name: importance, dtype: float64
    --------------sample18_dt--------------
    1    0.291926
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    11    0.0
    Name: importance, dtype: float64
    --------------sample19_dt--------------
    19    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    11    0.0
    Name: importance, dtype: float64
    --------------sample20_dt--------------
    19    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    0    0.488671
    Name: importance, dtype: float64
    --------------sample21_dt--------------
    1    0.237266
    Name: importance, dtype: float64
    16    0.0
    Name: importance, dtype: float64
    12    0.0
    Name: importance, dtype: float64
    --------------sample22_dt--------------
    19    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    0    0.488671
    Name: importance, dtype: float64
    --------------sample23_dt--------------
    18    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample24_dt--------------
    19    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample25_dt--------------
    18    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample26_dt--------------
    18    0.0
    Name: importance, dtype: float64
    24    0.0
    Name: importance, dtype: float64
    0    0.513574
    Name: importance, dtype: float64
    --------------sample27_dt--------------
    1    0.291926
    Name: importance, dtype: float64
    25    0.0
    Name: importance, dtype: float64
    11    0.0
    Name: importance, dtype: float64
    --------------sample28_dt--------------
    27    0.0
    Name: importance, dtype: float64
    18    0.0
    Name: importance, dtype: float64
    20    0.0
    Name: importance, dtype: float64
    --------------sample29_dt--------------
    1    0.291926
    Name: importance, dtype: float64
    25    0.0
    Name: importance, dtype: float64
    11    0.0
    Name: importance, dtype: float64
    --------------sample30_dt--------------
    5    0.030632
    Name: importance, dtype: float64
    19    0.0
    Name: importance, dtype: float64
    0    0.390737
    Name: importance, dtype: float64
    --------------sample31_dt--------------
    19    0.0
    Name: importance, dtype: float64
    25    0.0
    Name: importance, dtype: float64
    10    0.0
    Name: importance, dtype: float64
    --------------sample32_dt--------------
    1    0.236372
    Name: importance, dtype: float64
    16    0.0
    Name: importance, dtype: float64
    11    0.0
    Name: importance, dtype: float64
    


```python
feature_order_dt
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
      <th>sample1_dt</th>
      <th>sample2_dt</th>
      <th>sample3_dt</th>
      <th>sample4_dt</th>
      <th>sample5_dt</th>
      <th>sample6_dt</th>
      <th>sample7_dt</th>
      <th>sample8_dt</th>
      <th>sample9_dt</th>
      <th>sample10_dt</th>
      <th>...</th>
      <th>sample23_dt</th>
      <th>sample24_dt</th>
      <th>sample25_dt</th>
      <th>sample26_dt</th>
      <th>sample27_dt</th>
      <th>sample28_dt</th>
      <th>sample29_dt</th>
      <th>sample30_dt</th>
      <th>sample31_dt</th>
      <th>sample32_dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>work_dummy</td>
      <td>work_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>...</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>work_dummy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>laborcontract_dummy</td>
      <td>psy_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>...</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>psy_dummy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>satisfaction_dummy</td>
      <td>laborcontract_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>...</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>jobdoctorex_dummy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>labortime_dummy</td>
      <td>satisfaction_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>satisfaction_dummy</td>
      <td>labortime_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>...</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>satisfaction_dummy</td>
      <td>age_dummy</td>
      <td>satisfaction_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>labortime_dummy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alcohol_dummy</td>
      <td>smoke_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>smoke_dummy</td>
      <td>spouse_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>...</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>alcohol_dummy</td>
      <td>spouse_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborcontract_dummy</td>
      <td>doctorex_dummy</td>
      <td>laborcontract_dummy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>...</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>laborcontract_dummy</td>
      <td>psy_dummy</td>
      <td>laborcontract_dummy</td>
      <td>income</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>dis_range</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>...</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>satisfaction_dummy</td>
      <td>edu_dummy</td>
      <td>spouse_dummy</td>
      <td>edu_dummy</td>
      <td>spouse_dummy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>...</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_range</td>
      <td>cureperiod_dummy</td>
      <td>satisfaction_dummy</td>
      <td>cureperiod_dummy</td>
      <td>satisfaction_dummy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>age_dummy</td>
      <td>workday_dummy</td>
      <td>labortime_dummy</td>
      <td>labortime_dummy</td>
      <td>...</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>sex_dummy</td>
      <td>age_dummy</td>
      <td>dis_range</td>
      <td>age_dummy</td>
      <td>dis_range</td>
    </tr>
    <tr>
      <th>9</th>
      <td>med_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>labortime_dummy</td>
      <td>con_dummy</td>
      <td>age_dummy</td>
      <td>age_dummy</td>
      <td>...</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>labortime_dummy</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>laborchange_dummy</td>
      <td>emplev_cont</td>
      <td>jobreturnopinion_dummy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>med_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>work_dummy</td>
      <td>edu_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>...</td>
      <td>med_dummy</td>
      <td>med_dummy</td>
      <td>med_dummy</td>
      <td>laborchange_dummy</td>
      <td>emplev_cont</td>
      <td>alcohol_dummy</td>
      <td>emplev_cont</td>
      <td>income</td>
      <td>med_dummy</td>
      <td>doctorex_dummy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>smoke_dummy</td>
      <td>emplev_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>age_dummy</td>
      <td>emplev_dummy</td>
      <td>emplev_dummy</td>
      <td>...</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>labor_dummy</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>doctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>emplev_dummy</td>
      <td>doctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>...</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_dummy</td>
      <td>emplev_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>smoke_dummy</td>
      <td>ralation_dummy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>con_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>...</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>work_dummy</td>
      <td>laborchange_dummy</td>
      <td>work_dummy</td>
      <td>alcohol_dummy</td>
      <td>edu_dummy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sex_dummy</td>
      <td>con_dummy</td>
      <td>doctorex_dummy</td>
      <td>labor_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>...</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>labor_dummy</td>
      <td>smoke_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_level</td>
      <td>labor_dummy</td>
      <td>cureperiod_dummy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>jobreturnopinion_dummy</td>
      <td>doctorex_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
      <td>cureperiod_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>...</td>
      <td>labor_dummy</td>
      <td>labor_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>doctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>con_dummy</td>
      <td>age_dummy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>psy_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>doctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>...</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
      <td>ralation_dummy</td>
      <td>con_dummy</td>
      <td>edu_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>workday_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>...</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>income</td>
      <td>labor_dummy</td>
      <td>labortime_dummy</td>
      <td>sex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>emplev_cont</td>
    </tr>
    <tr>
      <th>18</th>
      <td>dis_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>dis_level</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>...</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>labortime_dummy</td>
      <td>income</td>
      <td>recoveryex_dummy</td>
    </tr>
    <tr>
      <th>19</th>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>workday_dummy</td>
      <td>emplev_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>...</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>emplev_dummy</td>
      <td>income</td>
      <td>js_dummy</td>
      <td>psy_dummy</td>
      <td>workhour_dummy</td>
    </tr>
    <tr>
      <th>20</th>
      <td>recoveryex_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>...</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>med_dummy</td>
      <td>workday_dummy</td>
      <td>recoveryex_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
    </tr>
    <tr>
      <th>21</th>
      <td>js_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_level</td>
      <td>smoke_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>...</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>sex_dummy</td>
      <td>workhour_dummy</td>
      <td>laborchange_dummy</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ralation_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>recoveryex_dummy</td>
      <td>alcohol_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>...</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_dummy</td>
      <td>smoke_dummy</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>js_dummy</td>
      <td>labor_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>...</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_level</td>
      <td>con_dummy</td>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
    </tr>
    <tr>
      <th>24</th>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>sex_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>...</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>recoveryex_dummy</td>
      <td>labor_dummy</td>
      <td>recoveryex_dummy</td>
      <td>smoke_dummy</td>
      <td>recoveryex_dummy</td>
      <td>con_dummy</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>...</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>js_dummy</td>
      <td>recoveryex_dummy</td>
      <td>js_dummy</td>
      <td>alcohol_dummy</td>
      <td>js_dummy</td>
      <td>labor_dummy</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>spouse_dummy</td>
      <td>laborchange_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>...</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>dis_level</td>
      <td>ralation_dummy</td>
      <td>doctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>sex_dummy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>psy_dummy</td>
      <td>dis_range</td>
      <td>con_dummy</td>
      <td>dis_range</td>
      <td>dis_level</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
      <td>spouse_dummy</td>
      <td>workhour_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_dummy</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>emplev_dummy</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 32 columns</p>
</div>




```python
service_dt
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
      <th>psy_order</th>
      <th>psy_score</th>
      <th>job_order</th>
      <th>job_score</th>
      <th>med_order</th>
      <th>med_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample1_dt</th>
      <td>16.0</td>
      <td>0.000000</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample2_dt</th>
      <td>1.0</td>
      <td>0.291926</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample3_dt</th>
      <td>17.0</td>
      <td>0.000000</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample4_dt</th>
      <td>17.0</td>
      <td>0.000000</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample5_dt</th>
      <td>16.0</td>
      <td>0.000000</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.513574</td>
    </tr>
    <tr>
      <th>sample6_dt</th>
      <td>17.0</td>
      <td>0.000000</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample7_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample8_dt</th>
      <td>1.0</td>
      <td>0.238084</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample9_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.488671</td>
    </tr>
    <tr>
      <th>sample10_dt</th>
      <td>1.0</td>
      <td>0.291926</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample11_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample12_dt</th>
      <td>17.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample13_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample14_dt</th>
      <td>17.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.513574</td>
    </tr>
    <tr>
      <th>sample15_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample16_dt</th>
      <td>17.0</td>
      <td>0.000000</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.513574</td>
    </tr>
    <tr>
      <th>sample17_dt</th>
      <td>26.0</td>
      <td>0.000000</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample18_dt</th>
      <td>1.0</td>
      <td>0.291926</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample19_dt</th>
      <td>19.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample20_dt</th>
      <td>19.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.488671</td>
    </tr>
    <tr>
      <th>sample21_dt</th>
      <td>1.0</td>
      <td>0.237266</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample22_dt</th>
      <td>19.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.488671</td>
    </tr>
    <tr>
      <th>sample23_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample24_dt</th>
      <td>19.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample25_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample26_dt</th>
      <td>18.0</td>
      <td>0.000000</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.513574</td>
    </tr>
    <tr>
      <th>sample27_dt</th>
      <td>1.0</td>
      <td>0.291926</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample28_dt</th>
      <td>27.0</td>
      <td>0.000000</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>20.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample29_dt</th>
      <td>1.0</td>
      <td>0.291926</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample30_dt</th>
      <td>5.0</td>
      <td>0.030632</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.390737</td>
    </tr>
    <tr>
      <th>sample31_dt</th>
      <td>19.0</td>
      <td>0.000000</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sample32_dt</th>
      <td>1.0</td>
      <td>0.236372</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Random Forest Classifier

#### before parameter tuning


```python
rf_class = RandomForestClassifier(random_state=42)
```


```python
temp_rf = plot_feature_importances(rf_class)
temp_rf
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>work_dummy</td>
      <td>0.211936</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ralation_dummy</td>
      <td>0.110997</td>
    </tr>
    <tr>
      <th>21</th>
      <td>workday_dummy</td>
      <td>0.090923</td>
    </tr>
    <tr>
      <th>16</th>
      <td>satisfaction_dummy</td>
      <td>0.073896</td>
    </tr>
    <tr>
      <th>2</th>
      <td>con_dummy</td>
      <td>0.062570</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dis_level</td>
      <td>0.057558</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age_dummy</td>
      <td>0.054212</td>
    </tr>
  </tbody>
</table>
</div>




![png](https://drive.google.com/uc?export=view&id=1U__JO0c-s7aIgrTrccQWHNlRjtcYVKUh)



```python
selected_variables_rf = sorted(list(set(common_sample.columns) - set(jobcondition_features)))
```


```python
sample1_rf = df[["js_dummy", "psy_dummy", "med_dummy", *selected_variables_rf]] # 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample2_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", *selected_variables_rf]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample3_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", *selected_variables_rf]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample4_rf = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", *selected_variables_rf]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample5_rf = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample6_rf = df[["js_dummy", "psy_dummy", "med_dummy", "income", *selected_variables_rf]] # 임금/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample7_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", *selected_variables_rf]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample8_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", *selected_variables_rf]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample9_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", *selected_variables_rf]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample10_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample11_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample12_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample13_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample14_rf = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample15_rf = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample16_rf = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample17_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", *selected_variables_rf]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample18_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", *selected_variables_rf]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample19_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "income", *selected_variables_rf]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample20_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample21_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample22_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample23_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample24_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample25_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample26_rf = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample27_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_rf]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample28_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_rf]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample29_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_rf]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample30_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample31_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample32_rf = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_rf]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스
```


```python
samples_rf = [sample1_rf, sample2_rf, sample3_rf, sample4_rf, sample5_rf, sample6_rf, sample7_rf, sample8_rf, sample9_rf, sample10_rf
             , sample11_rf, sample12_rf, sample13_rf, sample14_rf, sample15_rf, sample16_rf,sample17_rf, sample18_rf, sample19_rf, sample20_rf, 
              sample21_rf, sample22_rf, sample23_rf, sample24_rf, sample25_rf, sample26_rf
             , sample27_rf, sample28_rf, sample29_rf, sample30_rf, sample31_rf, sample32_rf]
features_rf=[]
X_rf = []
y_rf = []
X_train_rf = []
X_test_rf = []
y_train_rf = []
y_test_rf = []

for i,sample in enumerate(samples_rf):
    features_rf.append(sorted(list(set(sample.columns) - set(label_features))))
    X_rf.append(sample[features_rf[i]])
    y_rf.append(sample[label_features])
    X_train_rf.append(train_test_split(X_rf[i], y_rf[i], test_size=0.3, random_state=42)[0])
    X_test_rf.append(train_test_split(X_rf[i], y_rf[i], test_size=0.3, random_state=42)[1])
    y_train_rf.append(train_test_split(X_rf[i], y_rf[i], test_size=0.3, random_state=42)[2])
    y_test_rf.append(train_test_split(X_rf[i], y_rf[i], test_size=0.3, random_state=42)[3])
    X_train_rf[i] = X_train_rf[i].reset_index(drop=True)
    X_test_rf[i] = X_test_rf[i].reset_index(drop=True)                   
    y_train_rf[i] = y_train_rf[i].reset_index(drop=True)
    y_test_rf[i] = y_test_rf[i].reset_index(drop=True)
```


```python
for i in range(16):
    print('---- sample{}_rf ----'.format(i+1))
    accuracy = cross_val_score(rf_class, X_train_rf[i], y_train_rf[i], scoring='accuracy', cv = 10).mean() * 100
    print("Accuracy of Random Forests is: " , accuracy)
```

    ---- sample1_rf ----
    Accuracy of Random Forests is:  75.98484848484847
    ---- sample2_rf ----
    Accuracy of Random Forests is:  78.48484848484848
    ---- sample3_rf ----
    Accuracy of Random Forests is:  77.42424242424242
    ---- sample4_rf ----
    Accuracy of Random Forests is:  76.2121212121212
    ---- sample5_rf ----
    Accuracy of Random Forests is:  76.66666666666667
    ---- sample6_rf ----
    Accuracy of Random Forests is:  75.75757575757575
    ---- sample7_rf ----
    Accuracy of Random Forests is:  77.5
    ---- sample8_rf ----
    Accuracy of Random Forests is:  77.87878787878788
    ---- sample9_rf ----
    Accuracy of Random Forests is:  77.04545454545455
    ---- sample10_rf ----
    Accuracy of Random Forests is:  77.27272727272728
    ---- sample11_rf ----
    Accuracy of Random Forests is:  76.2878787878788
    ---- sample12_rf ----
    Accuracy of Random Forests is:  77.27272727272727
    ---- sample13_rf ----
    Accuracy of Random Forests is:  78.33333333333334
    ---- sample14_rf ----
    Accuracy of Random Forests is:  77.27272727272727
    ---- sample15_rf ----
    Accuracy of Random Forests is:  76.51515151515152
    ---- sample16_rf ----
    Accuracy of Random Forests is:  76.89393939393938
    

#### randomsearch parameter tuning


```python
rf_params = {'max_depth':np.arange(3, 30), 
            'n_estimators':np.arange(100, 400),
            'min_samples_split':np.arange(2, 10)}
params_list = [rf_params]

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=5, scoring='accuracy', random_state=42)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
best_param_dict_rf = dict()
samples_rf = ['sample1_rf', 'sample2_rf', 'sample3_rf', 'sample4_rf', 'sample5_rf', 'sample6_rf', 'sample7_rf', 'sample8_rf',
              'sample9_rf', 'sample10_rf', 'sample11_rf', 'sample12_rf', 'sample13_rf', 'sample14_rf', 'sample15_rf', 'sample16_rf',
             'sample17_rf', 'sample18_rf', 'sample19_rf', 'sample20_rf', 'sample21_rf', 'sample22_rf', 'sample23_rf', 'sample24_rf', 
              'sample25_rf', 'sample26_rf', 'sample27_rf', 'sample28_rf', 'sample29_rf', 'sample30_rf', 'sample31_rf', 'sample32_rf']

for i,sample in enumerate(samples_rf):
    print('---sample{}_rf---'.format(i+1))
    best_params = hypertuning_rscv(rf_class, rf_params, 30, X_train_rf[i], y_train_rf[i])
    best_param_dict_rf[sample] = best_params[0]
    print('best_params : ', best_params[0])
```

    ---sample1_rf---
    best_params :  {'n_estimators': 150, 'min_samples_split': 2, 'max_depth': 10}
    ---sample2_rf---
    best_params :  {'n_estimators': 133, 'min_samples_split': 2, 'max_depth': 4}
    ---sample3_rf---
    best_params :  {'n_estimators': 386, 'min_samples_split': 8, 'max_depth': 25}
    ---sample4_rf---
    best_params :  {'n_estimators': 268, 'min_samples_split': 4, 'max_depth': 10}
    ---sample5_rf---
    best_params :  {'n_estimators': 363, 'min_samples_split': 2, 'max_depth': 28}
    ---sample6_rf---
    best_params :  {'n_estimators': 158, 'min_samples_split': 9, 'max_depth': 18}
    ---sample7_rf---
    best_params :  {'n_estimators': 223, 'min_samples_split': 7, 'max_depth': 9}
    ---sample8_rf---
    best_params :  {'n_estimators': 311, 'min_samples_split': 3, 'max_depth': 5}
    ---sample9_rf---
    best_params :  {'n_estimators': 131, 'min_samples_split': 5, 'max_depth': 21}
    ---sample10_rf---
    best_params :  {'n_estimators': 284, 'min_samples_split': 7, 'max_depth': 7}
    ---sample11_rf---
    best_params :  {'n_estimators': 143, 'min_samples_split': 7, 'max_depth': 25}
    ---sample12_rf---
    best_params :  {'n_estimators': 394, 'min_samples_split': 5, 'max_depth': 18}
    ---sample13_rf---
    best_params :  {'n_estimators': 288, 'min_samples_split': 4, 'max_depth': 28}
    ---sample14_rf---
    best_params :  {'n_estimators': 131, 'min_samples_split': 5, 'max_depth': 21}
    ---sample15_rf---
    best_params :  {'n_estimators': 191, 'min_samples_split': 7, 'max_depth': 22}
    ---sample16_rf---
    best_params :  {'n_estimators': 158, 'min_samples_split': 9, 'max_depth': 18}
    ---sample17_rf---
    best_params :  {'n_estimators': 284, 'min_samples_split': 7, 'max_depth': 7}
    ---sample18_rf---
    best_params :  {'n_estimators': 284, 'min_samples_split': 7, 'max_depth': 7}
    ---sample19_rf---
    best_params :  {'n_estimators': 295, 'min_samples_split': 6, 'max_depth': 9}
    ---sample20_rf---
    best_params :  {'n_estimators': 162, 'min_samples_split': 3, 'max_depth': 12}
    ---sample21_rf---
    best_params :  {'n_estimators': 191, 'min_samples_split': 7, 'max_depth': 22}
    ---sample22_rf---
    best_params :  {'n_estimators': 284, 'min_samples_split': 7, 'max_depth': 7}
    ---sample23_rf---
    best_params :  {'n_estimators': 394, 'min_samples_split': 5, 'max_depth': 18}
    ---sample24_rf---
    best_params :  {'n_estimators': 363, 'min_samples_split': 2, 'max_depth': 28}
    ---sample25_rf---
    best_params :  {'n_estimators': 119, 'min_samples_split': 8, 'max_depth': 18}
    ---sample26_rf---
    best_params :  {'n_estimators': 143, 'min_samples_split': 7, 'max_depth': 25}
    ---sample27_rf---
    best_params :  {'n_estimators': 365, 'min_samples_split': 6, 'max_depth': 5}
    ---sample28_rf---
    best_params :  {'n_estimators': 122, 'min_samples_split': 6, 'max_depth': 26}
    ---sample29_rf---
    best_params :  {'n_estimators': 122, 'min_samples_split': 6, 'max_depth': 26}
    ---sample30_rf---
    best_params :  {'n_estimators': 143, 'min_samples_split': 7, 'max_depth': 25}
    ---sample31_rf---
    best_params :  {'n_estimators': 150, 'min_samples_split': 2, 'max_depth': 10}
    ---sample32_rf---
    best_params :  {'n_estimators': 284, 'min_samples_split': 7, 'max_depth': 7}
    

#### score test


```python
samples_rf = ['sample1_rf', 'sample2_rf', 'sample3_rf', 'sample4_rf', 'sample5_rf', 'sample6_rf', 'sample7_rf', 'sample8_rf',
              'sample9_rf', 'sample10_rf', 'sample11_rf', 'sample12_rf', 'sample13_rf', 'sample14_rf', 'sample15_rf', 'sample16_rf',
             'sample17_rf', 'sample18_rf', 'sample19_rf', 'sample20_rf', 'sample21_rf', 'sample22_rf', 'sample23_rf', 'sample24_rf', 
              'sample25_rf', 'sample26_rf', 'sample27_rf', 'sample28_rf', 'sample29_rf', 'sample30_rf', 'sample31_rf', 'sample32_rf']

accuracy = []
precision = []
sensitivity = []
Auc = []
for i,sample in enumerate(samples_rf):
    print('--------------sample{}_rf--------------'.format(i+1))
    clf = RandomForestClassifier(random_state=42, **best_param_dict_rf[sample])
    clf.fit(X_train_rf[i], y_train_rf[i])
    y_pred_rf = clf.predict(X_test_rf[i])
    y_pred_proba_rf = clf.predict_proba(X_test_rf[i])
    
    print("accuracy_score: {}".format( cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='accuracy', cv = 5).mean() * 100))
    accuracy.append(cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='accuracy', cv = 5).mean() * 100)
    print("precision_score: {}".format( cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='precision', cv = 5).mean() * 100))
    precision.append(cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='precision', cv = 5).mean() * 100)
    print("sensitivity_score: {}".format( cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='recall', cv = 5).mean() * 100))
    sensitivity.append(cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='recall', cv = 5).mean() * 100)
    try:
        print("AUC: Area Under Curve: {}".format(cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='roc_auc', cv = 5).mean() * 100))
        Auc.append(cross_val_score(rf_class, y_test_rf[i], y_pred_rf, scoring='roc_auc', cv = 5).mean() * 100)
    except ValueError:
        pass
    
score_rf = pd.DataFrame({'accuracy':accuracy, 'precision':precision, 'sensitivity': sensitivity, 'Auc': Auc})
    
    #y_score = clf.decision_path(X_test[i])
    #precision, recall, _ = precision_recall_curve(y_test[i], y_score[1])
    #fpr, tpr, _ = roc_curve(y_test[i], y_score)
    
    #f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.set_size_inches((8, 4)) 
    #axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #axes[0].set_title('Recall-Precision Curve')
    
    #axes[1].plot(fpr, tpr)
    #axes[1].plot([0, 1], [0, 1], linestyle='--')
    #axes[1].set_title('ROC curve')

```

    --------------sample1_rf--------------
    accuracy_score: 78.82937432075764
    precision_score: 70.83528250549527
    sensitivity_score: 69.65853658536585
    AUC: Area Under Curve: 76.767439340443
    --------------sample2_rf--------------
    accuracy_score: 80.59462816332868
    precision_score: 70.06636393478499
    sensitivity_score: 73.5135135135135
    AUC: Area Under Curve: 78.81816026552869
    --------------sample3_rf--------------
    accuracy_score: 79.1802515137401
    precision_score: 68.00759584634164
    sensitivity_score: 71.39402560455193
    AUC: Area Under Curve: 77.22508297771455
    --------------sample4_rf--------------
    accuracy_score: 80.40987424312995
    precision_score: 70.21022025369852
    sensitivity_score: 73.15789473684211
    AUC: Area Under Curve: 78.61052631578947
    --------------sample5_rf--------------
    accuracy_score: 80.4145319049837
    precision_score: 71.97382713661781
    sensitivity_score: 72.44871794871794
    AUC: Area Under Curve: 78.5396742896743
    --------------sample6_rf--------------
    accuracy_score: 80.23598820058997
    precision_score: 71.8002223002223
    sensitivity_score: 72.06410256410255
    AUC: Area Under Curve: 78.32934857934858
    --------------sample7_rf--------------
    accuracy_score: 78.8309268747089
    precision_score: 70.50930000586752
    sensitivity_score: 69.64634146341463
    AUC: Area Under Curve: 76.7613417794674
    --------------sample8_rf--------------
    accuracy_score: 80.58997050147492
    precision_score: 70.56034735668601
    sensitivity_score: 73.27935222672065
    AUC: Area Under Curve: 78.7923076923077
    --------------sample9_rf--------------
    accuracy_score: 79.88976866946126
    precision_score: 71.21130331656647
    sensitivity_score: 71.57692307692308
    AUC: Area Under Curve: 77.9506237006237
    --------------sample10_rf--------------
    accuracy_score: 80.24064586244371
    precision_score: 71.63010127644273
    sensitivity_score: 72.07692307692308
    AUC: Area Under Curve: 78.33575883575882
    --------------sample11_rf--------------
    accuracy_score: 78.29529576152773
    precision_score: 67.5188859065624
    sensitivity_score: 69.75708502024293
    AUC: Area Under Curve: 76.21187584345479
    --------------sample12_rf--------------
    accuracy_score: 80.24064586244373
    precision_score: 70.07936507936508
    sensitivity_score: 72.75303643724695
    AUC: Area Under Curve: 78.39406207827263
    --------------sample13_rf--------------
    accuracy_score: 78.12606738084148
    precision_score: 67.84287338408343
    sensitivity_score: 69.23076923076923
    AUC: Area Under Curve: 76.0081774081774
    --------------sample14_rf--------------
    accuracy_score: 79.52802359882006
    precision_score: 70.77987320008596
    sensitivity_score: 71.03846153846153
    AUC: Area Under Curve: 77.5462577962578
    --------------sample15_rf--------------
    accuracy_score: 80.59152305542618
    precision_score: 70.42655699177438
    sensitivity_score: 73.29284750337382
    AUC: Area Under Curve: 78.79730094466937
    --------------sample16_rf--------------
    accuracy_score: 80.24219841639498
    precision_score: 70.97342785661144
    sensitivity_score: 72.30769230769229
    AUC: Area Under Curve: 78.35204435204435
    --------------sample17_rf--------------
    accuracy_score: 79.36034777208508
    precision_score: 72.47032455260059
    sensitivity_score: 69.89547038327527
    AUC: Area Under Curve: 77.327872177939
    --------------sample18_rf--------------
    accuracy_score: 79.71588262692129
    precision_score: 72.05610318202491
    sensitivity_score: 70.80487804878048
    AUC: Area Under Curve: 77.73120614767792
    --------------sample19_rf--------------
    accuracy_score: 78.83403198261138
    precision_score: 71.10133966080832
    sensitivity_score: 69.45121951219512
    AUC: Area Under Curve: 76.75186546386013
    --------------sample20_rf--------------
    accuracy_score: 79.88355845365625
    precision_score: 71.69120948190717
    sensitivity_score: 71.34615384615385
    AUC: Area Under Curve: 77.92594623074075
    --------------sample21_rf--------------
    accuracy_score: 79.71122496506753
    precision_score: 71.14957264957265
    sensitivity_score: 71.2051282051282
    AUC: Area Under Curve: 77.73510760497062
    --------------sample22_rf--------------
    accuracy_score: 80.77006675981991
    precision_score: 72.07519720677615
    sensitivity_score: 72.96153846153845
    AUC: Area Under Curve: 78.93302148302148
    --------------sample23_rf--------------
    accuracy_score: 79.70967241111629
    precision_score: 69.36141603449116
    sensitivity_score: 72.10526315789474
    AUC: Area Under Curve: 77.82105263157895
    --------------sample24_rf--------------
    accuracy_score: 79.53578636857631
    precision_score: 69.03076048118886
    sensitivity_score: 71.72739541160593
    AUC: Area Under Curve: 77.61632928475034
    --------------sample25_rf--------------
    accuracy_score: 78.8340319826114
    precision_score: 68.41234291402264
    sensitivity_score: 70.41835357624831
    AUC: Area Under Curve: 76.78214976109713
    --------------sample26_rf--------------
    accuracy_score: 80.23909330849246
    precision_score: 70.50203869819853
    sensitivity_score: 72.48313090418354
    AUC: Area Under Curve: 78.34787175839807
    --------------sample27_rf--------------
    accuracy_score: 79.70967241111629
    precision_score: 70.05354342196448
    sensitivity_score: 71.6059379217274
    AUC: Area Under Curve: 77.75792391581867
    --------------sample28_rf--------------
    accuracy_score: 79.36034777208508
    precision_score: 71.57336357336358
    sensitivity_score: 70.30487804878048
    AUC: Area Under Curve: 77.34421984630805
    --------------sample29_rf--------------
    accuracy_score: 80.24219841639496
    precision_score: 71.04093567251462
    sensitivity_score: 72.3076923076923
    AUC: Area Under Curve: 78.35204435204435
    --------------sample30_rf--------------
    accuracy_score: 81.29793510324484
    precision_score: 72.74149079534637
    sensitivity_score: 73.84615384615385
    AUC: Area Under Curve: 79.52307692307691
    --------------sample31_rf--------------
    accuracy_score: 79.18490917559384
    precision_score: 67.99071716092993
    sensitivity_score: 71.40825035561879
    AUC: Area Under Curve: 77.23745851114272
    --------------sample32_rf--------------
    accuracy_score: 81.12715416860736
    precision_score: 73.05111448243669
    sensitivity_score: 73.23076923076923
    AUC: Area Under Curve: 79.29402215703585
    


```python
score_rf
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
      <th>accuracy</th>
      <th>precision</th>
      <th>sensitivity</th>
      <th>Auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>6</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>9</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>10</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>11</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>12</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>13</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>14</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>15</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>16</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>17</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>18</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>19</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>20</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>21</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>22</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>23</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>24</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>25</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>26</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>27</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>28</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>29</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>30</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
    <tr>
      <th>31</th>
      <td>77.590436</td>
      <td>62.73919</td>
      <td>70.428571</td>
      <td>75.616752</td>
    </tr>
  </tbody>
</table>
</div>




```python
samples_rf = ['sample1_rf', 'sample2_rf', 'sample3_rf', 'sample4_rf', 'sample5_rf', 'sample6_rf', 'sample7_rf', 'sample8_rf',
              'sample9_rf', 'sample10_rf', 'sample11_rf', 'sample12_rf', 'sample13_rf', 'sample14_rf', 'sample15_rf', 'sample16_rf',
             'sample17_rf', 'sample18_rf', 'sample19_rf', 'sample20_rf', 'sample21_rf', 'sample22_rf', 'sample23_rf', 'sample24_rf', 
              'sample25_rf', 'sample26_rf', 'sample27_rf', 'sample28_rf', 'sample29_rf', 'sample30_rf', 'sample31_rf', 'sample32_rf']

samples_rf1 = [sample1_rf, sample2_rf, sample3_rf, sample4_rf, sample5_rf, sample6_rf, sample7_rf, sample8_rf, sample9_rf, sample10_rf
             , sample11_rf, sample12_rf, sample13_rf, sample14_rf, sample15_rf, sample16_rf,sample17_rf, sample18_rf, sample19_rf, sample20_rf, 
              sample21_rf, sample22_rf, sample23_rf, sample24_rf, sample25_rf, sample26_rf
             , sample27_rf, sample28_rf, sample29_rf, sample30_rf, sample31_rf, sample32_rf]

feature_order = []
psyservice_order = []
psyservice_score = []
jobservice_order = []
jobservice_score = []
medservice_order = []
medservice_score = []

for i,(sample,sample1) in enumerate(zip(samples_rf, samples_rf1)):
    print('--------------sample{}_rf--------------'.format(i+1))
    clf = RandomForestClassifier(random_state=42, **best_param_dict_rf[sample])
    clf.fit(X_train_rf[i], y_train_rf[i])
    
    crucial_features = pd.DataFrame({'feature':list(set(sample1.columns) - set(label_features)), 'importance':clf.feature_importances_})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features.reset_index(drop = True, inplace=True)
    
    feature = pd.DataFrame({sample: crucial_features['feature']})
    feature_order.append(feature)
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')])
    psyservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].index.to_list())
    psyservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')])
    jobservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].index.to_list())
    jobservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')])
    medservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].index.to_list())
    medservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].to_list())

feature_order_rf = pd.concat(feature_order, axis =1 )
service_rf = pd.DataFrame({'psy_order': psyservice_order, 'psy_score': psyservice_score, 'job_order': jobservice_order,
                          'job_score': jobservice_score, 'med_order': medservice_order, 'med_score': medservice_score}, index = samples_rf)
for col in service_rf.columns:
       service_rf[col] = service_rf[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
```

    --------------sample1_rf--------------
    12    0.020421
    Name: importance, dtype: float64
    22    0.009737
    Name: importance, dtype: float64
    15    0.017367
    Name: importance, dtype: float64
    --------------sample2_rf--------------
    2    0.192216
    Name: importance, dtype: float64
    23    0.002142
    Name: importance, dtype: float64
    19    0.002904
    Name: importance, dtype: float64
    --------------sample3_rf--------------
    8    0.028394
    Name: importance, dtype: float64
    23    0.008402
    Name: importance, dtype: float64
    18    0.015058
    Name: importance, dtype: float64
    --------------sample4_rf--------------
    12    0.018416
    Name: importance, dtype: float64
    23    0.008576
    Name: importance, dtype: float64
    16    0.014789
    Name: importance, dtype: float64
    --------------sample5_rf--------------
    21    0.012598
    Name: importance, dtype: float64
    23    0.00853
    Name: importance, dtype: float64
    1    0.107558
    Name: importance, dtype: float64
    --------------sample6_rf--------------
    2    0.085716
    Name: importance, dtype: float64
    23    0.006864
    Name: importance, dtype: float64
    16    0.014307
    Name: importance, dtype: float64
    --------------sample7_rf--------------
    9    0.024432
    Name: importance, dtype: float64
    24    0.006708
    Name: importance, dtype: float64
    18    0.011835
    Name: importance, dtype: float64
    --------------sample8_rf--------------
    2    0.187477
    Name: importance, dtype: float64
    24    0.003049
    Name: importance, dtype: float64
    20    0.004835
    Name: importance, dtype: float64
    --------------sample9_rf--------------
    14    0.018274
    Name: importance, dtype: float64
    24    0.007694
    Name: importance, dtype: float64
    1    0.119611
    Name: importance, dtype: float64
    --------------sample10_rf--------------
    2    0.150868
    Name: importance, dtype: float64
    23    0.005601
    Name: importance, dtype: float64
    20    0.00877
    Name: importance, dtype: float64
    --------------sample11_rf--------------
    9    0.026749
    Name: importance, dtype: float64
    24    0.007568
    Name: importance, dtype: float64
    18    0.014657
    Name: importance, dtype: float64
    --------------sample12_rf--------------
    15    0.018578
    Name: importance, dtype: float64
    24    0.008125
    Name: importance, dtype: float64
    17    0.01647
    Name: importance, dtype: float64
    --------------sample13_rf--------------
    9    0.027173
    Name: importance, dtype: float64
    24    0.007971
    Name: importance, dtype: float64
    18    0.01546
    Name: importance, dtype: float64
    --------------sample14_rf--------------
    20    0.012279
    Name: importance, dtype: float64
    24    0.008305
    Name: importance, dtype: float64
    1    0.120573
    Name: importance, dtype: float64
    --------------sample15_rf--------------
    2    0.081037
    Name: importance, dtype: float64
    24    0.006605
    Name: importance, dtype: float64
    16    0.013564
    Name: importance, dtype: float64
    --------------sample16_rf--------------
    14    0.014493
    Name: importance, dtype: float64
    24    0.007177
    Name: importance, dtype: float64
    1    0.143488
    Name: importance, dtype: float64
    --------------sample17_rf--------------
    10    0.018224
    Name: importance, dtype: float64
    25    0.004835
    Name: importance, dtype: float64
    20    0.00867
    Name: importance, dtype: float64
    --------------sample18_rf--------------
    2    0.138856
    Name: importance, dtype: float64
    25    0.005364
    Name: importance, dtype: float64
    19    0.009256
    Name: importance, dtype: float64
    --------------sample19_rf--------------
    10    0.020841
    Name: importance, dtype: float64
    25    0.006364
    Name: importance, dtype: float64
    18    0.011233
    Name: importance, dtype: float64
    --------------sample20_rf--------------
    14    0.017473
    Name: importance, dtype: float64
    25    0.007577
    Name: importance, dtype: float64
    1    0.110139
    Name: importance, dtype: float64
    --------------sample21_rf--------------
    2    0.112898
    Name: importance, dtype: float64
    25    0.006323
    Name: importance, dtype: float64
    18    0.012085
    Name: importance, dtype: float64
    --------------sample22_rf--------------
    4    0.05649
    Name: importance, dtype: float64
    26    0.004508
    Name: importance, dtype: float64
    1    0.169895
    Name: importance, dtype: float64
    --------------sample23_rf--------------
    14    0.017928
    Name: importance, dtype: float64
    25    0.007726
    Name: importance, dtype: float64
    17    0.015251
    Name: importance, dtype: float64
    --------------sample24_rf--------------
    11    0.026398
    Name: importance, dtype: float64
    25    0.007759
    Name: importance, dtype: float64
    19    0.015047
    Name: importance, dtype: float64
    --------------sample25_rf--------------
    2    0.076402
    Name: importance, dtype: float64
    25    0.00664
    Name: importance, dtype: float64
    20    0.011993
    Name: importance, dtype: float64
    --------------sample26_rf--------------
    16    0.014332
    Name: importance, dtype: float64
    25    0.006009
    Name: importance, dtype: float64
    1    0.131176
    Name: importance, dtype: float64
    --------------sample27_rf--------------
    2    0.164118
    Name: importance, dtype: float64
    25    0.003058
    Name: importance, dtype: float64
    22    0.004066
    Name: importance, dtype: float64
    --------------sample28_rf--------------
    13    0.021002
    Name: importance, dtype: float64
    26    0.006785
    Name: importance, dtype: float64
    20    0.011925
    Name: importance, dtype: float64
    --------------sample29_rf--------------
    2    0.101336
    Name: importance, dtype: float64
    26    0.006727
    Name: importance, dtype: float64
    19    0.011996
    Name: importance, dtype: float64
    --------------sample30_rf--------------
    3    0.074182
    Name: importance, dtype: float64
    26    0.00614
    Name: importance, dtype: float64
    1    0.126279
    Name: importance, dtype: float64
    --------------sample31_rf--------------
    3    0.071256
    Name: importance, dtype: float64
    26    0.006363
    Name: importance, dtype: float64
    18    0.01323
    Name: importance, dtype: float64
    --------------sample32_rf--------------
    2    0.130561
    Name: importance, dtype: float64
    26    0.004942
    Name: importance, dtype: float64
    21    0.00834
    Name: importance, dtype: float64
    


```python
feature_order_rf
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
      <th>sample1_rf</th>
      <th>sample2_rf</th>
      <th>sample3_rf</th>
      <th>sample4_rf</th>
      <th>sample5_rf</th>
      <th>sample6_rf</th>
      <th>sample7_rf</th>
      <th>sample8_rf</th>
      <th>sample9_rf</th>
      <th>sample10_rf</th>
      <th>...</th>
      <th>sample23_rf</th>
      <th>sample24_rf</th>
      <th>sample25_rf</th>
      <th>sample26_rf</th>
      <th>sample27_rf</th>
      <th>sample28_rf</th>
      <th>sample29_rf</th>
      <th>sample30_rf</th>
      <th>sample31_rf</th>
      <th>sample32_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>laborcontract_dummy</td>
      <td>work_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>...</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>work_dummy</td>
      <td>laborcontract_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>...</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>work_dummy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>labortime_dummy</td>
      <td>psy_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>...</td>
      <td>labortime_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>labortime_dummy</td>
      <td>psy_dummy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spouse_dummy</td>
      <td>labortime_dummy</td>
      <td>spouse_dummy</td>
      <td>labor_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>...</td>
      <td>labor_dummy</td>
      <td>spouse_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>labortime_dummy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>income</td>
      <td>...</td>
      <td>laborcontract_dummy</td>
      <td>age_dummy</td>
      <td>laborcontract_dummy</td>
      <td>sex_dummy</td>
      <td>doctorex_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labor_dummy</td>
      <td>income</td>
    </tr>
    <tr>
      <th>5</th>
      <td>recoveryex_dummy</td>
      <td>workday_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>...</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>doctorex_dummy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>...</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>alcohol_dummy</td>
      <td>spouse_dummy</td>
      <td>labor_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>workday_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workday_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workday_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>...</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>con_dummy</td>
      <td>sex_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workday_dummy</td>
      <td>...</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>workhour_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
    </tr>
    <tr>
      <th>9</th>
      <td>edu_dummy</td>
      <td>recoveryex_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>cureperiod_dummy</td>
      <td>con_dummy</td>
      <td>psy_dummy</td>
      <td>recoveryex_dummy</td>
      <td>workhour_dummy</td>
      <td>satisfaction_dummy</td>
      <td>...</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workday_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>dis_level</td>
      <td>smoke_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>con_dummy</td>
      <td>dis_level</td>
      <td>con_dummy</td>
      <td>ralation_dummy</td>
      <td>con_dummy</td>
      <td>dis_level</td>
      <td>...</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>edu_dummy</td>
      <td>dis_level</td>
      <td>smoke_dummy</td>
      <td>cureperiod_dummy</td>
      <td>con_dummy</td>
      <td>...</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>dis_level</td>
      <td>con_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_level</td>
      <td>con_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>satisfaction_dummy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>edu_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>dis_level</td>
      <td>edu_dummy</td>
      <td>...</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>cureperiod_dummy</td>
      <td>smoke_dummy</td>
      <td>con_dummy</td>
      <td>dis_level</td>
      <td>con_dummy</td>
      <td>dis_level</td>
      <td>workday_dummy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>doctorex_dummy</td>
      <td>income</td>
      <td>ralation_dummy</td>
      <td>dis_level</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>...</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>cureperiod_dummy</td>
      <td>dis_level</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_level</td>
      <td>cureperiod_dummy</td>
      <td>dis_level</td>
    </tr>
    <tr>
      <th>14</th>
      <td>jobdoctorex_dummy</td>
      <td>dis_level</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>edu_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>edu_dummy</td>
      <td>psy_dummy</td>
      <td>smoke_dummy</td>
      <td>...</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>income</td>
      <td>ralation_dummy</td>
      <td>dis_level</td>
      <td>edu_dummy</td>
      <td>workday_dummy</td>
      <td>cureperiod_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>med_dummy</td>
      <td>con_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>workday_dummy</td>
      <td>doctorex_dummy</td>
      <td>sex_dummy</td>
      <td>con_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>...</td>
      <td>ralation_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>alcohol_dummy</td>
      <td>ralation_dummy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>doctorex_dummy</td>
      <td>edu_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>sex_dummy</td>
      <td>edu_dummy</td>
      <td>sex_dummy</td>
      <td>...</td>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
      <td>psy_dummy</td>
      <td>con_dummy</td>
      <td>ralation_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>ralation_dummy</td>
      <td>cureperiod_dummy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>laborchange_dummy</td>
      <td>emplev_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>work_dummy</td>
      <td>emplev_dummy</td>
      <td>emplev_dummy</td>
      <td>alcohol_dummy</td>
      <td>...</td>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>laborchange_dummy</td>
      <td>ralation_dummy</td>
      <td>income</td>
      <td>income</td>
      <td>smoke_dummy</td>
    </tr>
    <tr>
      <th>18</th>
      <td>jobreturnopinion_dummy</td>
      <td>alcohol_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>sex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>med_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>emplev_dummy</td>
      <td>...</td>
      <td>edu_dummy</td>
      <td>work_dummy</td>
      <td>doctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
    </tr>
    <tr>
      <th>19</th>
      <td>age_dummy</td>
      <td>med_dummy</td>
      <td>sex_dummy</td>
      <td>doctorex_dummy</td>
      <td>work_dummy</td>
      <td>laborchange_dummy</td>
      <td>doctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>...</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>laborchange_dummy</td>
      <td>alcohol_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>emplev_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
    </tr>
    <tr>
      <th>20</th>
      <td>cureperiod_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>...</td>
      <td>jobreturnopinion_dummy</td>
      <td>labortime_dummy</td>
      <td>med_dummy</td>
      <td>labor_dummy</td>
      <td>cureperiod_dummy</td>
      <td>med_dummy</td>
      <td>emplev_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>sex_dummy</td>
    </tr>
    <tr>
      <th>21</th>
      <td>sex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>cureperiod_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>doctorex_dummy</td>
      <td>age_dummy</td>
      <td>...</td>
      <td>doctorex_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>income</td>
      <td>emplev_dummy</td>
      <td>smoke_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>med_dummy</td>
    </tr>
    <tr>
      <th>22</th>
      <td>js_dummy</td>
      <td>dis_dummy</td>
      <td>doctorex_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>sex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>labortime_dummy</td>
      <td>cureperiod_dummy</td>
      <td>...</td>
      <td>emplev_cont</td>
      <td>sex_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>labor_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>doctorex_dummy</td>
      <td>alcohol_dummy</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>...</td>
      <td>age_dummy</td>
      <td>labor_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
      <td>alcohol_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>emplev_dummy</td>
    </tr>
    <tr>
      <th>24</th>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>doctorex_dummy</td>
      <td>...</td>
      <td>sex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>age_dummy</td>
      <td>doctorex_dummy</td>
      <td>emplev_cont</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>emplev_cont</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>doctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>...</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>age_dummy</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>emplev_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>...</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>labor_dummy</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>laborchange_dummy</td>
      <td>emplev_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_dummy</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>laborchange_dummy</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 32 columns</p>
</div>




```python
service_rf
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
      <th>psy_order</th>
      <th>psy_score</th>
      <th>job_order</th>
      <th>job_score</th>
      <th>med_order</th>
      <th>med_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample1_rf</th>
      <td>12.0</td>
      <td>0.020421</td>
      <td>22.0</td>
      <td>0.009737</td>
      <td>15.0</td>
      <td>0.017367</td>
    </tr>
    <tr>
      <th>sample2_rf</th>
      <td>2.0</td>
      <td>0.192216</td>
      <td>23.0</td>
      <td>0.002142</td>
      <td>19.0</td>
      <td>0.002904</td>
    </tr>
    <tr>
      <th>sample3_rf</th>
      <td>8.0</td>
      <td>0.028394</td>
      <td>23.0</td>
      <td>0.008402</td>
      <td>18.0</td>
      <td>0.015058</td>
    </tr>
    <tr>
      <th>sample4_rf</th>
      <td>12.0</td>
      <td>0.018416</td>
      <td>23.0</td>
      <td>0.008576</td>
      <td>16.0</td>
      <td>0.014789</td>
    </tr>
    <tr>
      <th>sample5_rf</th>
      <td>21.0</td>
      <td>0.012598</td>
      <td>23.0</td>
      <td>0.008530</td>
      <td>1.0</td>
      <td>0.107558</td>
    </tr>
    <tr>
      <th>sample6_rf</th>
      <td>2.0</td>
      <td>0.085716</td>
      <td>23.0</td>
      <td>0.006864</td>
      <td>16.0</td>
      <td>0.014307</td>
    </tr>
    <tr>
      <th>sample7_rf</th>
      <td>9.0</td>
      <td>0.024432</td>
      <td>24.0</td>
      <td>0.006708</td>
      <td>18.0</td>
      <td>0.011835</td>
    </tr>
    <tr>
      <th>sample8_rf</th>
      <td>2.0</td>
      <td>0.187477</td>
      <td>24.0</td>
      <td>0.003049</td>
      <td>20.0</td>
      <td>0.004835</td>
    </tr>
    <tr>
      <th>sample9_rf</th>
      <td>14.0</td>
      <td>0.018274</td>
      <td>24.0</td>
      <td>0.007694</td>
      <td>1.0</td>
      <td>0.119611</td>
    </tr>
    <tr>
      <th>sample10_rf</th>
      <td>2.0</td>
      <td>0.150868</td>
      <td>23.0</td>
      <td>0.005601</td>
      <td>20.0</td>
      <td>0.008770</td>
    </tr>
    <tr>
      <th>sample11_rf</th>
      <td>9.0</td>
      <td>0.026749</td>
      <td>24.0</td>
      <td>0.007568</td>
      <td>18.0</td>
      <td>0.014657</td>
    </tr>
    <tr>
      <th>sample12_rf</th>
      <td>15.0</td>
      <td>0.018578</td>
      <td>24.0</td>
      <td>0.008125</td>
      <td>17.0</td>
      <td>0.016470</td>
    </tr>
    <tr>
      <th>sample13_rf</th>
      <td>9.0</td>
      <td>0.027173</td>
      <td>24.0</td>
      <td>0.007971</td>
      <td>18.0</td>
      <td>0.015460</td>
    </tr>
    <tr>
      <th>sample14_rf</th>
      <td>20.0</td>
      <td>0.012279</td>
      <td>24.0</td>
      <td>0.008305</td>
      <td>1.0</td>
      <td>0.120573</td>
    </tr>
    <tr>
      <th>sample15_rf</th>
      <td>2.0</td>
      <td>0.081037</td>
      <td>24.0</td>
      <td>0.006605</td>
      <td>16.0</td>
      <td>0.013564</td>
    </tr>
    <tr>
      <th>sample16_rf</th>
      <td>14.0</td>
      <td>0.014493</td>
      <td>24.0</td>
      <td>0.007177</td>
      <td>1.0</td>
      <td>0.143488</td>
    </tr>
    <tr>
      <th>sample17_rf</th>
      <td>10.0</td>
      <td>0.018224</td>
      <td>25.0</td>
      <td>0.004835</td>
      <td>20.0</td>
      <td>0.008670</td>
    </tr>
    <tr>
      <th>sample18_rf</th>
      <td>2.0</td>
      <td>0.138856</td>
      <td>25.0</td>
      <td>0.005364</td>
      <td>19.0</td>
      <td>0.009256</td>
    </tr>
    <tr>
      <th>sample19_rf</th>
      <td>10.0</td>
      <td>0.020841</td>
      <td>25.0</td>
      <td>0.006364</td>
      <td>18.0</td>
      <td>0.011233</td>
    </tr>
    <tr>
      <th>sample20_rf</th>
      <td>14.0</td>
      <td>0.017473</td>
      <td>25.0</td>
      <td>0.007577</td>
      <td>1.0</td>
      <td>0.110139</td>
    </tr>
    <tr>
      <th>sample21_rf</th>
      <td>2.0</td>
      <td>0.112898</td>
      <td>25.0</td>
      <td>0.006323</td>
      <td>18.0</td>
      <td>0.012085</td>
    </tr>
    <tr>
      <th>sample22_rf</th>
      <td>4.0</td>
      <td>0.056490</td>
      <td>26.0</td>
      <td>0.004508</td>
      <td>1.0</td>
      <td>0.169895</td>
    </tr>
    <tr>
      <th>sample23_rf</th>
      <td>14.0</td>
      <td>0.017928</td>
      <td>25.0</td>
      <td>0.007726</td>
      <td>17.0</td>
      <td>0.015251</td>
    </tr>
    <tr>
      <th>sample24_rf</th>
      <td>11.0</td>
      <td>0.026398</td>
      <td>25.0</td>
      <td>0.007759</td>
      <td>19.0</td>
      <td>0.015047</td>
    </tr>
    <tr>
      <th>sample25_rf</th>
      <td>2.0</td>
      <td>0.076402</td>
      <td>25.0</td>
      <td>0.006640</td>
      <td>20.0</td>
      <td>0.011993</td>
    </tr>
    <tr>
      <th>sample26_rf</th>
      <td>16.0</td>
      <td>0.014332</td>
      <td>25.0</td>
      <td>0.006009</td>
      <td>1.0</td>
      <td>0.131176</td>
    </tr>
    <tr>
      <th>sample27_rf</th>
      <td>2.0</td>
      <td>0.164118</td>
      <td>25.0</td>
      <td>0.003058</td>
      <td>22.0</td>
      <td>0.004066</td>
    </tr>
    <tr>
      <th>sample28_rf</th>
      <td>13.0</td>
      <td>0.021002</td>
      <td>26.0</td>
      <td>0.006785</td>
      <td>20.0</td>
      <td>0.011925</td>
    </tr>
    <tr>
      <th>sample29_rf</th>
      <td>2.0</td>
      <td>0.101336</td>
      <td>26.0</td>
      <td>0.006727</td>
      <td>19.0</td>
      <td>0.011996</td>
    </tr>
    <tr>
      <th>sample30_rf</th>
      <td>3.0</td>
      <td>0.074182</td>
      <td>26.0</td>
      <td>0.006140</td>
      <td>1.0</td>
      <td>0.126279</td>
    </tr>
    <tr>
      <th>sample31_rf</th>
      <td>3.0</td>
      <td>0.071256</td>
      <td>26.0</td>
      <td>0.006363</td>
      <td>18.0</td>
      <td>0.013230</td>
    </tr>
    <tr>
      <th>sample32_rf</th>
      <td>2.0</td>
      <td>0.130561</td>
      <td>26.0</td>
      <td>0.004942</td>
      <td>21.0</td>
      <td>0.008340</td>
    </tr>
  </tbody>
</table>
</div>



### Support Vector Machine

#### before parameter tuning


```python
sv_class = LinearSVC(random_state=42)
```


```python
def plot_feature_importances_sv(model, top_features=20):
    n_features = cx.columns
    sv_class.fit(cx_train, cy_train)
    #plt.barh(n_features, model.coef_.ravel(), align='center')
    #plt.yticks(np.arange(len(n_features)))
    #plt.xlabel("특성 중요도")
    #plt.ylabel("특성")
    
    coef = sv_class.coef_.ravel()
    top_coefficients = np.argsort(coef)

    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(len(coef)), coef[top_coefficients], color=colors)
    feature_names = np.array(n_features)
    plt.xticks(np.arange(len(coef)), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()
    
    crucial_features = pd.DataFrame({'feature':n_features, 'importance':model.coef_.ravel()})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features = crucial_features[crucial_features['importance']>0]
    return crucial_features
```


```python
temp_sv = plot_feature_importances_sv(sv_class)
temp_sv
```


![png](https://drive.google.com/uc?export=view&id=1StI2aw75Lqt62a9gWhwfcW5qmkF5aHdl)





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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>ralation_dummy</td>
      <td>0.567421</td>
    </tr>
    <tr>
      <th>16</th>
      <td>satisfaction_dummy</td>
      <td>0.234948</td>
    </tr>
    <tr>
      <th>7</th>
      <td>doctorex_dummy</td>
      <td>0.201105</td>
    </tr>
    <tr>
      <th>8</th>
      <td>edu_dummy</td>
      <td>0.125854</td>
    </tr>
    <tr>
      <th>10</th>
      <td>jobreturnopinion_dummy</td>
      <td>0.098623</td>
    </tr>
    <tr>
      <th>20</th>
      <td>work_dummy</td>
      <td>0.067368</td>
    </tr>
    <tr>
      <th>9</th>
      <td>jobdoctorex_dummy</td>
      <td>0.063995</td>
    </tr>
    <tr>
      <th>19</th>
      <td>spouse_dummy</td>
      <td>0.061106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dis_dummy</td>
      <td>0.041362</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age_dummy</td>
      <td>0.037978</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cureperiod_dummy</td>
      <td>0.034179</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alcohol_dummy</td>
      <td>0.026205</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dis_level</td>
      <td>0.024988</td>
    </tr>
    <tr>
      <th>12</th>
      <td>laborcontract_dummy</td>
      <td>0.011096</td>
    </tr>
    <tr>
      <th>18</th>
      <td>smoke_dummy</td>
      <td>0.007908</td>
    </tr>
    <tr>
      <th>13</th>
      <td>labortime_dummy</td>
      <td>0.006619</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dis_range</td>
      <td>0.005219</td>
    </tr>
  </tbody>
</table>
</div>




```python
#selected_variables_sv = sorted(list(set(temp_sv.columns)+set(label_features)))
selected_variables_sv = temp_sv['feature'].to_list()+ ['rtor_dummy']
```


```python
sample1_sv = df[["js_dummy", "psy_dummy", "med_dummy", *selected_variables_sv]] # 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample2_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", *selected_variables_sv]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample3_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", *selected_variables_sv]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample4_sv = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", *selected_variables_sv]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample5_sv = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample6_sv = df[["js_dummy", "psy_dummy", "med_dummy", "income", *selected_variables_sv]] # 임금/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample7_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", *selected_variables_sv]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample8_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", *selected_variables_sv]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample9_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", *selected_variables_sv]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample10_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample11_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample12_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample13_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample14_sv = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample15_sv = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample16_sv = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample17_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", *selected_variables_sv]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample18_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", *selected_variables_sv]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample19_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "income", *selected_variables_sv]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample20_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample21_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample22_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample23_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample24_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample25_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample26_sv = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample27_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_sv]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample28_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_sv]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample29_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_sv]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample30_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample31_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample32_sv = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_sv]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스
```


```python
samples_sv = [sample1_sv, sample2_sv, sample3_sv, sample4_sv, sample5_sv, sample6_sv, sample7_sv, sample8_sv, sample9_sv, sample10_sv
             , sample11_sv, sample12_sv, sample13_sv, sample14_sv, sample15_sv, sample16_sv,sample17_sv, sample18_sv, sample19_sv, sample20_sv, 
              sample21_sv, sample22_sv, sample23_sv, sample24_sv, sample25_sv, sample26_sv
             , sample27_sv, sample28_sv, sample29_sv, sample30_sv, sample31_sv, sample32_sv]
features_sv=[]
X_sv = []
y_sv = []
X_train_sv = []
X_test_sv = []
y_train_sv = []
y_test_sv = []

for i,sample in enumerate(samples_sv):
    features_sv.append(sorted(list(set(sample.columns) - set(label_features))))
    X_sv.append(sample[features_sv[i]])
    y_sv.append(sample[label_features])
    X_train_sv.append(train_test_split(X_sv[i], y_sv[i], test_size=0.3, random_state=42)[0])
    X_test_sv.append(train_test_split(X_sv[i], y_sv[i], test_size=0.3, random_state=42)[1])
    y_train_sv.append(train_test_split(X_sv[i], y_sv[i], test_size=0.3, random_state=42)[2])
    y_test_sv.append(train_test_split(X_sv[i], y_sv[i], test_size=0.3, random_state=42)[3])
    X_train_sv[i] = X_train_sv[i].reset_index(drop=True)
    X_test_sv[i] = X_test_sv[i].reset_index(drop=True)                   
    y_train_sv[i] = y_train_sv[i].reset_index(drop=True)
    y_test_sv[i] = y_test_sv[i].reset_index(drop=True)
```


```python
for i in range(32):
    print('---- sample{}_sv ----'.format(i+1))
    accuracy = cross_val_score(sv_class, X_train_sv[i], y_train_sv[i], scoring='accuracy', cv = 10).mean() * 100
    print("Accuracy of support vector machine is: " , accuracy)
```

    ---- sample1_sv ----
    Accuracy of support vector machine is:  77.12121212121212
    ---- sample2_sv ----
    Accuracy of support vector machine is:  76.89393939393939
    ---- sample3_sv ----
    Accuracy of support vector machine is:  77.12121212121212
    ---- sample4_sv ----
    Accuracy of support vector machine is:  77.04545454545453
    ---- sample5_sv ----
    Accuracy of support vector machine is:  75.53030303030303
    ---- sample6_sv ----
    Accuracy of support vector machine is:  55.6060606060606
    ---- sample7_sv ----
    Accuracy of support vector machine is:  75.83333333333334
    ---- sample8_sv ----
    Accuracy of support vector machine is:  77.5
    ---- sample9_sv ----
    Accuracy of support vector machine is:  77.65151515151516
    ---- sample10_sv ----
    Accuracy of support vector machine is:  58.10606060606061
    ---- sample11_sv ----
    Accuracy of support vector machine is:  75.68181818181819
    ---- sample12_sv ----
    Accuracy of support vector machine is:  76.96969696969698
    ---- sample13_sv ----
    Accuracy of support vector machine is:  57.80303030303029
    ---- sample14_sv ----
    Accuracy of support vector machine is:  76.28787878787878
    ---- sample15_sv ----
    Accuracy of support vector machine is:  60.45454545454546
    ---- sample16_sv ----
    Accuracy of support vector machine is:  63.93939393939394
    ---- sample17_sv ----
    Accuracy of support vector machine is:  77.95454545454545
    ---- sample18_sv ----
    Accuracy of support vector machine is:  77.1969696969697
    ---- sample19_sv ----
    Accuracy of support vector machine is:  56.060606060606055
    ---- sample20_sv ----
    Accuracy of support vector machine is:  75.68181818181819
    ---- sample21_sv ----
    Accuracy of support vector machine is:  63.63636363636363
    ---- sample22_sv ----
    Accuracy of support vector machine is:  62.803030303030305
    ---- sample23_sv ----
    Accuracy of support vector machine is:  75.22727272727273
    ---- sample24_sv ----
    Accuracy of support vector machine is:  56.8939393939394
    ---- sample25_sv ----
    Accuracy of support vector machine is:  64.31818181818181
    ---- sample26_sv ----
    Accuracy of support vector machine is:  68.33333333333333
    ---- sample27_sv ----
    Accuracy of support vector machine is:  77.5
    ---- sample28_sv ----
    Accuracy of support vector machine is:  59.46969696969696
    ---- sample29_sv ----
    Accuracy of support vector machine is:  65.45454545454545
    ---- sample30_sv ----
    Accuracy of support vector machine is:  65.07575757575756
    ---- sample31_sv ----
    Accuracy of support vector machine is:  65.53030303030303
    ---- sample32_sv ----
    Accuracy of support vector machine is:  62.196969696969695
    

#### randomsearch parameter tuning


```python
sv_params = {'C': range(1,100)}
params_list = [sv_params]

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=5, scoring='accuracy', random_state=42)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
best_param_dict_sv = dict()
samples_sv = ['sample1_sv', 'sample2_sv', 'sample3_sv', 'sample4_sv', 'sample5_sv', 'sample6_sv', 'sample7_sv', 'sample8_sv',
              'sample9_sv', 'sample10_sv', 'sample11_sv', 'sample12_sv', 'sample13_sv', 'sample14_sv', 'sample15_sv', 'sample16_sv',
             'sample17_sv', 'sample18_sv', 'sample19_sv', 'sample20_sv', 'sample21_sv', 'sample22_sv', 'sample23_sv', 'sample24_sv', 
              'sample25_sv', 'sample26_sv', 'sample27_sv', 'sample28_sv', 'sample29_sv', 'sample30_sv', 'sample31_sv', 'sample32_sv']

samples_sv1 = [sample1_sv, sample2_sv, sample3_sv, sample4_sv, sample5_sv, sample6_sv, sample7_sv, sample8_sv, sample9_sv, sample10_sv
             , sample11_sv, sample12_sv, sample13_sv, sample14_sv, sample15_sv, sample16_sv,sample17_sv, sample18_sv, sample19_sv, sample20_sv, 
              sample21_sv, sample22_sv, sample23_sv, sample24_sv, sample25_sv, sample26_sv
             , sample27_sv, sample28_sv, sample29_sv, sample30_sv, sample31_sv, sample32_sv]

for i,sample in enumerate(samples_sv):
    print('---sample{}_sv---'.format(i+1))
    best_params = hypertuning_rscv(sv_class, sv_params, 30, X_train_sv[i], y_train_sv[i])
    best_param_dict_sv[sample] = best_params[0]
    print('best_params : ', best_params[0])
```

    ---sample1_sv---
    best_params :  {'C': 1}
    ---sample2_sv---
    best_params :  {'C': 1}
    ---sample3_sv---
    best_params :  {'C': 1}
    ---sample4_sv---
    best_params :  {'C': 1}
    ---sample5_sv---
    best_params :  {'C': 1}
    ---sample6_sv---
    best_params :  {'C': 1}
    ---sample7_sv---
    best_params :  {'C': 1}
    ---sample8_sv---
    best_params :  {'C': 1}
    ---sample9_sv---
    best_params :  {'C': 1}
    ---sample10_sv---
    best_params :  {'C': 63}
    ---sample11_sv---
    best_params :  {'C': 1}
    ---sample12_sv---
    best_params :  {'C': 1}
    ---sample13_sv---
    best_params :  {'C': 1}
    ---sample14_sv---
    best_params :  {'C': 1}
    ---sample15_sv---
    best_params :  {'C': 63}
    ---sample16_sv---
    best_params :  {'C': 1}
    ---sample17_sv---
    best_params :  {'C': 1}
    ---sample18_sv---
    best_params :  {'C': 1}
    ---sample19_sv---
    best_params :  {'C': 1}
    ---sample20_sv---
    best_params :  {'C': 1}
    ---sample21_sv---
    best_params :  {'C': 1}
    ---sample22_sv---
    best_params :  {'C': 1}
    ---sample23_sv---
    best_params :  {'C': 1}
    ---sample24_sv---
    best_params :  {'C': 1}
    ---sample25_sv---
    best_params :  {'C': 1}
    ---sample26_sv---
    best_params :  {'C': 5}
    ---sample27_sv---
    best_params :  {'C': 85}
    ---sample28_sv---
    best_params :  {'C': 63}
    ---sample29_sv---
    best_params :  {'C': 1}
    ---sample30_sv---
    best_params :  {'C': 1}
    ---sample31_sv---
    best_params :  {'C': 63}
    ---sample32_sv---
    best_params :  {'C': 63}
    


```python
samples_sv = ['sample1_sv', 'sample2_sv', 'sample3_sv', 'sample4_sv', 'sample5_sv', 'sample6_sv', 'sample7_sv', 'sample8_sv',
              'sample9_sv', 'sample10_sv', 'sample11_sv', 'sample12_sv', 'sample13_sv', 'sample14_sv', 'sample15_sv', 'sample16_sv',
             'sample17_sv', 'sample18_sv', 'sample19_sv', 'sample20_sv', 'sample21_sv', 'sample22_sv', 'sample23_sv', 'sample24_sv', 
              'sample25_sv', 'sample26_sv', 'sample27_sv', 'sample28_sv', 'sample29_sv', 'sample30_sv', 'sample31_sv', 'sample32_sv']


accuracy = []
precision = []
sensitivity = []
Auc = []

for i,sample in enumerate(samples_sv):
    print('--------------sample{}_sv--------------'.format(i+1))
    clf = LinearSVC(random_state=42, **best_param_dict_sv[sample])
    clf.fit(X_train_sv[i], y_train_sv[i])
    y_pred_sv = clf.predict(X_test_sv[i])
    #y_pred_proba_sv = clf.predict_proba(X_test_sv[i])
    print("accuracy_score: {}".format( cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='accuracy', cv = 5).mean() * 100))
    accuracy.append(cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='accuracy', cv = 5).mean() * 100)
    print("precision_score: {}".format( cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='precision', cv = 5).mean() * 100))
    precision.append(cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='precision', cv = 5).mean() * 100)
    print("sensitivity_score: {}".format( cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='recall', cv = 5).mean() * 100))
    sensitivity.append(cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='recall', cv = 5).mean() * 100)
    try:
        print("AUC: Area Under Curve: {}".format(cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='roc_auc', cv = 5).mean() * 100))
        Auc.append(cross_val_score(sv_class, y_test_sv[i], y_pred_sv, scoring='roc_auc', cv = 5).mean() * 100)
    except ValueError:
        Auc.append(np.nan)
    
score_sv = pd.DataFrame({'accuracy':accuracy, 'precision':precision, 'sensitivity': sensitivity, 'Auc': Auc})
    #y_score = clf.decision_path(X_test[i])
    #precision, recall, _ = precision_recall_curve(y_test[i], y_score[1])
    #fpr, tpr, _ = roc_curve(y_test[i], y_score)
    
    #f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.set_size_inches((8, 4)) 
    #axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #axes[0].set_title('Recall-Precision Curve')
    
    #axes[1].plot(fpr, tpr)
    #axes[1].plot([0, 1], [0, 1], linestyle='--')
    #axes[1].set_title('ROC curve')

```

    --------------sample1_sv--------------
    accuracy_score: 79.52957615277131
    precision_score: 59.700417513716744
    sensitivity_score: 77.03225806451613
    AUC: Area Under Curve: 78.72917634345407
    --------------sample2_sv--------------
    accuracy_score: 78.83247942866014
    precision_score: 71.6786133960047
    sensitivity_score: 69.26829268292684
    AUC: Area Under Curve: 76.75553142517725
    --------------sample3_sv--------------
    accuracy_score: 76.71634839310666
    precision_score: 77.0871327254306
    sensitivity_score: 64.14007092198581
    AUC: Area Under Curve: 74.94882333978079
    --------------sample4_sv--------------
    accuracy_score: 79.17714640583759
    precision_score: 69.71215363256945
    sensitivity_score: 70.76923076923077
    AUC: Area Under Curve: 77.17560637560638
    --------------sample5_sv--------------
    accuracy_score: 80.41297935103245
    precision_score: 65.61449357755788
    sensitivity_score: 75.54621848739497
    AUC: Area Under Curve: 79.03893202850762
    --------------sample6_sv--------------
    accuracy_score: 89.94876571960874
    precision_score: 89.94876571960874
    sensitivity_score: 100.0
    AUC: Area Under Curve: 57.878787878787875
    --------------sample7_sv--------------
    accuracy_score: 78.12761993479273
    precision_score: 72.73529460218614
    sensitivity_score: 67.58582502768549
    AUC: Area Under Curve: 76.01625255408419
    --------------sample8_sv--------------
    accuracy_score: 79.88666356155876
    precision_score: 67.04326179726638
    sensitivity_score: 73.46846846846846
    AUC: Area Under Curve: 78.18278568278568
    --------------sample9_sv--------------
    accuracy_score: 77.5997515913678
    precision_score: 75.0494227994228
    sensitivity_score: 66.07070707070706
    AUC: Area Under Curve: 75.6014234415769
    --------------sample10_sv--------------
    accuracy_score: 90.29964291259121
    precision_score: 90.29964291259121
    sensitivity_score: 100.0
    AUC: Area Under Curve: 57.35475831992108
    --------------sample11_sv--------------
    accuracy_score: 78.6508306163639
    precision_score: 69.33037596568794
    sensitivity_score: 69.88461538461539
    AUC: Area Under Curve: 76.58735273735273
    --------------sample12_sv--------------
    accuracy_score: 74.59866480360195
    precision_score: 81.23100011393056
    sensitivity_score: 60.377358490566046
    AUC: Area Under Curve: 73.72419837096606
    --------------sample13_sv--------------
    accuracy_score: 76.89178698959788
    precision_score: 8.636363636363637
    sensitivity_score: 15.2
    AUC: Area Under Curve: 77.25633299284983
    --------------sample14_sv--------------
    accuracy_score: 78.8278217668064
    precision_score: 72.44304308980162
    sensitivity_score: 69.07084785133566
    AUC: Area Under Curve: 76.75764614789007
    --------------sample15_sv--------------
    accuracy_score: 89.77177456916628
    precision_score: 89.77177456916628
    sensitivity_score: 100.0
    AUC: Area Under Curve: 58.070101127759834
    --------------sample16_sv--------------
    accuracy_score: 90.29964291259121
    precision_score: 90.29964291259121
    sensitivity_score: 100.0
    AUC: Area Under Curve: 59.368066732429945
    --------------sample17_sv--------------
    accuracy_score: 75.12187548517313
    precision_score: 81.60306998063496
    sensitivity_score: 60.97968069666183
    AUC: Area Under Curve: 74.21934854505223
    --------------sample18_sv--------------
    accuracy_score: 77.95373389225276
    precision_score: 18.136363636363633
    sensitivity_score: 32.35507246376812
    AUC: Area Under Curve: 81.20772946859903
    --------------sample19_sv--------------
    accuracy_score: 77.95373389225276
    precision_score: 0.0
    sensitivity_score: 0.0
    AUC: Area Under Curve: 76.74269662921348
    --------------sample20_sv--------------
    accuracy_score: 77.59819903741655
    precision_score: 73.30240059378225
    sensitivity_score: 66.51162790697673
    AUC: Area Under Curve: 75.51254273195627
    --------------sample21_sv--------------
    accuracy_score: 90.29964291259121
    precision_score: 90.29964291259121
    sensitivity_score: 100.0
    AUC: Area Under Curve: 57.35475831992108
    --------------sample22_sv--------------
    accuracy_score: 89.94876571960874
    precision_score: 89.94876571960874
    sensitivity_score: 100.0
    AUC: Area Under Curve: 59.81729055258466
    --------------sample23_sv--------------
    accuracy_score: 77.94752367644774
    precision_score: 71.57458785435831
    sensitivity_score: 67.75842044134728
    AUC: Area Under Curve: 75.79626812364704
    --------------sample24_sv--------------
    accuracy_score: 90.4766340630337
    precision_score: 90.4766340630337
    sensitivity_score: 100.0
    AUC: Area Under Curve: 58.05013585310558
    --------------sample25_sv--------------
    accuracy_score: 73.18118304611086
    precision_score: 78.78500016607435
    sensitivity_score: 58.89695210449928
    AUC: Area Under Curve: 72.20530665334253
    --------------sample26_sv--------------
    accuracy_score: 86.24437199192671
    precision_score: 0.0
    sensitivity_score: 0.0
    AUC: Area Under Curve: 79.37510519671787
    --------------sample27_sv--------------
    accuracy_score: 69.84008694302128
    precision_score: 85.40712119023543
    sensitivity_score: 54.516129032258064
    AUC: Area Under Curve: 71.417189704666
    --------------sample28_sv--------------
    accuracy_score: 76.009936345288
    precision_score: 18.636363636363637
    sensitivity_score: 30.769230769230766
    AUC: Area Under Curve: 77.1305562253838
    --------------sample29_sv--------------
    accuracy_score: 73.71215649743829
    precision_score: 78.55716535143895
    sensitivity_score: 59.61538461538461
    AUC: Area Under Curve: 72.62895090102917
    --------------sample30_sv--------------
    accuracy_score: 86.06738084148424
    precision_score: 0.0
    sensitivity_score: 0.0
    AUC: Area Under Curve: 79.57795076793604
    --------------sample31_sv--------------
    accuracy_score: 98.76727216270764
    precision_score: 98.76727216270764
    sensitivity_score: 100.0
    AUC: Area Under Curve: 67.76785714285714
    --------------sample32_sv--------------
    accuracy_score: 98.76727216270764
    precision_score: 98.76727216270764
    sensitivity_score: 100.0
    AUC: Area Under Curve: 67.76785714285714
    


```python
score_sv
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
      <th>accuracy</th>
      <th>precision</th>
      <th>sensitivity</th>
      <th>Auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79.529576</td>
      <td>59.700418</td>
      <td>77.032258</td>
      <td>78.729176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78.832479</td>
      <td>71.678613</td>
      <td>69.268293</td>
      <td>76.755531</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76.716348</td>
      <td>77.087133</td>
      <td>64.140071</td>
      <td>74.948823</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79.177146</td>
      <td>69.712154</td>
      <td>70.769231</td>
      <td>77.175606</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80.412979</td>
      <td>65.614494</td>
      <td>75.546218</td>
      <td>79.038932</td>
    </tr>
    <tr>
      <th>5</th>
      <td>89.948766</td>
      <td>89.948766</td>
      <td>100.000000</td>
      <td>57.878788</td>
    </tr>
    <tr>
      <th>6</th>
      <td>78.127620</td>
      <td>72.735295</td>
      <td>67.585825</td>
      <td>76.016253</td>
    </tr>
    <tr>
      <th>7</th>
      <td>79.886664</td>
      <td>67.043262</td>
      <td>73.468468</td>
      <td>78.182786</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77.599752</td>
      <td>75.049423</td>
      <td>66.070707</td>
      <td>75.601423</td>
    </tr>
    <tr>
      <th>9</th>
      <td>90.299643</td>
      <td>90.299643</td>
      <td>100.000000</td>
      <td>57.354758</td>
    </tr>
    <tr>
      <th>10</th>
      <td>78.650831</td>
      <td>69.330376</td>
      <td>69.884615</td>
      <td>76.587353</td>
    </tr>
    <tr>
      <th>11</th>
      <td>74.598665</td>
      <td>81.231000</td>
      <td>60.377358</td>
      <td>73.724198</td>
    </tr>
    <tr>
      <th>12</th>
      <td>76.891787</td>
      <td>8.636364</td>
      <td>15.200000</td>
      <td>77.256333</td>
    </tr>
    <tr>
      <th>13</th>
      <td>78.827822</td>
      <td>72.443043</td>
      <td>69.070848</td>
      <td>76.757646</td>
    </tr>
    <tr>
      <th>14</th>
      <td>89.771775</td>
      <td>89.771775</td>
      <td>100.000000</td>
      <td>58.070101</td>
    </tr>
    <tr>
      <th>15</th>
      <td>90.299643</td>
      <td>90.299643</td>
      <td>100.000000</td>
      <td>59.368067</td>
    </tr>
    <tr>
      <th>16</th>
      <td>75.121875</td>
      <td>81.603070</td>
      <td>60.979681</td>
      <td>74.219349</td>
    </tr>
    <tr>
      <th>17</th>
      <td>77.953734</td>
      <td>18.136364</td>
      <td>32.355072</td>
      <td>81.207729</td>
    </tr>
    <tr>
      <th>18</th>
      <td>77.953734</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>76.742697</td>
    </tr>
    <tr>
      <th>19</th>
      <td>77.598199</td>
      <td>73.302401</td>
      <td>66.511628</td>
      <td>75.512543</td>
    </tr>
    <tr>
      <th>20</th>
      <td>90.299643</td>
      <td>90.299643</td>
      <td>100.000000</td>
      <td>57.354758</td>
    </tr>
    <tr>
      <th>21</th>
      <td>89.948766</td>
      <td>89.948766</td>
      <td>100.000000</td>
      <td>59.817291</td>
    </tr>
    <tr>
      <th>22</th>
      <td>77.947524</td>
      <td>71.574588</td>
      <td>67.758420</td>
      <td>75.796268</td>
    </tr>
    <tr>
      <th>23</th>
      <td>90.476634</td>
      <td>90.476634</td>
      <td>100.000000</td>
      <td>58.050136</td>
    </tr>
    <tr>
      <th>24</th>
      <td>73.181183</td>
      <td>78.785000</td>
      <td>58.896952</td>
      <td>72.205307</td>
    </tr>
    <tr>
      <th>25</th>
      <td>86.244372</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>79.375105</td>
    </tr>
    <tr>
      <th>26</th>
      <td>69.840087</td>
      <td>85.407121</td>
      <td>54.516129</td>
      <td>71.417190</td>
    </tr>
    <tr>
      <th>27</th>
      <td>76.009936</td>
      <td>18.636364</td>
      <td>30.769231</td>
      <td>77.130556</td>
    </tr>
    <tr>
      <th>28</th>
      <td>73.712156</td>
      <td>78.557165</td>
      <td>59.615385</td>
      <td>72.628951</td>
    </tr>
    <tr>
      <th>29</th>
      <td>86.067381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>79.577951</td>
    </tr>
    <tr>
      <th>30</th>
      <td>98.767272</td>
      <td>98.767272</td>
      <td>100.000000</td>
      <td>67.767857</td>
    </tr>
    <tr>
      <th>31</th>
      <td>98.767272</td>
      <td>98.767272</td>
      <td>100.000000</td>
      <td>67.767857</td>
    </tr>
  </tbody>
</table>
</div>




```python
samples_sv = ['sample1_sv', 'sample2_sv', 'sample3_sv', 'sample4_sv', 'sample5_sv', 'sample6_sv', 'sample7_sv', 'sample8_sv',
              'sample9_sv', 'sample10_sv', 'sample11_sv', 'sample12_sv', 'sample13_sv', 'sample14_sv', 'sample15_sv', 'sample16_sv',
             'sample17_sv', 'sample18_sv', 'sample19_sv', 'sample20_sv', 'sample21_sv', 'sample22_sv', 'sample23_sv', 'sample24_sv', 
              'sample25_sv', 'sample26_sv', 'sample27_sv', 'sample28_sv', 'sample29_sv', 'sample30_sv', 'sample31_sv', 'sample32_sv']

samples_sv1 = [sample1_sv, sample2_sv, sample3_sv, sample4_sv, sample5_sv, sample6_sv, sample7_sv, sample8_sv, sample9_sv, sample10_sv
             , sample11_sv, sample12_sv, sample13_sv, sample14_sv, sample15_sv, sample16_sv,sample17_sv, sample18_sv, sample19_sv, sample20_sv, 
              sample21_sv, sample22_sv, sample23_sv, sample24_sv, sample25_sv, sample26_sv
             , sample27_sv, sample28_sv, sample29_sv, sample30_sv, sample31_sv, sample32_sv]

feature_order = []
psyservice_order = []
psyservice_score = []
jobservice_order = []
jobservice_score = []
medservice_order = []
medservice_score = []

for i,(sample,sample1) in enumerate(zip(samples_sv, samples_sv1)):
    print('--------------sample{}_sv--------------'.format(i+1))
    clf = LinearSVC(random_state=42, **best_param_dict_sv[sample])
    clf.fit(X_train_sv[i], y_train_sv[i])
    
    crucial_features = pd.DataFrame({'feature':list(set(sample1.columns) - set(label_features)), 'importance':clf.coef_.ravel()})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features.reset_index(drop = True, inplace=True)
    
    feature = pd.DataFrame({sample: crucial_features['feature']})
    feature_order.append(feature)
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')])
    psyservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].index.to_list())
    psyservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')])
    jobservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].index.to_list())
    jobservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')])
    medservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].index.to_list())
    medservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].to_list())

feature_order_sv = pd.concat(feature_order, axis =1 )
service_sv = pd.DataFrame({'psy_order': psyservice_order, 'psy_score': psyservice_score, 'job_order': jobservice_order,
                          'job_score': jobservice_score, 'med_order': medservice_order, 'med_score': medservice_score}, index = samples_sv)
for col in service_sv.columns:
       service_sv[col] = service_sv[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
```

    --------------sample1_sv--------------
    3    0.121825
    Name: importance, dtype: float64
    17   -0.031537
    Name: importance, dtype: float64
    15    0.00175
    Name: importance, dtype: float64
    --------------sample2_sv--------------
    4    0.099475
    Name: importance, dtype: float64
    17   -0.017924
    Name: importance, dtype: float64
    14    0.006459
    Name: importance, dtype: float64
    --------------sample3_sv--------------
    2    0.124344
    Name: importance, dtype: float64
    18   -0.043212
    Name: importance, dtype: float64
    19   -0.099063
    Name: importance, dtype: float64
    --------------sample4_sv--------------
    3    0.126587
    Name: importance, dtype: float64
    18   -0.036843
    Name: importance, dtype: float64
    13    0.008532
    Name: importance, dtype: float64
    --------------sample5_sv--------------
    2    0.145224
    Name: importance, dtype: float64
    18   -0.066892
    Name: importance, dtype: float64
    9    0.044697
    Name: importance, dtype: float64
    --------------sample6_sv--------------
    12   -0.026667
    Name: importance, dtype: float64
    17   -0.052898
    Name: importance, dtype: float64
    2    0.041955
    Name: importance, dtype: float64
    --------------sample7_sv--------------
    5    0.107954
    Name: importance, dtype: float64
    18   -0.038662
    Name: importance, dtype: float64
    20   -0.076303
    Name: importance, dtype: float64
    --------------sample8_sv--------------
    4    0.100347
    Name: importance, dtype: float64
    18   -0.024247
    Name: importance, dtype: float64
    16    0.003895
    Name: importance, dtype: float64
    --------------sample9_sv--------------
    3    0.115012
    Name: importance, dtype: float64
    19   -0.051508
    Name: importance, dtype: float64
    18   -0.031108
    Name: importance, dtype: float64
    --------------sample10_sv--------------
    13   -0.02699
    Name: importance, dtype: float64
    18   -0.053298
    Name: importance, dtype: float64
    2    0.042328
    Name: importance, dtype: float64
    --------------sample11_sv--------------
    3    0.125656
    Name: importance, dtype: float64
    19   -0.040828
    Name: importance, dtype: float64
    17    0.002257
    Name: importance, dtype: float64
    --------------sample12_sv--------------
    2    0.12558
    Name: importance, dtype: float64
    19   -0.026949
    Name: importance, dtype: float64
    16   -0.011255
    Name: importance, dtype: float64
    --------------sample13_sv--------------
    13   -0.026551
    Name: importance, dtype: float64
    17   -0.052672
    Name: importance, dtype: float64
    16   -0.042359
    Name: importance, dtype: float64
    --------------sample14_sv--------------
    2    0.150873
    Name: importance, dtype: float64
    19   -0.047103
    Name: importance, dtype: float64
    9    0.05054
    Name: importance, dtype: float64
    --------------sample15_sv--------------
    14   -0.02656
    Name: importance, dtype: float64
    18   -0.052922
    Name: importance, dtype: float64
    2    0.043397
    Name: importance, dtype: float64
    --------------sample16_sv--------------
    7   -0.00014
    Name: importance, dtype: float64
    15   -0.047221
    Name: importance, dtype: float64
    21   -0.058041
    Name: importance, dtype: float64
    --------------sample17_sv--------------
    3    0.116751
    Name: importance, dtype: float64
    19   -0.039766
    Name: importance, dtype: float64
    16    0.00111
    Name: importance, dtype: float64
    --------------sample18_sv--------------
    3    0.10294
    Name: importance, dtype: float64
    20   -0.056849
    Name: importance, dtype: float64
    16   -0.013696
    Name: importance, dtype: float64
    --------------sample19_sv--------------
    14   -0.026545
    Name: importance, dtype: float64
    18   -0.052651
    Name: importance, dtype: float64
    17   -0.042329
    Name: importance, dtype: float64
    --------------sample20_sv--------------
    3    0.118807
    Name: importance, dtype: float64
    20   -0.06392
    Name: importance, dtype: float64
    17   -0.014183
    Name: importance, dtype: float64
    --------------sample21_sv--------------
    15   -0.026252
    Name: importance, dtype: float64
    18   -0.052499
    Name: importance, dtype: float64
    2    0.042883
    Name: importance, dtype: float64
    --------------sample22_sv--------------
    8   -0.000414
    Name: importance, dtype: float64
    16   -0.047182
    Name: importance, dtype: float64
    22   -0.058095
    Name: importance, dtype: float64
    --------------sample23_sv--------------
    2    0.123475
    Name: importance, dtype: float64
    20   -0.067871
    Name: importance, dtype: float64
    17   -0.008485
    Name: importance, dtype: float64
    --------------sample24_sv--------------
    15   -0.026053
    Name: importance, dtype: float64
    18   -0.052196
    Name: importance, dtype: float64
    9   -0.004173
    Name: importance, dtype: float64
    --------------sample25_sv--------------
    8   -0.001309
    Name: importance, dtype: float64
    16   -0.046949
    Name: importance, dtype: float64
    3    0.03827
    Name: importance, dtype: float64
    --------------sample26_sv--------------
    6   -0.000026
    Name: importance, dtype: float64
    16   -0.045563
    Name: importance, dtype: float64
    21   -0.057484
    Name: importance, dtype: float64
    --------------sample27_sv--------------
    4    0.141596
    Name: importance, dtype: float64
    22   -0.104345
    Name: importance, dtype: float64
    17   -0.025628
    Name: importance, dtype: float64
    --------------sample28_sv--------------
    16   -0.02641
    Name: importance, dtype: float64
    19   -0.052647
    Name: importance, dtype: float64
    10   -0.004717
    Name: importance, dtype: float64
    --------------sample29_sv--------------
    9   -0.001574
    Name: importance, dtype: float64
    17   -0.046912
    Name: importance, dtype: float64
    3    0.038208
    Name: importance, dtype: float64
    --------------sample30_sv--------------
    8   -0.000406
    Name: importance, dtype: float64
    17   -0.045173
    Name: importance, dtype: float64
    22   -0.056919
    Name: importance, dtype: float64
    --------------sample31_sv--------------
    8   -0.00088
    Name: importance, dtype: float64
    17   -0.045841
    Name: importance, dtype: float64
    3    0.040983
    Name: importance, dtype: float64
    --------------sample32_sv--------------
    9   -0.001151
    Name: importance, dtype: float64
    18   -0.045803
    Name: importance, dtype: float64
    3    0.040925
    Name: importance, dtype: float64
    


```python
feature_order_sv
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
      <th>sample1_sv</th>
      <th>sample2_sv</th>
      <th>sample3_sv</th>
      <th>sample4_sv</th>
      <th>sample5_sv</th>
      <th>sample6_sv</th>
      <th>sample7_sv</th>
      <th>sample8_sv</th>
      <th>sample9_sv</th>
      <th>sample10_sv</th>
      <th>...</th>
      <th>sample23_sv</th>
      <th>sample24_sv</th>
      <th>sample25_sv</th>
      <th>sample26_sv</th>
      <th>sample27_sv</th>
      <th>sample28_sv</th>
      <th>sample29_sv</th>
      <th>sample30_sv</th>
      <th>sample31_sv</th>
      <th>sample32_sv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>spouse_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>spouse_dummy</td>
      <td>...</td>
      <td>age_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>age_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cureperiod_dummy</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
      <td>...</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>labor_dummy</td>
      <td>edu_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
    </tr>
    <tr>
      <th>2</th>
      <td>jobreturnopinion_dummy</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>med_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>edu_dummy</td>
      <td>med_dummy</td>
      <td>...</td>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>smoke_dummy</td>
      <td>workhour_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>emplev_dummy</td>
      <td>income</td>
      <td>income</td>
    </tr>
    <tr>
      <th>3</th>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>workhour_dummy</td>
      <td>dis_range</td>
      <td>emplev_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>dis_range</td>
      <td>...</td>
      <td>workhour_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>edu_dummy</td>
      <td>med_dummy</td>
      <td>med_dummy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alcohol_dummy</td>
      <td>psy_dummy</td>
      <td>smoke_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>workhour_dummy</td>
      <td>cureperiod_dummy</td>
      <td>...</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>psy_dummy</td>
      <td>dis_range</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>spouse_dummy</td>
      <td>smoke_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>dis_dummy</td>
      <td>psy_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>...</td>
      <td>laborcontract_dummy</td>
      <td>dis_range</td>
      <td>age_dummy</td>
      <td>dis_dummy</td>
      <td>laborcontract_dummy</td>
      <td>cureperiod_dummy</td>
      <td>age_dummy</td>
      <td>income</td>
      <td>age_dummy</td>
      <td>age_dummy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>doctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>dis_dummy</td>
      <td>...</td>
      <td>jobdoctorex_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>psy_dummy</td>
      <td>dis_level</td>
      <td>labor_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>jobreturnopinion_dummy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>laborcontract_dummy</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
      <td>labor_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>laborcontract_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>doctorex_dummy</td>
      <td>...</td>
      <td>labor_dummy</td>
      <td>labor_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>satisfaction_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>dis_dummy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dis_range</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>income</td>
      <td>smoke_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>...</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>labor_dummy</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ralation_dummy</td>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>income</td>
      <td>...</td>
      <td>dis_range</td>
      <td>med_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>dis_level</td>
      <td>ralation_dummy</td>
      <td>doctorex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>...</td>
      <td>satisfaction_dummy</td>
      <td>income</td>
      <td>ralation_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>med_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>ralation_dummy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>jobdoctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>satisfaction_dummy</td>
      <td>dis_level</td>
      <td>laborcontract_dummy</td>
      <td>doctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>alcohol_dummy</td>
      <td>...</td>
      <td>work_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>workhour_dummy</td>
      <td>alcohol_dummy</td>
      <td>income</td>
      <td>ralation_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>cureperiod_dummy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>work_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_level</td>
      <td>laborcontract_dummy</td>
      <td>...</td>
      <td>ralation_dummy</td>
      <td>doctorex_dummy</td>
      <td>workhour_dummy</td>
      <td>dis_level</td>
      <td>doctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
      <td>jobdoctorex_dummy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>labortime_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>med_dummy</td>
      <td>cureperiod_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>emplev_dummy</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>...</td>
      <td>dis_level</td>
      <td>laborcontract_dummy</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
      <td>work_dummy</td>
      <td>alcohol_dummy</td>
      <td>workhour_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>workhour_dummy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>dis_level</td>
      <td>med_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>smoke_dummy</td>
      <td>labortime_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>emplev_dummy</td>
      <td>smoke_dummy</td>
      <td>...</td>
      <td>cureperiod_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>dis_level</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_level</td>
    </tr>
    <tr>
      <th>15</th>
      <td>med_dummy</td>
      <td>dis_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>labortime_dummy</td>
      <td>...</td>
      <td>smoke_dummy</td>
      <td>psy_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
      <td>ralation_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
      <td>labor_dummy</td>
      <td>emplev_dummy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>dis_dummy</td>
      <td>labortime_dummy</td>
      <td>work_dummy</td>
      <td>smoke_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_level</td>
      <td>age_dummy</td>
      <td>med_dummy</td>
      <td>age_dummy</td>
      <td>emplev_dummy</td>
      <td>...</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>smoke_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>work_dummy</td>
      <td>labortime_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_level</td>
      <td>...</td>
      <td>med_dummy</td>
      <td>smoke_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>med_dummy</td>
      <td>age_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>alcohol_dummy</td>
    </tr>
    <tr>
      <th>18</th>
      <td>smoke_dummy</td>
      <td>age_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>age_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>med_dummy</td>
      <td>js_dummy</td>
      <td>...</td>
      <td>spouse_dummy</td>
      <td>js_dummy</td>
      <td>emplev_cont</td>
      <td>doctorex_dummy</td>
      <td>dis_dummy</td>
      <td>emplev_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>js_dummy</td>
    </tr>
    <tr>
      <th>19</th>
      <td>age_dummy</td>
      <td>emplev_dummy</td>
      <td>med_dummy</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>smoke_dummy</td>
      <td>js_dummy</td>
      <td>age_dummy</td>
      <td>...</td>
      <td>dis_dummy</td>
      <td>dis_level</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>spouse_dummy</td>
      <td>js_dummy</td>
      <td>emplev_cont</td>
      <td>alcohol_dummy</td>
      <td>alcohol_dummy</td>
      <td>satisfaction_dummy</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NaN</td>
      <td>work_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>satisfaction_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>work_dummy</td>
      <td>work_dummy</td>
      <td>...</td>
      <td>js_dummy</td>
      <td>labortime_dummy</td>
      <td>smoke_dummy</td>
      <td>work_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_level</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>smoke_dummy</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>smoke_dummy</td>
      <td>satisfaction_dummy</td>
      <td>...</td>
      <td>alcohol_dummy</td>
      <td>emplev_cont</td>
      <td>spouse_dummy</td>
      <td>med_dummy</td>
      <td>emplev_dummy</td>
      <td>labortime_dummy</td>
      <td>emplev_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>emplev_cont</td>
      <td>satisfaction_dummy</td>
      <td>work_dummy</td>
      <td>spouse_dummy</td>
      <td>js_dummy</td>
      <td>emplev_cont</td>
      <td>spouse_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
    </tr>
    <tr>
      <th>23</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>emplev_cont</td>
      <td>satisfaction_dummy</td>
      <td>work_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>work_dummy</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>spouse_dummy</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 32 columns</p>
</div>




```python
service_rf
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
      <th>psy_order</th>
      <th>psy_score</th>
      <th>job_order</th>
      <th>job_score</th>
      <th>med_order</th>
      <th>med_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample1_rf</th>
      <td>12.0</td>
      <td>0.020421</td>
      <td>22.0</td>
      <td>0.009737</td>
      <td>15.0</td>
      <td>0.017367</td>
    </tr>
    <tr>
      <th>sample2_rf</th>
      <td>2.0</td>
      <td>0.192216</td>
      <td>23.0</td>
      <td>0.002142</td>
      <td>19.0</td>
      <td>0.002904</td>
    </tr>
    <tr>
      <th>sample3_rf</th>
      <td>8.0</td>
      <td>0.028394</td>
      <td>23.0</td>
      <td>0.008402</td>
      <td>18.0</td>
      <td>0.015058</td>
    </tr>
    <tr>
      <th>sample4_rf</th>
      <td>12.0</td>
      <td>0.018416</td>
      <td>23.0</td>
      <td>0.008576</td>
      <td>16.0</td>
      <td>0.014789</td>
    </tr>
    <tr>
      <th>sample5_rf</th>
      <td>21.0</td>
      <td>0.012598</td>
      <td>23.0</td>
      <td>0.008530</td>
      <td>1.0</td>
      <td>0.107558</td>
    </tr>
    <tr>
      <th>sample6_rf</th>
      <td>2.0</td>
      <td>0.085716</td>
      <td>23.0</td>
      <td>0.006864</td>
      <td>16.0</td>
      <td>0.014307</td>
    </tr>
    <tr>
      <th>sample7_rf</th>
      <td>9.0</td>
      <td>0.024432</td>
      <td>24.0</td>
      <td>0.006708</td>
      <td>18.0</td>
      <td>0.011835</td>
    </tr>
    <tr>
      <th>sample8_rf</th>
      <td>2.0</td>
      <td>0.187477</td>
      <td>24.0</td>
      <td>0.003049</td>
      <td>20.0</td>
      <td>0.004835</td>
    </tr>
    <tr>
      <th>sample9_rf</th>
      <td>14.0</td>
      <td>0.018274</td>
      <td>24.0</td>
      <td>0.007694</td>
      <td>1.0</td>
      <td>0.119611</td>
    </tr>
    <tr>
      <th>sample10_rf</th>
      <td>2.0</td>
      <td>0.150868</td>
      <td>23.0</td>
      <td>0.005601</td>
      <td>20.0</td>
      <td>0.008770</td>
    </tr>
    <tr>
      <th>sample11_rf</th>
      <td>9.0</td>
      <td>0.026749</td>
      <td>24.0</td>
      <td>0.007568</td>
      <td>18.0</td>
      <td>0.014657</td>
    </tr>
    <tr>
      <th>sample12_rf</th>
      <td>15.0</td>
      <td>0.018578</td>
      <td>24.0</td>
      <td>0.008125</td>
      <td>17.0</td>
      <td>0.016470</td>
    </tr>
    <tr>
      <th>sample13_rf</th>
      <td>9.0</td>
      <td>0.027173</td>
      <td>24.0</td>
      <td>0.007971</td>
      <td>18.0</td>
      <td>0.015460</td>
    </tr>
    <tr>
      <th>sample14_rf</th>
      <td>20.0</td>
      <td>0.012279</td>
      <td>24.0</td>
      <td>0.008305</td>
      <td>1.0</td>
      <td>0.120573</td>
    </tr>
    <tr>
      <th>sample15_rf</th>
      <td>2.0</td>
      <td>0.081037</td>
      <td>24.0</td>
      <td>0.006605</td>
      <td>16.0</td>
      <td>0.013564</td>
    </tr>
    <tr>
      <th>sample16_rf</th>
      <td>14.0</td>
      <td>0.014493</td>
      <td>24.0</td>
      <td>0.007177</td>
      <td>1.0</td>
      <td>0.143488</td>
    </tr>
    <tr>
      <th>sample17_rf</th>
      <td>10.0</td>
      <td>0.018224</td>
      <td>25.0</td>
      <td>0.004835</td>
      <td>20.0</td>
      <td>0.008670</td>
    </tr>
    <tr>
      <th>sample18_rf</th>
      <td>2.0</td>
      <td>0.138856</td>
      <td>25.0</td>
      <td>0.005364</td>
      <td>19.0</td>
      <td>0.009256</td>
    </tr>
    <tr>
      <th>sample19_rf</th>
      <td>10.0</td>
      <td>0.020841</td>
      <td>25.0</td>
      <td>0.006364</td>
      <td>18.0</td>
      <td>0.011233</td>
    </tr>
    <tr>
      <th>sample20_rf</th>
      <td>14.0</td>
      <td>0.017473</td>
      <td>25.0</td>
      <td>0.007577</td>
      <td>1.0</td>
      <td>0.110139</td>
    </tr>
    <tr>
      <th>sample21_rf</th>
      <td>2.0</td>
      <td>0.112898</td>
      <td>25.0</td>
      <td>0.006323</td>
      <td>18.0</td>
      <td>0.012085</td>
    </tr>
    <tr>
      <th>sample22_rf</th>
      <td>4.0</td>
      <td>0.056490</td>
      <td>26.0</td>
      <td>0.004508</td>
      <td>1.0</td>
      <td>0.169895</td>
    </tr>
    <tr>
      <th>sample23_rf</th>
      <td>14.0</td>
      <td>0.017928</td>
      <td>25.0</td>
      <td>0.007726</td>
      <td>17.0</td>
      <td>0.015251</td>
    </tr>
    <tr>
      <th>sample24_rf</th>
      <td>11.0</td>
      <td>0.026398</td>
      <td>25.0</td>
      <td>0.007759</td>
      <td>19.0</td>
      <td>0.015047</td>
    </tr>
    <tr>
      <th>sample25_rf</th>
      <td>2.0</td>
      <td>0.076402</td>
      <td>25.0</td>
      <td>0.006640</td>
      <td>20.0</td>
      <td>0.011993</td>
    </tr>
    <tr>
      <th>sample26_rf</th>
      <td>16.0</td>
      <td>0.014332</td>
      <td>25.0</td>
      <td>0.006009</td>
      <td>1.0</td>
      <td>0.131176</td>
    </tr>
    <tr>
      <th>sample27_rf</th>
      <td>2.0</td>
      <td>0.164118</td>
      <td>25.0</td>
      <td>0.003058</td>
      <td>22.0</td>
      <td>0.004066</td>
    </tr>
    <tr>
      <th>sample28_rf</th>
      <td>13.0</td>
      <td>0.021002</td>
      <td>26.0</td>
      <td>0.006785</td>
      <td>20.0</td>
      <td>0.011925</td>
    </tr>
    <tr>
      <th>sample29_rf</th>
      <td>2.0</td>
      <td>0.101336</td>
      <td>26.0</td>
      <td>0.006727</td>
      <td>19.0</td>
      <td>0.011996</td>
    </tr>
    <tr>
      <th>sample30_rf</th>
      <td>3.0</td>
      <td>0.074182</td>
      <td>26.0</td>
      <td>0.006140</td>
      <td>1.0</td>
      <td>0.126279</td>
    </tr>
    <tr>
      <th>sample31_rf</th>
      <td>3.0</td>
      <td>0.071256</td>
      <td>26.0</td>
      <td>0.006363</td>
      <td>18.0</td>
      <td>0.013230</td>
    </tr>
    <tr>
      <th>sample32_rf</th>
      <td>2.0</td>
      <td>0.130561</td>
      <td>26.0</td>
      <td>0.004942</td>
      <td>21.0</td>
      <td>0.008340</td>
    </tr>
  </tbody>
</table>
</div>



### Light GBM

#### before parameter tuning


```python
lgb_class = lgb.LGBMClassifier(random_state=42)
```


```python
temp_lgb = plot_feature_importances(lgb_class)
temp_lgb
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>work_dummy</td>
      <td>454</td>
    </tr>
    <tr>
      <th>21</th>
      <td>workday_dummy</td>
      <td>431</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dis_level</td>
      <td>339</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age_dummy</td>
      <td>244</td>
    </tr>
    <tr>
      <th>2</th>
      <td>con_dummy</td>
      <td>229</td>
    </tr>
    <tr>
      <th>16</th>
      <td>satisfaction_dummy</td>
      <td>167</td>
    </tr>
    <tr>
      <th>8</th>
      <td>edu_dummy</td>
      <td>150</td>
    </tr>
  </tbody>
</table>
</div>




![png](https://drive.google.com/uc?export=view&id=1Um060zL3eaCflANfWiHg7Q53WNubtr1T)



```python
selected_variables_lgb = sorted(list(set(common_sample.columns) - set(jobcondition_features)))
```


```python
sample1_lgb = df[["js_dummy", "psy_dummy", "med_dummy", *selected_variables_lgb]] # 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample2_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", *selected_variables_lgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample3_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", *selected_variables_lgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample4_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", *selected_variables_lgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample5_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample6_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "income", *selected_variables_lgb]] # 임금/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample7_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", *selected_variables_lgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample8_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", *selected_variables_lgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample9_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", *selected_variables_lgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample10_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample11_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample12_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample13_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample14_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample15_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample16_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample17_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", *selected_variables_lgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample18_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", *selected_variables_lgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample19_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "income", *selected_variables_lgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample20_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample21_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample22_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample23_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample24_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample25_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample26_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample27_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_lgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample28_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_lgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample29_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_lgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample30_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample31_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample32_lgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_lgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스
```


```python
samples_lgb = [sample1_lgb, sample2_lgb, sample3_lgb, sample4_lgb, sample5_lgb, sample6_lgb, sample7_lgb, sample8_lgb, sample9_lgb, sample10_lgb
             , sample11_lgb, sample12_lgb, sample13_lgb, sample14_lgb, sample15_lgb, sample16_lgb,sample17_lgb, sample18_lgb, sample19_lgb, sample20_lgb, 
              sample21_lgb, sample22_lgb, sample23_lgb, sample24_lgb, sample25_lgb, sample26_lgb
             , sample27_lgb, sample28_lgb, sample29_lgb, sample30_lgb, sample31_lgb, sample32_lgb]
features_lgb=[]
X_lgb = []
y_lgb = []
X_train_lgb = []
X_test_lgb = []
y_train_lgb = []
y_test_lgb = []

for i,sample in enumerate(samples_lgb):
    features_lgb.append(sorted(list(set(sample.columns) - set(label_features))))
    X_lgb.append(sample[features_lgb[i]])
    y_lgb.append(sample[label_features])
    X_train_lgb.append(train_test_split(X_lgb[i], y_lgb[i], test_size=0.3, random_state=42)[0])
    X_test_lgb.append(train_test_split(X_lgb[i], y_lgb[i], test_size=0.3, random_state=42)[1])
    y_train_lgb.append(train_test_split(X_lgb[i], y_lgb[i], test_size=0.3, random_state=42)[2])
    y_test_lgb.append(train_test_split(X_lgb[i], y_lgb[i], test_size=0.3, random_state=42)[3])
    X_train_lgb[i] = X_train_lgb[i].reset_index(drop=True)
    X_test_lgb[i] = X_test_lgb[i].reset_index(drop=True)                   
    y_train_lgb[i] = y_train_lgb[i].reset_index(drop=True)
    y_test_lgb[i] = y_test_lgb[i].reset_index(drop=True)
```


```python
for i in range(2):
    print('---- sample{}_lgb ----'.format(i+1))
    accuracy = cross_val_score(lgb_class, X_train_lgb[i], y_train_lgb[i], scoring='accuracy', cv = 10).mean() * 100
    print("Accuracy of Light Gradient Boosting is: " , accuracy)
```

    ---- sample1_lgb ----
    Accuracy of Light Gradient Boosting is:  75.60606060606061
    ---- sample2_lgb ----
    Accuracy of Light Gradient Boosting is:  76.81818181818181
    

#### randomsearch parameter tuning


```python
lgb_params = {'max_depth': np.arange(3, 30),
             'num_leaves': np.arange(10, 100), 
             'learning_rate': np.linspace(0.001, 0.4, 50),
             'min_child_samples': randint(2, 30),
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             #'subsample': np.linspace(0.6, 0.9, 30, endpoint=True), 
             #'colsample_bytree': np.linspace(0.1, 0.8, 100, endpoint=True),
             #'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             #'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
             'n_estimators': np.arange(10, 500)}
params_list = [lgb_params]

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=5, scoring='accuracy', random_state=42)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
best_param_dict_lgb = dict()
samples_lgb = ['sample1_lgb', 'sample2_lgb', 'sample3_lgb', 'sample4_lgb', 'sample5_lgb', 'sample6_lgb', 'sample7_lgb', 'sample8_lgb',
              'sample9_lgb', 'sample10_lgb', 'sample11_lgb', 'sample12_lgb', 'sample13_lgb', 'sample14_lgb', 'sample15_lgb', 'sample16_lgb',
             'sample17_lgb', 'sample18_lgb', 'sample19_lgb', 'sample20_lgb', 'sample21_lgb', 'sample22_lgb', 'sample23_lgb', 'sample24_lgb', 
              'sample25_lgb', 'sample26_lgb', 'sample27_lgb', 'sample28_lgb', 'sample29_lgb', 'sample30_lgb', 'sample31_lgb', 'sample32_lgb']

samples_lgb1 = [sample1_lgb, sample2_lgb, sample3_lgb, sample4_lgb, sample5_lgb, sample6_lgb, sample7_lgb, sample8_lgb, sample9_lgb, sample10_lgb
             , sample11_lgb, sample12_lgb, sample13_lgb, sample14_lgb, sample15_lgb, sample16_lgb,sample17_lgb, sample18_lgb, sample19_lgb, sample20_lgb, 
              sample21_lgb, sample22_lgb, sample23_lgb, sample24_lgb, sample25_lgb, sample26_lgb
             , sample27_lgb, sample28_lgb, sample29_lgb, sample30_lgb, sample31_lgb, sample32_lgb]
for i,sample in enumerate(samples_lgb):
    print('---sample{}_lgb---'.format(i+1))
    best_params = hypertuning_rscv(lgb_class, lgb_params, 30, X_train_lgb[i], y_train_lgb[i])
    best_param_dict_lgb[sample] = best_params[0]
    print('best_params : ', best_params[0])
```

    ---sample1_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample2_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 8, 'min_child_samples': 23, 'n_estimators': 115, 'num_leaves': 13}
    ---sample3_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample4_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample5_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample6_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample7_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample8_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample9_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample10_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample11_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 24, 'min_child_samples': 22, 'n_estimators': 267, 'num_leaves': 97}
    ---sample12_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample13_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample14_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 24, 'min_child_samples': 22, 'n_estimators': 267, 'num_leaves': 97}
    ---sample15_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample16_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 24, 'min_child_samples': 22, 'n_estimators': 267, 'num_leaves': 97}
    ---sample17_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample18_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample19_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 7, 'min_child_samples': 20, 'n_estimators': 144, 'num_leaves': 30}
    ---sample20_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample21_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample22_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 7, 'min_child_samples': 20, 'n_estimators': 144, 'num_leaves': 30}
    ---sample23_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 24, 'min_child_samples': 22, 'n_estimators': 267, 'num_leaves': 97}
    ---sample24_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample25_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample26_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample27_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample28_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample29_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample30_lgb---
    best_params :  {'learning_rate': 0.10685714285714286, 'max_depth': 19, 'min_child_samples': 5, 'n_estimators': 59, 'num_leaves': 13}
    ---sample31_lgb---
    best_params :  {'learning_rate': 0.009142857142857144, 'max_depth': 22, 'min_child_samples': 29, 'n_estimators': 376, 'num_leaves': 69}
    ---sample32_lgb---
    best_params :  {'learning_rate': 0.017285714285714286, 'max_depth': 7, 'min_child_samples': 20, 'n_estimators': 144, 'num_leaves': 30}
    


```python
samples_lgb = ['sample1_lgb', 'sample2_lgb', 'sample3_lgb', 'sample4_lgb', 'sample5_lgb', 'sample6_lgb', 'sample7_lgb', 'sample8_lgb',
              'sample9_lgb', 'sample10_lgb', 'sample11_lgb', 'sample12_lgb', 'sample13_lgb', 'sample14_lgb', 'sample15_lgb', 'sample16_lgb',
             'sample17_lgb', 'sample18_lgb', 'sample19_lgb', 'sample20_lgb', 'sample21_lgb', 'sample22_lgb', 'sample23_lgb', 'sample24_lgb', 
              'sample25_lgb', 'sample26_lgb', 'sample27_lgb', 'sample28_lgb', 'sample29_lgb', 'sample30_lgb', 'sample31_lgb', 'sample32_lgb']

accuracy = []
precision = []
sensitivity = []
Auc = []

for i,sample in enumerate(samples_lgb):
    print('--------------sample{}_lgb--------------'.format(i+1))
    clf = lgb.LGBMClassifier(random_state=42, **best_param_dict_lgb[sample])
    clf.fit(X_train_lgb[i], y_train_lgb[i])
    y_pred_lgb = clf.predict(X_test_lgb[i])    
    
    print("accuracy_score: {}".format( cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='accuracy', cv = 5).mean() * 100))
    accuracy.append(cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='accuracy', cv = 5).mean() * 100)
    print("precision_score: {}".format( cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='precision', cv = 5).mean() * 100))
    precision.append(cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='precision', cv = 5).mean() * 100)
    print("sensitivity_score: {}".format( cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='recall', cv = 5).mean() * 100))
    sensitivity.append(cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='recall', cv = 5).mean() * 100)
    try:
        print("AUC: Area Under Curve: {}".format(cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='roc_auc', cv = 5).mean() * 100))
        Auc.append(cross_val_score(lgb_class, y_test_lgb[i], y_pred_lgb, scoring='roc_auc', cv = 5).mean() * 100)
    except ValueError:
        pass
    
score_lgb = pd.DataFrame({'accuracy':accuracy, 'precision':precision, 'sensitivity': sensitivity, 'Auc': Auc})
    
    #y_score = clf.decision_path(X_test[i])
    #precision, recall, _ = precision_recall_curve(y_test[i], y_score[1])
    #fpr, tpr, _ = roc_curve(y_test[i], y_score)
    
    #f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.set_size_inches((8, 4)) 
    #axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #axes[0].set_title('Recall-Precision Curve')
    
    #axes[1].plot(fpr, tpr)
    #axes[1].plot([0, 1], [0, 1], linestyle='--')
    #axes[1].set_title('ROC curve')

```

    --------------sample1_lgb--------------
    accuracy_score: 77.06256792423535
    precision_score: 68.17454292108907
    sensitivity_score: 67.15853658536585
    AUC: Area Under Curve: 74.83065666735897
    --------------sample2_lgb--------------
    accuracy_score: 78.47694457382394
    precision_score: 62.140659631479714
    sensitivity_score: 72.76292335115866
    AUC: Area Under Curve: 76.83399332114895
    --------------sample3_lgb--------------
    accuracy_score: 77.06256792423535
    precision_score: 66.51214994006756
    sensitivity_score: 67.6923076923077
    AUC: Area Under Curve: 74.83173943173944
    --------------sample4_lgb--------------
    accuracy_score: 77.06256792423535
    precision_score: 67.90395027834973
    sensitivity_score: 67.34615384615384
    AUC: Area Under Curve: 74.84116281719022
    --------------sample5_lgb--------------
    accuracy_score: 79.53112870672255
    precision_score: 72.63215349208278
    sensitivity_score: 70.2439024390244
    AUC: Area Under Curve: 77.51730890596578
    --------------sample6_lgb--------------
    accuracy_score: 80.76385654401491
    precision_score: 72.91611206504824
    sensitivity_score: 72.65384615384616
    AUC: Area Under Curve: 78.86857431720445
    --------------sample7_lgb--------------
    accuracy_score: 77.06101537028411
    precision_score: 68.70621290186507
    sensitivity_score: 66.97560975609757
    AUC: Area Under Curve: 74.82151872888593
    --------------sample8_lgb--------------
    accuracy_score: 77.77674274181028
    precision_score: 69.58712458712458
    sensitivity_score: 67.98780487804879
    AUC: Area Under Curve: 75.60919924267736
    --------------sample9_lgb--------------
    accuracy_score: 78.12451482689025
    precision_score: 69.25515625515627
    sensitivity_score: 68.84615384615384
    AUC: Area Under Curve: 75.99656822259561
    --------------sample10_lgb--------------
    accuracy_score: 79.8913212234125
    precision_score: 72.74790856604218
    sensitivity_score: 70.92682926829268
    AUC: Area Under Curve: 77.90062924601848
    --------------sample11_lgb--------------
    accuracy_score: 77.77363763390778
    precision_score: 69.86662307943773
    sensitivity_score: 67.8048780487805
    AUC: Area Under Curve: 75.60906002895645
    --------------sample12_lgb--------------
    accuracy_score: 78.8247166589039
    precision_score: 70.17484467141217
    sensitivity_score: 69.83333333333334
    AUC: Area Under Curve: 76.76042823645564
    --------------sample13_lgb--------------
    accuracy_score: 77.2411116286291
    precision_score: 68.14683744192033
    sensitivity_score: 67.5
    AUC: Area Under Curve: 75.02360236949278
    --------------sample14_lgb--------------
    accuracy_score: 76.89023443564663
    precision_score: 69.19216916565803
    sensitivity_score: 66.51567944250871
    AUC: Area Under Curve: 74.66765707285253
    --------------sample15_lgb--------------
    accuracy_score: 78.8278217668064
    precision_score: 70.19450317124736
    sensitivity_score: 69.84615384615384
    AUC: Area Under Curve: 76.77054082533535
    --------------sample16_lgb--------------
    accuracy_score: 78.65393572426642
    precision_score: 70.861188520763
    sensitivity_score: 69.0609756097561
    AUC: Area Under Curve: 76.53771763745036
    --------------sample17_lgb--------------
    accuracy_score: 77.41965533302283
    precision_score: 70.21751499783537
    sensitivity_score: 67.15447154471545
    AUC: Area Under Curve: 75.2439024390244
    --------------sample18_lgb--------------
    accuracy_score: 77.59354137556281
    precision_score: 69.10698568704932
    sensitivity_score: 67.81707317073172
    AUC: Area Under Curve: 75.41538590043434
    --------------sample19_lgb--------------
    accuracy_score: 78.4784971277752
    precision_score: 73.70282101989419
    sensitivity_score: 67.90697674418604
    AUC: Area Under Curve: 76.41827710448739
    --------------sample20_lgb--------------
    accuracy_score: 77.5997515913678
    precision_score: 70.16401726928042
    sensitivity_score: 67.47967479674797
    AUC: Area Under Curve: 75.42743252774994
    --------------sample21_lgb--------------
    accuracy_score: 79.35569011023132
    precision_score: 70.81504485852312
    sensitivity_score: 70.6923076923077
    AUC: Area Under Curve: 77.3417110471905
    --------------sample22_lgb--------------
    accuracy_score: 80.95482068001863
    precision_score: 75.73522792242116
    sensitivity_score: 71.77700348432057
    AUC: Area Under Curve: 79.04577873746545
    --------------sample23_lgb--------------
    accuracy_score: 77.41965533302282
    precision_score: 67.76982801801577
    sensitivity_score: 68.02564102564102
    AUC: Area Under Curve: 75.22903672903671
    --------------sample24_lgb--------------
    accuracy_score: 75.64819127464679
    precision_score: 65.13771415691498
    sensitivity_score: 65.44871794871796
    AUC: Area Under Curve: 73.26489951489951
    --------------sample25_lgb--------------
    accuracy_score: 78.30305853128397
    precision_score: 69.35598642120382
    sensitivity_score: 69.0
    AUC: Area Under Curve: 76.18456127360236
    --------------sample26_lgb--------------
    accuracy_score: 79.35724266418258
    precision_score: 70.78333312696886
    sensitivity_score: 70.65384615384616
    AUC: Area Under Curve: 77.3261826104292
    --------------sample27_lgb--------------
    accuracy_score: 77.95062878435026
    precision_score: 71.16840431262489
    sensitivity_score: 67.78164924506389
    AUC: Area Under Curve: 75.80397016243803
    --------------sample28_lgb--------------
    accuracy_score: 78.12451482689023
    precision_score: 70.59426791692238
    sensitivity_score: 68.29268292682927
    AUC: Area Under Curve: 75.98994876935072
    --------------sample29_lgb--------------
    accuracy_score: 79.35724266418258
    precision_score: 72.71189805706328
    sensitivity_score: 69.88385598141697
    AUC: Area Under Curve: 77.32206497700986
    --------------sample30_lgb--------------
    accuracy_score: 79.88976866946126
    precision_score: 73.77995987752085
    sensitivity_score: 70.53426248548199
    AUC: Area Under Curve: 77.90602013162989
    --------------sample31_lgb--------------
    accuracy_score: 78.11830461108525
    precision_score: 69.82739655547128
    sensitivity_score: 68.65853658536587
    AUC: Area Under Curve: 75.99161557146857
    --------------sample32_lgb--------------
    accuracy_score: 79.36500543393883
    precision_score: 74.02043191516876
    sensitivity_score: 69.3576965669989
    AUC: Area Under Curve: 77.35490462152762
    


```python
score_lgb
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
      <th>accuracy</th>
      <th>precision</th>
      <th>sensitivity</th>
      <th>Auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.062568</td>
      <td>68.174543</td>
      <td>67.158537</td>
      <td>74.830657</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78.476945</td>
      <td>62.140660</td>
      <td>72.762923</td>
      <td>76.833993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.062568</td>
      <td>66.512150</td>
      <td>67.692308</td>
      <td>74.831739</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77.062568</td>
      <td>67.903950</td>
      <td>67.346154</td>
      <td>74.841163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.531129</td>
      <td>72.632153</td>
      <td>70.243902</td>
      <td>77.517309</td>
    </tr>
    <tr>
      <th>5</th>
      <td>80.763857</td>
      <td>72.916112</td>
      <td>72.653846</td>
      <td>78.868574</td>
    </tr>
    <tr>
      <th>6</th>
      <td>77.061015</td>
      <td>68.706213</td>
      <td>66.975610</td>
      <td>74.821519</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77.776743</td>
      <td>69.587125</td>
      <td>67.987805</td>
      <td>75.609199</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78.124515</td>
      <td>69.255156</td>
      <td>68.846154</td>
      <td>75.996568</td>
    </tr>
    <tr>
      <th>9</th>
      <td>79.891321</td>
      <td>72.747909</td>
      <td>70.926829</td>
      <td>77.900629</td>
    </tr>
    <tr>
      <th>10</th>
      <td>77.773638</td>
      <td>69.866623</td>
      <td>67.804878</td>
      <td>75.609060</td>
    </tr>
    <tr>
      <th>11</th>
      <td>78.824717</td>
      <td>70.174845</td>
      <td>69.833333</td>
      <td>76.760428</td>
    </tr>
    <tr>
      <th>12</th>
      <td>77.241112</td>
      <td>68.146837</td>
      <td>67.500000</td>
      <td>75.023602</td>
    </tr>
    <tr>
      <th>13</th>
      <td>76.890234</td>
      <td>69.192169</td>
      <td>66.515679</td>
      <td>74.667657</td>
    </tr>
    <tr>
      <th>14</th>
      <td>78.827822</td>
      <td>70.194503</td>
      <td>69.846154</td>
      <td>76.770541</td>
    </tr>
    <tr>
      <th>15</th>
      <td>78.653936</td>
      <td>70.861189</td>
      <td>69.060976</td>
      <td>76.537718</td>
    </tr>
    <tr>
      <th>16</th>
      <td>77.419655</td>
      <td>70.217515</td>
      <td>67.154472</td>
      <td>75.243902</td>
    </tr>
    <tr>
      <th>17</th>
      <td>77.593541</td>
      <td>69.106986</td>
      <td>67.817073</td>
      <td>75.415386</td>
    </tr>
    <tr>
      <th>18</th>
      <td>78.478497</td>
      <td>73.702821</td>
      <td>67.906977</td>
      <td>76.418277</td>
    </tr>
    <tr>
      <th>19</th>
      <td>77.599752</td>
      <td>70.164017</td>
      <td>67.479675</td>
      <td>75.427433</td>
    </tr>
    <tr>
      <th>20</th>
      <td>79.355690</td>
      <td>70.815045</td>
      <td>70.692308</td>
      <td>77.341711</td>
    </tr>
    <tr>
      <th>21</th>
      <td>80.954821</td>
      <td>75.735228</td>
      <td>71.777003</td>
      <td>79.045779</td>
    </tr>
    <tr>
      <th>22</th>
      <td>77.419655</td>
      <td>67.769828</td>
      <td>68.025641</td>
      <td>75.229037</td>
    </tr>
    <tr>
      <th>23</th>
      <td>75.648191</td>
      <td>65.137714</td>
      <td>65.448718</td>
      <td>73.264900</td>
    </tr>
    <tr>
      <th>24</th>
      <td>78.303059</td>
      <td>69.355986</td>
      <td>69.000000</td>
      <td>76.184561</td>
    </tr>
    <tr>
      <th>25</th>
      <td>79.357243</td>
      <td>70.783333</td>
      <td>70.653846</td>
      <td>77.326183</td>
    </tr>
    <tr>
      <th>26</th>
      <td>77.950629</td>
      <td>71.168404</td>
      <td>67.781649</td>
      <td>75.803970</td>
    </tr>
    <tr>
      <th>27</th>
      <td>78.124515</td>
      <td>70.594268</td>
      <td>68.292683</td>
      <td>75.989949</td>
    </tr>
    <tr>
      <th>28</th>
      <td>79.357243</td>
      <td>72.711898</td>
      <td>69.883856</td>
      <td>77.322065</td>
    </tr>
    <tr>
      <th>29</th>
      <td>79.889769</td>
      <td>73.779960</td>
      <td>70.534262</td>
      <td>77.906020</td>
    </tr>
    <tr>
      <th>30</th>
      <td>78.118305</td>
      <td>69.827397</td>
      <td>68.658537</td>
      <td>75.991616</td>
    </tr>
    <tr>
      <th>31</th>
      <td>79.365005</td>
      <td>74.020432</td>
      <td>69.357697</td>
      <td>77.354905</td>
    </tr>
  </tbody>
</table>
</div>




```python
samples_lgb = ['sample1_lgb', 'sample2_lgb', 'sample3_lgb', 'sample4_lgb', 'sample5_lgb', 'sample6_lgb', 'sample7_lgb', 'sample8_lgb',
              'sample9_lgb', 'sample10_lgb', 'sample11_lgb', 'sample12_lgb', 'sample13_lgb', 'sample14_lgb', 'sample15_lgb', 'sample16_lgb',
             'sample17_lgb', 'sample18_lgb', 'sample19_lgb', 'sample20_lgb', 'sample21_lgb', 'sample22_lgb', 'sample23_lgb', 'sample24_lgb', 
              'sample25_lgb', 'sample26_lgb', 'sample27_lgb', 'sample28_lgb', 'sample29_lgb', 'sample30_lgb', 'sample31_lgb', 'sample32_lgb']

samples_lgb1 = [sample1_lgb, sample2_lgb, sample3_lgb, sample4_lgb, sample5_lgb, sample6_lgb, sample7_lgb, sample8_lgb, sample9_lgb, sample10_lgb
             , sample11_lgb, sample12_lgb, sample13_lgb, sample14_lgb, sample15_lgb, sample16_lgb,sample17_lgb, sample18_lgb, sample19_lgb, sample20_lgb, 
              sample21_lgb, sample22_lgb, sample23_lgb, sample24_lgb, sample25_lgb, sample26_lgb
             , sample27_lgb, sample28_lgb, sample29_lgb, sample30_lgb, sample31_lgb, sample32_lgb]

feature_order = []
psyservice_order = []
psyservice_score = []
jobservice_order = []
jobservice_score = []
medservice_order = []
medservice_score = []

for i,(sample,sample1) in enumerate(zip(samples_lgb, samples_lgb1)):
    print('--------------sample{}_lgb--------------'.format(i+1))
    clf = lgb.LGBMClassifier(random_state=42, **best_param_dict_lgb[sample])
    clf.fit(X_train_lgb[i], y_train_lgb[i])
    
    crucial_features = pd.DataFrame({'feature':list(set(sample1.columns) - set(label_features)), 'importance':clf.feature_importances_})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features.reset_index(drop = True, inplace=True)
    
    feature = pd.DataFrame({sample: crucial_features['feature']})
    feature_order.append(feature)
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')])
    psyservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].index.to_list())
    psyservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')])
    jobservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].index.to_list())
    jobservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')])
    medservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].index.to_list())
    medservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].to_list())

feature_order_lgb = pd.concat(feature_order, axis =1 )
service_lgb = pd.DataFrame({'psy_order': psyservice_order, 'psy_score': psyservice_score, 'job_order': jobservice_order,
                          'job_score': jobservice_score, 'med_order': medservice_order, 'med_score': medservice_score}, index = samples_lgb)
for col in service_lgb.columns:
       service_lgb[col] = service_lgb[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
```

    --------------sample1_lgb--------------
    12    303
    Name: importance, dtype: int32
    17    94
    Name: importance, dtype: int32
    16    118
    Name: importance, dtype: int32
    --------------sample2_lgb--------------
    3    131
    Name: importance, dtype: int32
    18    0
    Name: importance, dtype: int32
    19    0
    Name: importance, dtype: int32
    --------------sample3_lgb--------------
    10    417
    Name: importance, dtype: int32
    17    99
    Name: importance, dtype: int32
    11    413
    Name: importance, dtype: int32
    --------------sample4_lgb--------------
    13    315
    Name: importance, dtype: int32
    19    74
    Name: importance, dtype: int32
    16    119
    Name: importance, dtype: int32
    --------------sample5_lgb--------------
    14    13
    Name: importance, dtype: int32
    23    2
    Name: importance, dtype: int32
    8    31
    Name: importance, dtype: int32
    --------------sample6_lgb--------------
    1    104
    Name: importance, dtype: int32
    24    5
    Name: importance, dtype: int32
    18    8
    Name: importance, dtype: int32
    --------------sample7_lgb--------------
    12    18
    Name: importance, dtype: int32
    25    4
    Name: importance, dtype: int32
    15    10
    Name: importance, dtype: int32
    --------------sample8_lgb--------------
    11    416
    Name: importance, dtype: int32
    22    23
    Name: importance, dtype: int32
    15    130
    Name: importance, dtype: int32
    --------------sample9_lgb--------------
    11    17
    Name: importance, dtype: int32
    25    3
    Name: importance, dtype: int32
    5    45
    Name: importance, dtype: int32
    --------------sample10_lgb--------------
    12    383
    Name: importance, dtype: int32
    19    73
    Name: importance, dtype: int32
    16    163
    Name: importance, dtype: int32
    --------------sample11_lgb--------------
    11    307
    Name: importance, dtype: int32
    20    80
    Name: importance, dtype: int32
    12    295
    Name: importance, dtype: int32
    --------------sample12_lgb--------------
    9    21
    Name: importance, dtype: int32
    24    2
    Name: importance, dtype: int32
    17    11
    Name: importance, dtype: int32
    --------------sample13_lgb--------------
    12    352
    Name: importance, dtype: int32
    21    62
    Name: importance, dtype: int32
    13    307
    Name: importance, dtype: int32
    --------------sample14_lgb--------------
    19    82
    Name: importance, dtype: int32
    20    76
    Name: importance, dtype: int32
    14    264
    Name: importance, dtype: int32
    --------------sample15_lgb--------------
    0    2078
    Name: importance, dtype: int32
    21    46
    Name: importance, dtype: int32
    16    113
    Name: importance, dtype: int32
    --------------sample16_lgb--------------
    14    217
    Name: importance, dtype: int32
    21    72
    Name: importance, dtype: int32
    12    269
    Name: importance, dtype: int32
    --------------sample17_lgb--------------
    11    17
    Name: importance, dtype: int32
    24    6
    Name: importance, dtype: int32
    19    10
    Name: importance, dtype: int32
    --------------sample18_lgb--------------
    9    23
    Name: importance, dtype: int32
    22    5
    Name: importance, dtype: int32
    16    12
    Name: importance, dtype: int32
    --------------sample19_lgb--------------
    10    133
    Name: importance, dtype: int32
    19    30
    Name: importance, dtype: int32
    14    90
    Name: importance, dtype: int32
    --------------sample20_lgb--------------
    14    329
    Name: importance, dtype: int32
    26    5
    Name: importance, dtype: int32
    9    480
    Name: importance, dtype: int32
    --------------sample21_lgb--------------
    10    27
    Name: importance, dtype: int32
    25    3
    Name: importance, dtype: int32
    19    7
    Name: importance, dtype: int32
    --------------sample22_lgb--------------
    0    595
    Name: importance, dtype: int32
    18    29
    Name: importance, dtype: int32
    8    204
    Name: importance, dtype: int32
    --------------sample23_lgb--------------
    14    265
    Name: importance, dtype: int32
    20    79
    Name: importance, dtype: int32
    18    143
    Name: importance, dtype: int32
    --------------sample24_lgb--------------
    12    339
    Name: importance, dtype: int32
    22    41
    Name: importance, dtype: int32
    13    311
    Name: importance, dtype: int32
    --------------sample25_lgb--------------
    0    2059
    Name: importance, dtype: int32
    22    54
    Name: importance, dtype: int32
    20    70
    Name: importance, dtype: int32
    --------------sample26_lgb--------------
    15    229
    Name: importance, dtype: int32
    21    57
    Name: importance, dtype: int32
    11    377
    Name: importance, dtype: int32
    --------------sample27_lgb--------------
    10    26
    Name: importance, dtype: int32
    24    6
    Name: importance, dtype: int32
    22    9
    Name: importance, dtype: int32
    --------------sample28_lgb--------------
    11    17
    Name: importance, dtype: int32
    26    3
    Name: importance, dtype: int32
    19    8
    Name: importance, dtype: int32
    --------------sample29_lgb--------------
    12    351
    Name: importance, dtype: int32
    25    15
    Name: importance, dtype: int32
    17    150
    Name: importance, dtype: int32
    --------------sample30_lgb--------------
    1    87
    Name: importance, dtype: int32
    25    4
    Name: importance, dtype: int32
    5    43
    Name: importance, dtype: int32
    --------------sample31_lgb--------------
    0    1925
    Name: importance, dtype: int32
    23    40
    Name: importance, dtype: int32
    19    87
    Name: importance, dtype: int32
    --------------sample32_lgb--------------
    10    143
    Name: importance, dtype: int32
    20    20
    Name: importance, dtype: int32
    18    27
    Name: importance, dtype: int32
    


```python
feature_order_lgb
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
      <th>sample1_lgb</th>
      <th>sample2_lgb</th>
      <th>sample3_lgb</th>
      <th>sample4_lgb</th>
      <th>sample5_lgb</th>
      <th>sample6_lgb</th>
      <th>sample7_lgb</th>
      <th>sample8_lgb</th>
      <th>sample9_lgb</th>
      <th>sample10_lgb</th>
      <th>...</th>
      <th>sample23_lgb</th>
      <th>sample24_lgb</th>
      <th>sample25_lgb</th>
      <th>sample26_lgb</th>
      <th>sample27_lgb</th>
      <th>sample28_lgb</th>
      <th>sample29_lgb</th>
      <th>sample30_lgb</th>
      <th>sample31_lgb</th>
      <th>sample32_lgb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>income</td>
      <td>...</td>
      <td>jobdoctorex_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>income</td>
      <td>jobdoctorex_dummy</td>
      <td>psy_dummy</td>
      <td>income</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spouse_dummy</td>
      <td>labortime_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>psy_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>...</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>psy_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>recoveryex_dummy</td>
      <td>work_dummy</td>
      <td>recoveryex_dummy</td>
      <td>labor_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>doctorex_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>...</td>
      <td>labor_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>sex_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>labor_dummy</td>
      <td>laborcontract_dummy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>labortime_dummy</td>
      <td>psy_dummy</td>
      <td>emplev_cont</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>...</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
      <td>spouse_dummy</td>
      <td>laborcontract_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>satisfaction_dummy</td>
      <td>spouse_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>emplev_cont</td>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>...</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>laborcontract_dummy</td>
      <td>labortime_dummy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>labortime_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>med_dummy</td>
      <td>dis_range</td>
      <td>...</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>doctorex_dummy</td>
      <td>age_dummy</td>
      <td>satisfaction_dummy</td>
      <td>med_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>workday_dummy</td>
      <td>recoveryex_dummy</td>
      <td>workday_dummy</td>
      <td>dis_range</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>...</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>alcohol_dummy</td>
      <td>dis_range</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>recoveryex_dummy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>recoveryex_dummy</td>
      <td>edu_dummy</td>
      <td>...</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>labortime_dummy</td>
      <td>labor_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
    </tr>
    <tr>
      <th>8</th>
      <td>edu_dummy</td>
      <td>satisfaction_dummy</td>
      <td>smoke_dummy</td>
      <td>edu_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>smoke_dummy</td>
      <td>...</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>alcohol_dummy</td>
      <td>satisfaction_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>smoke_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ralation_dummy</td>
      <td>workday_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>work_dummy</td>
      <td>con_dummy</td>
      <td>edu_dummy</td>
      <td>work_dummy</td>
      <td>workday_dummy</td>
      <td>work_dummy</td>
      <td>...</td>
      <td>ralation_dummy</td>
      <td>emplev_cont</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>work_dummy</td>
      <td>recoveryex_dummy</td>
      <td>workhour_dummy</td>
      <td>work_dummy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>laborchange_dummy</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>laborchange_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>con_dummy</td>
      <td>smoke_dummy</td>
      <td>cureperiod_dummy</td>
      <td>workday_dummy</td>
      <td>...</td>
      <td>cureperiod_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>cureperiod_dummy</td>
      <td>workday_dummy</td>
      <td>ralation_dummy</td>
      <td>psy_dummy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>work_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>sex_dummy</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>emplev_dummy</td>
      <td>...</td>
      <td>alcohol_dummy</td>
      <td>edu_dummy</td>
      <td>cureperiod_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
      <td>work_dummy</td>
      <td>workhour_dummy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>psy_dummy</td>
      <td>sex_dummy</td>
      <td>age_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>alcohol_dummy</td>
      <td>psy_dummy</td>
      <td>...</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>cureperiod_dummy</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>psy_dummy</td>
      <td>con_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>jobdoctorex_dummy</td>
      <td>emplev_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>cureperiod_dummy</td>
      <td>ralation_dummy</td>
      <td>laborchange_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
      <td>ralation_dummy</td>
      <td>...</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>smoke_dummy</td>
      <td>ralation_dummy</td>
      <td>labor_dummy</td>
      <td>ralation_dummy</td>
      <td>income</td>
      <td>cureperiod_dummy</td>
      <td>jobreturnopinion_dummy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>con_dummy</td>
      <td>age_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>con_dummy</td>
      <td>psy_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>emplev_dummy</td>
      <td>laborchange_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>...</td>
      <td>psy_dummy</td>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>doctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>con_dummy</td>
      <td>workday_dummy</td>
      <td>cureperiod_dummy</td>
      <td>alcohol_dummy</td>
      <td>cureperiod_dummy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>cureperiod_dummy</td>
      <td>ralation_dummy</td>
      <td>con_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>labortime_dummy</td>
      <td>doctorex_dummy</td>
      <td>med_dummy</td>
      <td>med_dummy</td>
      <td>emplev_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>...</td>
      <td>work_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>cureperiod_dummy</td>
      <td>sex_dummy</td>
      <td>emplev_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>ralation_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>con_dummy</td>
      <td>labortime_dummy</td>
      <td>med_dummy</td>
      <td>...</td>
      <td>con_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>edu_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>income</td>
      <td>ralation_dummy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>js_dummy</td>
      <td>edu_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>smoke_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
      <td>...</td>
      <td>edu_dummy</td>
      <td>con_dummy</td>
      <td>doctorex_dummy</td>
      <td>edu_dummy</td>
      <td>dis_dummy</td>
      <td>labortime_dummy</td>
      <td>med_dummy</td>
      <td>alcohol_dummy</td>
      <td>con_dummy</td>
      <td>emplev_dummy</td>
    </tr>
    <tr>
      <th>18</th>
      <td>doctorex_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>doctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>med_dummy</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>ralation_dummy</td>
      <td>cureperiod_dummy</td>
      <td>...</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>emplev_cont</td>
      <td>ralation_dummy</td>
      <td>con_dummy</td>
      <td>emplev_dummy</td>
      <td>edu_dummy</td>
      <td>med_dummy</td>
    </tr>
    <tr>
      <th>19</th>
      <td>age_dummy</td>
      <td>med_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>dis_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_dummy</td>
      <td>alcohol_dummy</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>...</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>age_dummy</td>
      <td>med_dummy</td>
      <td>edu_dummy</td>
      <td>dis_dummy</td>
      <td>med_dummy</td>
      <td>con_dummy</td>
    </tr>
    <tr>
      <th>20</th>
      <td>jobreturnopinion_dummy</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
      <td>dis_dummy</td>
      <td>work_dummy</td>
      <td>dis_dummy</td>
      <td>work_dummy</td>
      <td>alcohol_dummy</td>
      <td>...</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>med_dummy</td>
      <td>income</td>
      <td>edu_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>js_dummy</td>
    </tr>
    <tr>
      <th>21</th>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>age_dummy</td>
      <td>edu_dummy</td>
      <td>age_dummy</td>
      <td>...</td>
      <td>jobreturnopinion_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>js_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_dummy</td>
      <td>alcohol_dummy</td>
      <td>labortime_dummy</td>
      <td>doctorex_dummy</td>
      <td>age_dummy</td>
    </tr>
    <tr>
      <th>22</th>
      <td>sex_dummy</td>
      <td>laborchange_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>sex_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>dis_dummy</td>
      <td>...</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>labor_dummy</td>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>age_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dis_level</td>
      <td>con_dummy</td>
      <td>doctorex_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>alcohol_dummy</td>
      <td>sex_dummy</td>
      <td>smoke_dummy</td>
      <td>sex_dummy</td>
      <td>...</td>
      <td>emplev_cont</td>
      <td>cureperiod_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>sex_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>emplev_cont</td>
    </tr>
    <tr>
      <th>24</th>
      <td>smoke_dummy</td>
      <td>doctorex_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>jobreturnopinion_dummy</td>
      <td>js_dummy</td>
      <td>emplev_dummy</td>
      <td>labor_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>...</td>
      <td>doctorex_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>sex_dummy</td>
      <td>edu_dummy</td>
      <td>age_dummy</td>
      <td>dis_dummy</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>js_dummy</td>
      <td>laborchange_dummy</td>
      <td>js_dummy</td>
      <td>dis_level</td>
      <td>...</td>
      <td>sex_dummy</td>
      <td>labor_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
      <td>doctorex_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>dis_dummy</td>
      <td>sex_dummy</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>laborchange_dummy</td>
      <td>...</td>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>laborchange_dummy</td>
      <td>js_dummy</td>
      <td>doctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>sex_dummy</td>
      <td>labor_dummy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_level</td>
      <td>doctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_level</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>laborchange_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>laborchange_dummy</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 32 columns</p>
</div>




```python
service_lgb
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
      <th>psy_order</th>
      <th>psy_score</th>
      <th>job_order</th>
      <th>job_score</th>
      <th>med_order</th>
      <th>med_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample1_lgb</th>
      <td>12.0</td>
      <td>303.0</td>
      <td>17.0</td>
      <td>94.0</td>
      <td>16.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>sample2_lgb</th>
      <td>3.0</td>
      <td>131.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sample3_lgb</th>
      <td>10.0</td>
      <td>417.0</td>
      <td>17.0</td>
      <td>99.0</td>
      <td>11.0</td>
      <td>413.0</td>
    </tr>
    <tr>
      <th>sample4_lgb</th>
      <td>13.0</td>
      <td>315.0</td>
      <td>19.0</td>
      <td>74.0</td>
      <td>16.0</td>
      <td>119.0</td>
    </tr>
    <tr>
      <th>sample5_lgb</th>
      <td>14.0</td>
      <td>13.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>sample6_lgb</th>
      <td>1.0</td>
      <td>104.0</td>
      <td>24.0</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>sample7_lgb</th>
      <td>12.0</td>
      <td>18.0</td>
      <td>25.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>sample8_lgb</th>
      <td>11.0</td>
      <td>416.0</td>
      <td>22.0</td>
      <td>23.0</td>
      <td>15.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>sample9_lgb</th>
      <td>11.0</td>
      <td>17.0</td>
      <td>25.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>sample10_lgb</th>
      <td>12.0</td>
      <td>383.0</td>
      <td>19.0</td>
      <td>73.0</td>
      <td>16.0</td>
      <td>163.0</td>
    </tr>
    <tr>
      <th>sample11_lgb</th>
      <td>11.0</td>
      <td>307.0</td>
      <td>20.0</td>
      <td>80.0</td>
      <td>12.0</td>
      <td>295.0</td>
    </tr>
    <tr>
      <th>sample12_lgb</th>
      <td>9.0</td>
      <td>21.0</td>
      <td>24.0</td>
      <td>2.0</td>
      <td>17.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>sample13_lgb</th>
      <td>12.0</td>
      <td>352.0</td>
      <td>21.0</td>
      <td>62.0</td>
      <td>13.0</td>
      <td>307.0</td>
    </tr>
    <tr>
      <th>sample14_lgb</th>
      <td>19.0</td>
      <td>82.0</td>
      <td>20.0</td>
      <td>76.0</td>
      <td>14.0</td>
      <td>264.0</td>
    </tr>
    <tr>
      <th>sample15_lgb</th>
      <td>0.0</td>
      <td>2078.0</td>
      <td>21.0</td>
      <td>46.0</td>
      <td>16.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>sample16_lgb</th>
      <td>14.0</td>
      <td>217.0</td>
      <td>21.0</td>
      <td>72.0</td>
      <td>12.0</td>
      <td>269.0</td>
    </tr>
    <tr>
      <th>sample17_lgb</th>
      <td>11.0</td>
      <td>17.0</td>
      <td>24.0</td>
      <td>6.0</td>
      <td>19.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>sample18_lgb</th>
      <td>9.0</td>
      <td>23.0</td>
      <td>22.0</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>sample19_lgb</th>
      <td>10.0</td>
      <td>133.0</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>sample20_lgb</th>
      <td>14.0</td>
      <td>329.0</td>
      <td>26.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>sample21_lgb</th>
      <td>10.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>sample22_lgb</th>
      <td>0.0</td>
      <td>595.0</td>
      <td>18.0</td>
      <td>29.0</td>
      <td>8.0</td>
      <td>204.0</td>
    </tr>
    <tr>
      <th>sample23_lgb</th>
      <td>14.0</td>
      <td>265.0</td>
      <td>20.0</td>
      <td>79.0</td>
      <td>18.0</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>sample24_lgb</th>
      <td>12.0</td>
      <td>339.0</td>
      <td>22.0</td>
      <td>41.0</td>
      <td>13.0</td>
      <td>311.0</td>
    </tr>
    <tr>
      <th>sample25_lgb</th>
      <td>0.0</td>
      <td>2059.0</td>
      <td>22.0</td>
      <td>54.0</td>
      <td>20.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>sample26_lgb</th>
      <td>15.0</td>
      <td>229.0</td>
      <td>21.0</td>
      <td>57.0</td>
      <td>11.0</td>
      <td>377.0</td>
    </tr>
    <tr>
      <th>sample27_lgb</th>
      <td>10.0</td>
      <td>26.0</td>
      <td>24.0</td>
      <td>6.0</td>
      <td>22.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>sample28_lgb</th>
      <td>11.0</td>
      <td>17.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>sample29_lgb</th>
      <td>12.0</td>
      <td>351.0</td>
      <td>25.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>sample30_lgb</th>
      <td>1.0</td>
      <td>87.0</td>
      <td>25.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>sample31_lgb</th>
      <td>0.0</td>
      <td>1925.0</td>
      <td>23.0</td>
      <td>40.0</td>
      <td>19.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>sample32_lgb</th>
      <td>10.0</td>
      <td>143.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>18.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>



### XGboost

#### before parameter tuning


```python
xgb_class = xgb.XGBClassifier(random_state=42)
```


```python
temp_xgb = plot_feature_importances(xgb_class)
temp_xgb
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>ralation_dummy</td>
      <td>0.388021</td>
    </tr>
    <tr>
      <th>20</th>
      <td>work_dummy</td>
      <td>0.114812</td>
    </tr>
    <tr>
      <th>16</th>
      <td>satisfaction_dummy</td>
      <td>0.067053</td>
    </tr>
    <tr>
      <th>2</th>
      <td>con_dummy</td>
      <td>0.035440</td>
    </tr>
    <tr>
      <th>9</th>
      <td>jobdoctorex_dummy</td>
      <td>0.031256</td>
    </tr>
    <tr>
      <th>15</th>
      <td>recoveryex_dummy</td>
      <td>0.029616</td>
    </tr>
    <tr>
      <th>8</th>
      <td>edu_dummy</td>
      <td>0.026751</td>
    </tr>
  </tbody>
</table>
</div>




![png](https://drive.google.com/uc?export=view&id=1nqI4BxvPlsuoHXFJjR2X7KrW_swaVjVo)



```python
selected_variables_xgb = sorted(list(set(common_sample.columns) - set(jobcondition_features)))
```


```python
sample1_xgb = df[["js_dummy", "psy_dummy", "med_dummy", *selected_variables_xgb]] # 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample2_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", *selected_variables_xgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample3_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", *selected_variables_xgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample4_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", *selected_variables_xgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample5_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample6_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "income", *selected_variables_xgb]] # 임금/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample7_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", *selected_variables_xgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample8_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", *selected_variables_xgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample9_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", *selected_variables_xgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample10_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample11_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample12_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample13_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample14_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample15_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample16_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample17_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", *selected_variables_xgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample18_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", *selected_variables_xgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample19_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "income", *selected_variables_xgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample20_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample21_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample22_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample23_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample24_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample25_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample26_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample27_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_xgb]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample28_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_xgb]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample29_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_xgb]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample30_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample31_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample32_xgb = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_xgb]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스
```


```python
samples_xgb = [sample1_xgb, sample2_xgb, sample3_xgb, sample4_xgb, sample5_xgb, sample6_xgb, sample7_xgb, sample8_xgb, sample9_xgb, sample10_xgb
             , sample11_xgb, sample12_xgb, sample13_xgb, sample14_xgb, sample15_xgb, sample16_xgb,sample17_xgb, sample18_xgb, sample19_xgb, sample20_xgb, 
              sample21_xgb, sample22_xgb, sample23_xgb, sample24_xgb, sample25_xgb, sample26_xgb
             , sample27_xgb, sample28_xgb, sample29_xgb, sample30_xgb, sample31_xgb, sample32_xgb]
features_xgb=[]
X_xgb = []
y_xgb = []
X_train_xgb = []
X_test_xgb = []
y_train_xgb = []
y_test_xgb = []

for i,sample in enumerate(samples_xgb):
    features_xgb.append(sorted(list(set(sample.columns) - set(label_features))))
    X_xgb.append(sample[features_xgb[i]])
    y_xgb.append(sample[label_features])
    X_train_xgb.append(train_test_split(X_xgb[i], y_xgb[i], test_size=0.3, random_state=42)[0])
    X_test_xgb.append(train_test_split(X_xgb[i], y_xgb[i], test_size=0.3, random_state=42)[1])
    y_train_xgb.append(train_test_split(X_xgb[i], y_xgb[i], test_size=0.3, random_state=42)[2])
    y_test_xgb.append(train_test_split(X_xgb[i], y_xgb[i], test_size=0.3, random_state=42)[3])
    X_train_xgb[i] = X_train_xgb[i].reset_index(drop=True)
    X_test_xgb[i] = X_test_xgb[i].reset_index(drop=True)                   
    y_train_xgb[i] = y_train_xgb[i].reset_index(drop=True)
    y_test_xgb[i] = y_test_xgb[i].reset_index(drop=True)
```


```python
for i in range(2):
    print('---- sample{}_xgb ----'.format(i+1))
    accuracy = cross_val_score(xgb_class, X_train_xgb[i], y_train_xgb[i], scoring='accuracy', cv = 10).mean() * 100
    print("Accuracy of XGboost is: " , accuracy)
```

    ---- sample1_xgb ----
    Accuracy of XGboost is:  77.57575757575756
    ---- sample2_xgb ----
    Accuracy of XGboost is:  77.8030303030303
    

#### randomsearch parameter tuning


```python
xgb_params = xgb_params = {"eta":  np.linspace(0.001, 0.4, 50),
              "gamma": np.arange(0, 20),
              "max_depth": np.arange(1, 500)}
params_list = [xgb_params]

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=5, scoring='accuracy', random_state=42)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
best_param_dict_xgb = dict()
samples_xgb = ['sample1_xgb', 'sample2_xgb', 'sample3_xgb', 'sample4_xgb', 'sample5_xgb', 'sample6_xgb', 'sample7_xgb', 'sample8_xgb',
              'sample9_xgb', 'sample10_xgb', 'sample11_xgb', 'sample12_xgb', 'sample13_xgb', 'sample14_xgb', 'sample15_xgb', 'sample16_xgb',
             'sample17_xgb', 'sample18_xgb', 'sample19_xgb', 'sample20_xgb', 'sample21_xgb', 'sample22_xgb', 'sample23_xgb', 'sample24_xgb', 
              'sample25_xgb', 'sample26_xgb', 'sample27_xgb', 'sample28_xgb', 'sample29_xgb', 'sample30_xgb', 'sample31_xgb', 'sample32_xgb']

samples_xgb1 = [sample1_xgb, sample2_xgb, sample3_xgb, sample4_xgb, sample5_xgb, sample6_xgb, sample7_xgb, sample8_xgb, sample9_xgb, sample10_xgb
             , sample11_xgb, sample12_xgb, sample13_xgb, sample14_xgb, sample15_xgb, sample16_xgb,sample17_xgb, sample18_xgb, sample19_xgb, sample20_xgb, 
              sample21_xgb, sample22_xgb, sample23_xgb, sample24_xgb, sample25_xgb, sample26_xgb
             , sample27_xgb, sample28_xgb, sample29_xgb, sample30_xgb, sample31_xgb, sample32_xgb]
for i,sample in enumerate(samples_xgb):
    print('---sample{}_xgb---'.format(i+1))
    best_params = hypertuning_rscv(xgb_class, xgb_params, 30, X_train_xgb[i], y_train_xgb[i])
    best_param_dict_xgb[sample] = best_params[0]
    print('best_params : ', best_params[0])
```

    ---sample1_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample2_xgb---
    best_params :  {'max_depth': 219, 'gamma': 3, 'eta': 0.15571428571428572}
    ---sample3_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample4_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample5_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample6_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample7_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample8_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample9_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample10_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample11_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample12_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample13_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample14_xgb---
    best_params :  {'max_depth': 219, 'gamma': 3, 'eta': 0.15571428571428572}
    ---sample15_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample16_xgb---
    best_params :  {'max_depth': 273, 'gamma': 2, 'eta': 0.35114285714285715}
    ---sample17_xgb---
    best_params :  {'max_depth': 219, 'gamma': 3, 'eta': 0.15571428571428572}
    ---sample18_xgb---
    best_params :  {'max_depth': 440, 'gamma': 6, 'eta': 0.21271428571428572}
    ---sample19_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample20_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample21_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample22_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample23_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample24_xgb---
    best_params :  {'max_depth': 219, 'gamma': 3, 'eta': 0.15571428571428572}
    ---sample25_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample26_xgb---
    best_params :  {'max_depth': 63, 'gamma': 7, 'eta': 0.08242857142857143}
    ---sample27_xgb---
    best_params :  {'max_depth': 25, 'gamma': 5, 'eta': 0.26157142857142857}
    ---sample28_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample29_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    ---sample30_xgb---
    best_params :  {'max_depth': 440, 'gamma': 6, 'eta': 0.21271428571428572}
    ---sample31_xgb---
    best_params :  {'max_depth': 273, 'gamma': 2, 'eta': 0.35114285714285715}
    ---sample32_xgb---
    best_params :  {'max_depth': 203, 'gamma': 4, 'eta': 0.09871428571428571}
    


```python
samples_xgb = ['sample1_xgb', 'sample2_xgb', 'sample3_xgb', 'sample4_xgb', 'sample5_xgb', 'sample6_xgb', 'sample7_xgb', 'sample8_xgb',
              'sample9_xgb', 'sample10_xgb', 'sample11_xgb', 'sample12_xgb', 'sample13_xgb', 'sample14_xgb', 'sample15_xgb', 'sample16_xgb',
             'sample17_xgb', 'sample18_xgb', 'sample19_xgb', 'sample20_xgb', 'sample21_xgb', 'sample22_xgb', 'sample23_xgb', 'sample24_xgb', 
              'sample25_xgb', 'sample26_xgb', 'sample27_xgb', 'sample28_xgb', 'sample29_xgb', 'sample30_xgb', 'sample31_xgb', 'sample32_xgb']

accuracy = []
precision = []
sensitivity = []
Auc = []

for i,sample in enumerate(samples_xgb):
    print('--------------sample{}_xgb--------------'.format(i+1))
    clf = xgb.XGBClassifier(random_state=42, **best_param_dict_xgb[sample])
    clf.fit(X_train_xgb[i], y_train_xgb[i])
    y_pred_xgb = clf.predict(X_test_xgb[i])
    #y_pred_proba_xgb = clf.predict_proba(X_test_xgb[i])
    
    print("accuracy_score: {}".format( cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='accuracy', cv = 5).mean() * 100))
    accuracy.append(cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='accuracy', cv = 5).mean() * 100)
    print("precision_score: {}".format( cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='precision', cv = 5).mean() * 100))
    precision.append(cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='precision', cv = 5).mean() * 100)
    print("sensitivity_score: {}".format( cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='recall', cv = 5).mean() * 100))
    sensitivity.append(cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='recall', cv = 5).mean() * 100)
    try:
        print("AUC: Area Under Curve: {}".format(cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='roc_auc', cv = 5).mean() * 100))
        Auc.append(cross_val_score(xgb_class, y_test_xgb[i], y_pred_xgb, scoring='roc_auc', cv = 5).mean() * 100)
    except ValueError:
        pass
    
score_xgb = pd.DataFrame({'accuracy':accuracy, 'precision':precision, 'sensitivity': sensitivity, 'Auc': Auc})   
    #y_score = clf.decision_path(X_test[i])
    #precision, recall, _ = precision_recall_curve(y_test[i], y_score[1])
    #fpr, tpr, _ = roc_curve(y_test[i], y_score)
    
    #f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.set_size_inches((8, 4)) 
    #axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #axes[0].set_title('Recall-Precision Curve')
    
    #axes[1].plot(fpr, tpr)
    #axes[1].plot([0, 1], [0, 1], linestyle='--')
    #axes[1].set_title('ROC curve')

```

    --------------sample1_xgb--------------
    accuracy_score: 77.23334885887284
    precision_score: 65.57133675797515
    sensitivity_score: 68.42105263157893
    AUC: Area Under Curve: 75.04736842105262
    --------------sample2_xgb--------------
    accuracy_score: 78.47383946592143
    precision_score: 68.90575127417233
    sensitivity_score: 69.74358974358974
    AUC: Area Under Curve: 76.3943173943174
    --------------sample3_xgb--------------
    accuracy_score: 77.23645396677534
    precision_score: 67.67391304347827
    sensitivity_score: 67.64102564102564
    AUC: Area Under Curve: 75.00340804450393
    --------------sample4_xgb--------------
    accuracy_score: 78.99705014749263
    precision_score: 67.87683010606827
    sensitivity_score: 71.19487908961594
    AUC: Area Under Curve: 77.01498340445708
    --------------sample5_xgb--------------
    accuracy_score: 78.29529576152771
    precision_score: 67.73451910408433
    sensitivity_score: 69.77058029689609
    AUC: Area Under Curve: 76.21862348178136
    --------------sample6_xgb--------------
    accuracy_score: 78.64462040055892
    precision_score: 68.36567415568784
    sensitivity_score: 70.26990553306344
    AUC: Area Under Curve: 76.60161943319837
    --------------sample7_xgb--------------
    accuracy_score: 79.00481291724887
    precision_score: 70.82966401051507
    sensitivity_score: 70.0
    AUC: Area Under Curve: 76.95483154387264
    --------------sample8_xgb--------------
    accuracy_score: 78.47539201987269
    precision_score: 70.16780666444721
    sensitivity_score: 69.14634146341463
    AUC: Area Under Curve: 76.3706531456281
    --------------sample9_xgb--------------
    accuracy_score: 78.65548827821766
    precision_score: 69.07293316904301
    sensitivity_score: 69.8974358974359
    AUC: Area Under Curve: 76.59015939015939
    --------------sample10_xgb--------------
    accuracy_score: 79.36190032603632
    precision_score: 70.87020455105562
    sensitivity_score: 70.7051282051282
    AUC: Area Under Curve: 77.34812130360076
    --------------sample11_xgb--------------
    accuracy_score: 78.12140971898774
    precision_score: 66.32965998167566
    sensitivity_score: 70.04267425320056
    AUC: Area Under Curve: 76.07396870554764
    --------------sample12_xgb--------------
    accuracy_score: 78.12140971898775
    precision_score: 68.18090206073686
    sensitivity_score: 69.23076923076923
    AUC: Area Under Curve: 76.00457380457381
    --------------sample13_xgb--------------
    accuracy_score: 77.5873311597578
    precision_score: 67.56193934974634
    sensitivity_score: 68.5425101214575
    AUC: Area Under Curve: 75.4208046102783
    --------------sample14_xgb--------------
    accuracy_score: 77.76742741810277
    precision_score: 70.39652337326756
    sensitivity_score: 67.8048780487805
    AUC: Area Under Curve: 75.6052548539184
    --------------sample15_xgb--------------
    accuracy_score: 78.46918180406769
    precision_score: 67.43346253229974
    sensitivity_score: 70.3556187766714
    AUC: Area Under Curve: 76.44447605500237
    --------------sample16_xgb--------------
    accuracy_score: 80.06054960409875
    precision_score: 73.18578584536033
    sensitivity_score: 71.28048780487805
    AUC: Area Under Curve: 78.10599732709656
    --------------sample17_xgb--------------
    accuracy_score: 77.94752367644776
    precision_score: 69.76237255307024
    sensitivity_score: 68.31707317073169
    AUC: Area Under Curve: 75.80237220180422
    --------------sample18_xgb--------------
    accuracy_score: 78.12761993479273
    precision_score: 70.63885778275476
    sensitivity_score: 68.29268292682927
    AUC: Area Under Curve: 75.99185135686974
    --------------sample19_xgb--------------
    accuracy_score: 79.00481291724887
    precision_score: 71.83248329377705
    sensitivity_score: 69.59756097560977
    AUC: Area Under Curve: 76.94489920926607
    --------------sample20_xgb--------------
    accuracy_score: 77.59354137556281
    precision_score: 69.32047510079546
    sensitivity_score: 67.81707317073172
    AUC: Area Under Curve: 75.41538590043434
    --------------sample21_xgb--------------
    accuracy_score: 78.30150597733272
    precision_score: 69.72303504010821
    sensitivity_score: 69.0
    AUC: Area Under Curve: 76.18271010736764
    --------------sample22_xgb--------------
    accuracy_score: 79.35879521813382
    precision_score: 70.21022025369852
    sensitivity_score: 70.91025641025641
    AUC: Area Under Curve: 77.36503811503812
    --------------sample23_xgb--------------
    accuracy_score: 77.76432231020027
    precision_score: 69.01355336783344
    sensitivity_score: 68.33333333333333
    AUC: Area Under Curve: 75.6031716648155
    --------------sample24_xgb--------------
    accuracy_score: 77.94752367644777
    precision_score: 68.79044201412623
    sensitivity_score: 68.64102564102564
    AUC: Area Under Curve: 75.77552948100893
    --------------sample25_xgb--------------
    accuracy_score: 79.00170780934637
    precision_score: 70.7381932188405
    sensitivity_score: 70.0
    AUC: Area Under Curve: 76.95298037763791
    --------------sample26_xgb--------------
    accuracy_score: 78.29684831547897
    precision_score: 68.56686806836845
    sensitivity_score: 69.58164642375169
    AUC: Area Under Curve: 76.21064303169567
    --------------sample27_xgb--------------
    accuracy_score: 77.77053252600527
    precision_score: 70.83937963577412
    sensitivity_score: 67.63066202090593
    AUC: Area Under Curve: 75.62088656600852
    --------------sample28_xgb--------------
    accuracy_score: 77.77674274181028
    precision_score: 70.08321868786986
    sensitivity_score: 67.8048780487805
    AUC: Area Under Curve: 75.61096261647549
    --------------sample29_xgb--------------
    accuracy_score: 79.36034777208508
    precision_score: 71.32920790927155
    sensitivity_score: 70.5
    AUC: Area Under Curve: 77.3418178452425
    --------------sample30_xgb--------------
    accuracy_score: 79.88821611551002
    precision_score: 71.28951000690131
    sensitivity_score: 71.57692307692307
    AUC: Area Under Curve: 77.9506237006237
    --------------sample31_xgb--------------
    accuracy_score: 77.59819903741655
    precision_score: 70.81493990946291
    sensitivity_score: 67.28222996515679
    AUC: Area Under Curve: 75.41732781512927
    --------------sample32_xgb--------------
    accuracy_score: 78.12606738084148
    precision_score: 69.64593532099254
    sensitivity_score: 68.65853658536587
    AUC: Area Under Curve: 75.99161557146856
    


```python
score_xgb
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
      <th>accuracy</th>
      <th>precision</th>
      <th>sensitivity</th>
      <th>Auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.233349</td>
      <td>65.571337</td>
      <td>68.421053</td>
      <td>75.047368</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78.473839</td>
      <td>68.905751</td>
      <td>69.743590</td>
      <td>76.394317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.236454</td>
      <td>67.673913</td>
      <td>67.641026</td>
      <td>75.003408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78.997050</td>
      <td>67.876830</td>
      <td>71.194879</td>
      <td>77.014983</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78.295296</td>
      <td>67.734519</td>
      <td>69.770580</td>
      <td>76.218623</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78.644620</td>
      <td>68.365674</td>
      <td>70.269906</td>
      <td>76.601619</td>
    </tr>
    <tr>
      <th>6</th>
      <td>79.004813</td>
      <td>70.829664</td>
      <td>70.000000</td>
      <td>76.954832</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78.475392</td>
      <td>70.167807</td>
      <td>69.146341</td>
      <td>76.370653</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78.655488</td>
      <td>69.072933</td>
      <td>69.897436</td>
      <td>76.590159</td>
    </tr>
    <tr>
      <th>9</th>
      <td>79.361900</td>
      <td>70.870205</td>
      <td>70.705128</td>
      <td>77.348121</td>
    </tr>
    <tr>
      <th>10</th>
      <td>78.121410</td>
      <td>66.329660</td>
      <td>70.042674</td>
      <td>76.073969</td>
    </tr>
    <tr>
      <th>11</th>
      <td>78.121410</td>
      <td>68.180902</td>
      <td>69.230769</td>
      <td>76.004574</td>
    </tr>
    <tr>
      <th>12</th>
      <td>77.587331</td>
      <td>67.561939</td>
      <td>68.542510</td>
      <td>75.420805</td>
    </tr>
    <tr>
      <th>13</th>
      <td>77.767427</td>
      <td>70.396523</td>
      <td>67.804878</td>
      <td>75.605255</td>
    </tr>
    <tr>
      <th>14</th>
      <td>78.469182</td>
      <td>67.433463</td>
      <td>70.355619</td>
      <td>76.444476</td>
    </tr>
    <tr>
      <th>15</th>
      <td>80.060550</td>
      <td>73.185786</td>
      <td>71.280488</td>
      <td>78.105997</td>
    </tr>
    <tr>
      <th>16</th>
      <td>77.947524</td>
      <td>69.762373</td>
      <td>68.317073</td>
      <td>75.802372</td>
    </tr>
    <tr>
      <th>17</th>
      <td>78.127620</td>
      <td>70.638858</td>
      <td>68.292683</td>
      <td>75.991851</td>
    </tr>
    <tr>
      <th>18</th>
      <td>79.004813</td>
      <td>71.832483</td>
      <td>69.597561</td>
      <td>76.944899</td>
    </tr>
    <tr>
      <th>19</th>
      <td>77.593541</td>
      <td>69.320475</td>
      <td>67.817073</td>
      <td>75.415386</td>
    </tr>
    <tr>
      <th>20</th>
      <td>78.301506</td>
      <td>69.723035</td>
      <td>69.000000</td>
      <td>76.182710</td>
    </tr>
    <tr>
      <th>21</th>
      <td>79.358795</td>
      <td>70.210220</td>
      <td>70.910256</td>
      <td>77.365038</td>
    </tr>
    <tr>
      <th>22</th>
      <td>77.764322</td>
      <td>69.013553</td>
      <td>68.333333</td>
      <td>75.603172</td>
    </tr>
    <tr>
      <th>23</th>
      <td>77.947524</td>
      <td>68.790442</td>
      <td>68.641026</td>
      <td>75.775529</td>
    </tr>
    <tr>
      <th>24</th>
      <td>79.001708</td>
      <td>70.738193</td>
      <td>70.000000</td>
      <td>76.952980</td>
    </tr>
    <tr>
      <th>25</th>
      <td>78.296848</td>
      <td>68.566868</td>
      <td>69.581646</td>
      <td>76.210643</td>
    </tr>
    <tr>
      <th>26</th>
      <td>77.770533</td>
      <td>70.839380</td>
      <td>67.630662</td>
      <td>75.620887</td>
    </tr>
    <tr>
      <th>27</th>
      <td>77.776743</td>
      <td>70.083219</td>
      <td>67.804878</td>
      <td>75.610963</td>
    </tr>
    <tr>
      <th>28</th>
      <td>79.360348</td>
      <td>71.329208</td>
      <td>70.500000</td>
      <td>77.341818</td>
    </tr>
    <tr>
      <th>29</th>
      <td>79.888216</td>
      <td>71.289510</td>
      <td>71.576923</td>
      <td>77.950624</td>
    </tr>
    <tr>
      <th>30</th>
      <td>77.598199</td>
      <td>70.814940</td>
      <td>67.282230</td>
      <td>75.417328</td>
    </tr>
    <tr>
      <th>31</th>
      <td>78.126067</td>
      <td>69.645935</td>
      <td>68.658537</td>
      <td>75.991616</td>
    </tr>
  </tbody>
</table>
</div>




```python
samples_xgb = ['sample1_xgb', 'sample2_xgb', 'sample3_xgb', 'sample4_xgb', 'sample5_xgb', 'sample6_xgb', 'sample7_xgb', 'sample8_xgb',
              'sample9_xgb', 'sample10_xgb', 'sample11_xgb', 'sample12_xgb', 'sample13_xgb', 'sample14_xgb', 'sample15_xgb', 'sample16_xgb',
             'sample17_xgb', 'sample18_xgb', 'sample19_xgb', 'sample20_xgb', 'sample21_xgb', 'sample22_xgb', 'sample23_xgb', 'sample24_xgb', 
              'sample25_xgb', 'sample26_xgb', 'sample27_xgb', 'sample28_xgb', 'sample29_xgb', 'sample30_xgb', 'sample31_xgb', 'sample32_xgb']

samples_xgb1 = [sample1_xgb, sample2_xgb, sample3_xgb, sample4_xgb, sample5_xgb, sample6_xgb, sample7_xgb, sample8_xgb, sample9_xgb, sample10_xgb
             , sample11_xgb, sample12_xgb, sample13_xgb, sample14_xgb, sample15_xgb, sample16_xgb,sample17_xgb, sample18_xgb, sample19_xgb, sample20_xgb, 
              sample21_xgb, sample22_xgb, sample23_xgb, sample24_xgb, sample25_xgb, sample26_xgb
             , sample27_xgb, sample28_xgb, sample29_xgb, sample30_xgb, sample31_xgb, sample32_xgb]

feature_order = []
psyservice_order = []
psyservice_score = []
jobservice_order = []
jobservice_score = []
medservice_order = []
medservice_score = []

for i,(sample,sample1) in enumerate(zip(samples_xgb, samples_xgb1)):
    print('--------------sample{}_xgb--------------'.format(i+1))
    clf = xgb.XGBClassifier(random_state=42, **best_param_dict_xgb[sample])
    clf.fit(X_train_xgb[i], y_train_xgb[i])
    
    crucial_features = pd.DataFrame({'feature':list(set(sample1.columns) - set(label_features)), 'importance':clf.feature_importances_})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features.reset_index(drop = True, inplace=True)

    feature = pd.DataFrame({sample: crucial_features['feature']})
    feature_order.append(feature)
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')])
    psyservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].index.to_list())
    psyservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')])
    jobservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].index.to_list())
    jobservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')])
    medservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].index.to_list())
    medservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].to_list())

feature_order_xgb = pd.concat(feature_order, axis =1 )
service_xgb = pd.DataFrame({'psy_order': psyservice_order, 'psy_score': psyservice_score, 'job_order': jobservice_order,
                          'job_score': jobservice_score, 'med_order': medservice_order, 'med_score': medservice_score}, index = samples_xgb)
for col in service_xgb.columns:
       service_xgb[col] = service_xgb[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
```

    --------------sample1_xgb--------------
    9    0.026179
    Name: importance, dtype: float32
    16    0.023624
    Name: importance, dtype: float32
    3    0.029468
    Name: importance, dtype: float32
    --------------sample2_xgb--------------
    1    0.194265
    Name: importance, dtype: float32
    21    0.018794
    Name: importance, dtype: float32
    12    0.022913
    Name: importance, dtype: float32
    --------------sample3_xgb--------------
    3    0.031854
    Name: importance, dtype: float32
    5    0.028602
    Name: importance, dtype: float32
    16    0.022101
    Name: importance, dtype: float32
    --------------sample4_xgb--------------
    9    0.025831
    Name: importance, dtype: float32
    5    0.02892
    Name: importance, dtype: float32
    15    0.024644
    Name: importance, dtype: float32
    --------------sample5_xgb--------------
    7    0.027533
    Name: importance, dtype: float32
    16    0.02312
    Name: importance, dtype: float32
    0    0.337881
    Name: importance, dtype: float32
    --------------sample6_xgb--------------
    8    0.029429
    Name: importance, dtype: float32
    9    0.028705
    Name: importance, dtype: float32
    16    0.024544
    Name: importance, dtype: float32
    --------------sample7_xgb--------------
    8    0.029178
    Name: importance, dtype: float32
    20    0.015732
    Name: importance, dtype: float32
    11    0.027432
    Name: importance, dtype: float32
    --------------sample8_xgb--------------
    0    0.22935
    Name: importance, dtype: float32
    19    0.021391
    Name: importance, dtype: float32
    16    0.023044
    Name: importance, dtype: float32
    --------------sample9_xgb--------------
    13    0.023399
    Name: importance, dtype: float32
    15    0.020745
    Name: importance, dtype: float32
    1    0.171751
    Name: importance, dtype: float32
    --------------sample10_xgb--------------
    1    0.214907
    Name: importance, dtype: float32
    19    0.021177
    Name: importance, dtype: float32
    21    0.018467
    Name: importance, dtype: float32
    --------------sample11_xgb--------------
    19    0.024917
    Name: importance, dtype: float32
    3    0.039766
    Name: importance, dtype: float32
    14    0.026924
    Name: importance, dtype: float32
    --------------sample12_xgb--------------
    10    0.028858
    Name: importance, dtype: float32
    6    0.03326
    Name: importance, dtype: float32
    21    0.021654
    Name: importance, dtype: float32
    --------------sample13_xgb--------------
    9    0.027548
    Name: importance, dtype: float32
    13    0.024368
    Name: importance, dtype: float32
    14    0.024257
    Name: importance, dtype: float32
    --------------sample14_xgb--------------
    6    0.028509
    Name: importance, dtype: float32
    9    0.026415
    Name: importance, dtype: float32
    0    0.327278
    Name: importance, dtype: float32
    --------------sample15_xgb--------------
    8    0.027004
    Name: importance, dtype: float32
    6    0.027864
    Name: importance, dtype: float32
    19    0.022661
    Name: importance, dtype: float32
    --------------sample16_xgb--------------
    6    0.024917
    Name: importance, dtype: float32
    20    0.021202
    Name: importance, dtype: float32
    0    0.363484
    Name: importance, dtype: float32
    --------------sample17_xgb--------------
    4    0.026928
    Name: importance, dtype: float32
    26    0.015355
    Name: importance, dtype: float32
    19    0.020106
    Name: importance, dtype: float32
    --------------sample18_xgb--------------
    0    0.254535
    Name: importance, dtype: float32
    18    0.017644
    Name: importance, dtype: float32
    16    0.020052
    Name: importance, dtype: float32
    --------------sample19_xgb--------------
    8    0.02783
    Name: importance, dtype: float32
    21    0.017683
    Name: importance, dtype: float32
    19    0.018805
    Name: importance, dtype: float32
    --------------sample20_xgb--------------
    18    0.021304
    Name: importance, dtype: float32
    17    0.022305
    Name: importance, dtype: float32
    0    0.222851
    Name: importance, dtype: float32
    --------------sample21_xgb--------------
    1    0.193007
    Name: importance, dtype: float32
    21    0.018949
    Name: importance, dtype: float32
    20    0.020442
    Name: importance, dtype: float32
    --------------sample22_xgb--------------
    12    0.024999
    Name: importance, dtype: float32
    24    0.014178
    Name: importance, dtype: float32
    0    0.204265
    Name: importance, dtype: float32
    --------------sample23_xgb--------------
    20    0.021946
    Name: importance, dtype: float32
    4    0.030193
    Name: importance, dtype: float32
    21    0.021843
    Name: importance, dtype: float32
    --------------sample24_xgb--------------
    3    0.028556
    Name: importance, dtype: float32
    7    0.02698
    Name: importance, dtype: float32
    9    0.026507
    Name: importance, dtype: float32
    --------------sample25_xgb--------------
    9    0.027422
    Name: importance, dtype: float32
    7    0.029439
    Name: importance, dtype: float32
    15    0.023943
    Name: importance, dtype: float32
    --------------sample26_xgb--------------
    7    0.041958
    Name: importance, dtype: float32
    8    0.040797
    Name: importance, dtype: float32
    0    0.317682
    Name: importance, dtype: float32
    --------------sample27_xgb--------------
    0    0.220028
    Name: importance, dtype: float32
    12    0.025972
    Name: importance, dtype: float32
    14    0.025413
    Name: importance, dtype: float32
    --------------sample28_xgb--------------
    4    0.028595
    Name: importance, dtype: float32
    19    0.019224
    Name: importance, dtype: float32
    16    0.022066
    Name: importance, dtype: float32
    --------------sample29_xgb--------------
    1    0.197209
    Name: importance, dtype: float32
    23    0.017794
    Name: importance, dtype: float32
    24    0.017356
    Name: importance, dtype: float32
    --------------sample30_xgb--------------
    9    0.029676
    Name: importance, dtype: float32
    5    0.032763
    Name: importance, dtype: float32
    1    0.19409
    Name: importance, dtype: float32
    --------------sample31_xgb--------------
    14    0.023517
    Name: importance, dtype: float32
    9    0.026523
    Name: importance, dtype: float32
    24    0.018873
    Name: importance, dtype: float32
    --------------sample32_xgb--------------
    0    0.193458
    Name: importance, dtype: float32
    13    0.024052
    Name: importance, dtype: float32
    23    0.017588
    Name: importance, dtype: float32
    


```python
feature_order_xgb
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
      <th>sample1_xgb</th>
      <th>sample2_xgb</th>
      <th>sample3_xgb</th>
      <th>sample4_xgb</th>
      <th>sample5_xgb</th>
      <th>sample6_xgb</th>
      <th>sample7_xgb</th>
      <th>sample8_xgb</th>
      <th>sample9_xgb</th>
      <th>sample10_xgb</th>
      <th>...</th>
      <th>sample23_xgb</th>
      <th>sample24_xgb</th>
      <th>sample25_xgb</th>
      <th>sample26_xgb</th>
      <th>sample27_xgb</th>
      <th>sample28_xgb</th>
      <th>sample29_xgb</th>
      <th>sample30_xgb</th>
      <th>sample31_xgb</th>
      <th>sample32_xgb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>work_dummy</td>
      <td>work_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>work_dummy</td>
      <td>...</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>psy_dummy</td>
      <td>income</td>
      <td>work_dummy</td>
      <td>workday_dummy</td>
      <td>work_dummy</td>
      <td>psy_dummy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>laborcontract_dummy</td>
      <td>psy_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>emplev_cont</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>psy_dummy</td>
      <td>...</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>psy_dummy</td>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>work_dummy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>...</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>labortime_dummy</td>
      <td>jobdoctorex_dummy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>med_dummy</td>
      <td>labortime_dummy</td>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>edu_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>...</td>
      <td>satisfaction_dummy</td>
      <td>psy_dummy</td>
      <td>satisfaction_dummy</td>
      <td>cureperiod_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>workday_dummy</td>
      <td>labortime_dummy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>alcohol_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>cureperiod_dummy</td>
      <td>satisfaction_dummy</td>
      <td>...</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>satisfaction_dummy</td>
      <td>cureperiod_dummy</td>
      <td>psy_dummy</td>
      <td>edu_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>cureperiod_dummy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ralation_dummy</td>
      <td>dis_range</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>spouse_dummy</td>
      <td>edu_dummy</td>
      <td>dis_range</td>
      <td>spouse_dummy</td>
      <td>...</td>
      <td>laborchange_dummy</td>
      <td>satisfaction_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>satisfaction_dummy</td>
      <td>js_dummy</td>
      <td>ralation_dummy</td>
      <td>workday_dummy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>age_dummy</td>
      <td>edu_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_range</td>
      <td>laborcontract_dummy</td>
      <td>recoveryex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_range</td>
      <td>jobreturnopinion_dummy</td>
      <td>dis_range</td>
      <td>...</td>
      <td>jobreturnopinion_dummy</td>
      <td>income</td>
      <td>workday_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_dummy</td>
      <td>satisfaction_dummy</td>
      <td>workhour_dummy</td>
      <td>cureperiod_dummy</td>
      <td>edu_dummy</td>
      <td>workhour_dummy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sex_dummy</td>
      <td>spouse_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>recoveryex_dummy</td>
      <td>psy_dummy</td>
      <td>smoke_dummy</td>
      <td>satisfaction_dummy</td>
      <td>spouse_dummy</td>
      <td>satisfaction_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>...</td>
      <td>dis_range</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>psy_dummy</td>
      <td>dis_range</td>
      <td>workday_dummy</td>
      <td>dis_dummy</td>
      <td>ralation_dummy</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>ralation_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_range</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>con_dummy</td>
      <td>laborcontract_dummy</td>
      <td>workday_dummy</td>
      <td>...</td>
      <td>emplev_cont</td>
      <td>ralation_dummy</td>
      <td>dis_range</td>
      <td>js_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborchange_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>jobreturnopinion_dummy</td>
      <td>dis_range</td>
    </tr>
    <tr>
      <th>9</th>
      <td>psy_dummy</td>
      <td>sex_dummy</td>
      <td>laborchange_dummy</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>js_dummy</td>
      <td>recoveryex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>ralation_dummy</td>
      <td>age_dummy</td>
      <td>...</td>
      <td>workday_dummy</td>
      <td>med_dummy</td>
      <td>psy_dummy</td>
      <td>laborcontract_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
      <td>js_dummy</td>
      <td>satisfaction_dummy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>workday_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>sex_dummy</td>
      <td>labor_dummy</td>
      <td>recoveryex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>workday_dummy</td>
      <td>smoke_dummy</td>
      <td>workhour_dummy</td>
      <td>income</td>
      <td>...</td>
      <td>laborcontract_dummy</td>
      <td>dis_range</td>
      <td>laborchange_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>recoveryex_dummy</td>
      <td>alcohol_dummy</td>
      <td>con_dummy</td>
      <td>emplev_cont</td>
      <td>ralation_dummy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>spouse_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>workhour_dummy</td>
      <td>dis_range</td>
      <td>med_dummy</td>
      <td>recoveryex_dummy</td>
      <td>work_dummy</td>
      <td>smoke_dummy</td>
      <td>...</td>
      <td>ralation_dummy</td>
      <td>labortime_dummy</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>workhour_dummy</td>
      <td>sex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>alcohol_dummy</td>
      <td>age_dummy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>jobdoctorex_dummy</td>
      <td>med_dummy</td>
      <td>spouse_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>spouse_dummy</td>
      <td>ralation_dummy</td>
      <td>doctorex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>emplev_dummy</td>
      <td>...</td>
      <td>workhour_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
      <td>js_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>dis_range</td>
      <td>laborcontract_dummy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>dis_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>income</td>
      <td>edu_dummy</td>
      <td>age_dummy</td>
      <td>psy_dummy</td>
      <td>dis_dummy</td>
      <td>...</td>
      <td>spouse_dummy</td>
      <td>workday_dummy</td>
      <td>alcohol_dummy</td>
      <td>sex_dummy</td>
      <td>age_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>cureperiod_dummy</td>
      <td>workhour_dummy</td>
      <td>workhour_dummy</td>
      <td>js_dummy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>recoveryex_dummy</td>
      <td>workday_dummy</td>
      <td>workday_dummy</td>
      <td>laborchange_dummy</td>
      <td>cureperiod_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>ralation_dummy</td>
      <td>labortime_dummy</td>
      <td>recoveryex_dummy</td>
      <td>...</td>
      <td>edu_dummy</td>
      <td>con_dummy</td>
      <td>doctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>edu_dummy</td>
      <td>emplev_dummy</td>
      <td>income</td>
      <td>psy_dummy</td>
      <td>edu_dummy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>con_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>med_dummy</td>
      <td>doctorex_dummy</td>
      <td>alcohol_dummy</td>
      <td>sex_dummy</td>
      <td>workday_dummy</td>
      <td>js_dummy</td>
      <td>edu_dummy</td>
      <td>...</td>
      <td>labor_dummy</td>
      <td>recoveryex_dummy</td>
      <td>med_dummy</td>
      <td>income</td>
      <td>smoke_dummy</td>
      <td>ralation_dummy</td>
      <td>income</td>
      <td>recoveryex_dummy</td>
      <td>income</td>
      <td>recoveryex_dummy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>js_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>alcohol_dummy</td>
      <td>js_dummy</td>
      <td>med_dummy</td>
      <td>dis_dummy</td>
      <td>med_dummy</td>
      <td>con_dummy</td>
      <td>ralation_dummy</td>
      <td>...</td>
      <td>dis_dummy</td>
      <td>smoke_dummy</td>
      <td>con_dummy</td>
      <td>con_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>labor_dummy</td>
      <td>cureperiod_dummy</td>
      <td>jobreturnopinion_dummy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>jobreturnopinion_dummy</td>
      <td>doctorex_dummy</td>
      <td>edu_dummy</td>
      <td>workday_dummy</td>
      <td>dis_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>...</td>
      <td>con_dummy</td>
      <td>cureperiod_dummy</td>
      <td>emplev_cont</td>
      <td>doctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
      <td>emplev_dummy</td>
    </tr>
    <tr>
      <th>18</th>
      <td>edu_dummy</td>
      <td>con_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>labortime_dummy</td>
      <td>doctorex_dummy</td>
      <td>con_dummy</td>
      <td>emplev_dummy</td>
      <td>spouse_dummy</td>
      <td>sex_dummy</td>
      <td>...</td>
      <td>cureperiod_dummy</td>
      <td>alcohol_dummy</td>
      <td>spouse_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>alcohol_dummy</td>
      <td>ralation_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborcontract_dummy</td>
      <td>income</td>
    </tr>
    <tr>
      <th>19</th>
      <td>doctorex_dummy</td>
      <td>emplev_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>con_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>...</td>
      <td>recoveryex_dummy</td>
      <td>edu_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborchange_dummy</td>
      <td>sex_dummy</td>
      <td>js_dummy</td>
      <td>emplev_cont</td>
      <td>jobreturnopinion_dummy</td>
      <td>labor_dummy</td>
      <td>emplev_cont</td>
    </tr>
    <tr>
      <th>20</th>
      <td>dis_dummy</td>
      <td>recoveryex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>doctorex_dummy</td>
      <td>sex_dummy</td>
      <td>workday_dummy</td>
      <td>js_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>alcohol_dummy</td>
      <td>con_dummy</td>
      <td>...</td>
      <td>psy_dummy</td>
      <td>spouse_dummy</td>
      <td>income</td>
      <td>jobreturnopinion_dummy</td>
      <td>spouse_dummy</td>
      <td>work_dummy</td>
      <td>spouse_dummy</td>
      <td>smoke_dummy</td>
      <td>recoveryex_dummy</td>
      <td>spouse_dummy</td>
    </tr>
    <tr>
      <th>21</th>
      <td>alcohol_dummy</td>
      <td>js_dummy</td>
      <td>con_dummy</td>
      <td>edu_dummy</td>
      <td>con_dummy</td>
      <td>dis_dummy</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>emplev_dummy</td>
      <td>med_dummy</td>
      <td>...</td>
      <td>med_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
      <td>con_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
    </tr>
    <tr>
      <th>22</th>
      <td>laborchange_dummy</td>
      <td>alcohol_dummy</td>
      <td>labortime_dummy</td>
      <td>dis_dummy</td>
      <td>edu_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
      <td>laborchange_dummy</td>
      <td>dis_level</td>
      <td>...</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>emplev_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>sex_dummy</td>
      <td>doctorex_dummy</td>
      <td>spouse_dummy</td>
      <td>doctorex_dummy</td>
    </tr>
    <tr>
      <th>23</th>
      <td>cureperiod_dummy</td>
      <td>age_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>cureperiod_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_level</td>
      <td>edu_dummy</td>
      <td>laborchange_dummy</td>
      <td>...</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_dummy</td>
      <td>edu_dummy</td>
      <td>emplev_dummy</td>
      <td>js_dummy</td>
      <td>work_dummy</td>
      <td>con_dummy</td>
      <td>med_dummy</td>
    </tr>
    <tr>
      <th>24</th>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>work_dummy</td>
      <td>sex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>sex_dummy</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>...</td>
      <td>sex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>edu_dummy</td>
      <td>laborchange_dummy</td>
      <td>labor_dummy</td>
      <td>med_dummy</td>
      <td>sex_dummy</td>
      <td>med_dummy</td>
      <td>alcohol_dummy</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>labortime_dummy</td>
      <td>labor_dummy</td>
      <td>smoke_dummy</td>
      <td>alcohol_dummy</td>
      <td>...</td>
      <td>age_dummy</td>
      <td>labor_dummy</td>
      <td>age_dummy</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>sex_dummy</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>alcohol_dummy</td>
      <td>sex_dummy</td>
      <td>dis_level</td>
      <td>doctorex_dummy</td>
      <td>...</td>
      <td>smoke_dummy</td>
      <td>work_dummy</td>
      <td>sex_dummy</td>
      <td>ralation_dummy</td>
      <td>con_dummy</td>
      <td>dis_level</td>
      <td>age_dummy</td>
      <td>edu_dummy</td>
      <td>sex_dummy</td>
      <td>con_dummy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>labor_dummy</td>
      <td>labor_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>smoke_dummy</td>
      <td>laborchange_dummy</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>dis_level</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>labor_dummy</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>dis_level</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 32 columns</p>
</div>




```python
service_xgb
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
      <th>psy_order</th>
      <th>psy_score</th>
      <th>job_order</th>
      <th>job_score</th>
      <th>med_order</th>
      <th>med_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample1_xgb</th>
      <td>9.0</td>
      <td>0.026179</td>
      <td>16.0</td>
      <td>0.023624</td>
      <td>3.0</td>
      <td>0.029468</td>
    </tr>
    <tr>
      <th>sample2_xgb</th>
      <td>1.0</td>
      <td>0.194265</td>
      <td>21.0</td>
      <td>0.018794</td>
      <td>12.0</td>
      <td>0.022913</td>
    </tr>
    <tr>
      <th>sample3_xgb</th>
      <td>3.0</td>
      <td>0.031854</td>
      <td>5.0</td>
      <td>0.028602</td>
      <td>16.0</td>
      <td>0.022101</td>
    </tr>
    <tr>
      <th>sample4_xgb</th>
      <td>9.0</td>
      <td>0.025831</td>
      <td>5.0</td>
      <td>0.028920</td>
      <td>15.0</td>
      <td>0.024644</td>
    </tr>
    <tr>
      <th>sample5_xgb</th>
      <td>7.0</td>
      <td>0.027533</td>
      <td>16.0</td>
      <td>0.023120</td>
      <td>0.0</td>
      <td>0.337881</td>
    </tr>
    <tr>
      <th>sample6_xgb</th>
      <td>8.0</td>
      <td>0.029429</td>
      <td>9.0</td>
      <td>0.028705</td>
      <td>16.0</td>
      <td>0.024544</td>
    </tr>
    <tr>
      <th>sample7_xgb</th>
      <td>8.0</td>
      <td>0.029178</td>
      <td>20.0</td>
      <td>0.015732</td>
      <td>11.0</td>
      <td>0.027432</td>
    </tr>
    <tr>
      <th>sample8_xgb</th>
      <td>0.0</td>
      <td>0.229350</td>
      <td>19.0</td>
      <td>0.021391</td>
      <td>16.0</td>
      <td>0.023044</td>
    </tr>
    <tr>
      <th>sample9_xgb</th>
      <td>13.0</td>
      <td>0.023399</td>
      <td>15.0</td>
      <td>0.020745</td>
      <td>1.0</td>
      <td>0.171751</td>
    </tr>
    <tr>
      <th>sample10_xgb</th>
      <td>1.0</td>
      <td>0.214907</td>
      <td>19.0</td>
      <td>0.021177</td>
      <td>21.0</td>
      <td>0.018467</td>
    </tr>
    <tr>
      <th>sample11_xgb</th>
      <td>19.0</td>
      <td>0.024917</td>
      <td>3.0</td>
      <td>0.039766</td>
      <td>14.0</td>
      <td>0.026924</td>
    </tr>
    <tr>
      <th>sample12_xgb</th>
      <td>10.0</td>
      <td>0.028858</td>
      <td>6.0</td>
      <td>0.033260</td>
      <td>21.0</td>
      <td>0.021654</td>
    </tr>
    <tr>
      <th>sample13_xgb</th>
      <td>9.0</td>
      <td>0.027548</td>
      <td>13.0</td>
      <td>0.024368</td>
      <td>14.0</td>
      <td>0.024257</td>
    </tr>
    <tr>
      <th>sample14_xgb</th>
      <td>6.0</td>
      <td>0.028509</td>
      <td>9.0</td>
      <td>0.026415</td>
      <td>0.0</td>
      <td>0.327278</td>
    </tr>
    <tr>
      <th>sample15_xgb</th>
      <td>8.0</td>
      <td>0.027004</td>
      <td>6.0</td>
      <td>0.027864</td>
      <td>19.0</td>
      <td>0.022661</td>
    </tr>
    <tr>
      <th>sample16_xgb</th>
      <td>6.0</td>
      <td>0.024917</td>
      <td>20.0</td>
      <td>0.021202</td>
      <td>0.0</td>
      <td>0.363484</td>
    </tr>
    <tr>
      <th>sample17_xgb</th>
      <td>4.0</td>
      <td>0.026928</td>
      <td>26.0</td>
      <td>0.015355</td>
      <td>19.0</td>
      <td>0.020106</td>
    </tr>
    <tr>
      <th>sample18_xgb</th>
      <td>0.0</td>
      <td>0.254535</td>
      <td>18.0</td>
      <td>0.017644</td>
      <td>16.0</td>
      <td>0.020052</td>
    </tr>
    <tr>
      <th>sample19_xgb</th>
      <td>8.0</td>
      <td>0.027830</td>
      <td>21.0</td>
      <td>0.017683</td>
      <td>19.0</td>
      <td>0.018805</td>
    </tr>
    <tr>
      <th>sample20_xgb</th>
      <td>18.0</td>
      <td>0.021304</td>
      <td>17.0</td>
      <td>0.022305</td>
      <td>0.0</td>
      <td>0.222851</td>
    </tr>
    <tr>
      <th>sample21_xgb</th>
      <td>1.0</td>
      <td>0.193007</td>
      <td>21.0</td>
      <td>0.018949</td>
      <td>20.0</td>
      <td>0.020442</td>
    </tr>
    <tr>
      <th>sample22_xgb</th>
      <td>12.0</td>
      <td>0.024999</td>
      <td>24.0</td>
      <td>0.014178</td>
      <td>0.0</td>
      <td>0.204265</td>
    </tr>
    <tr>
      <th>sample23_xgb</th>
      <td>20.0</td>
      <td>0.021946</td>
      <td>4.0</td>
      <td>0.030193</td>
      <td>21.0</td>
      <td>0.021843</td>
    </tr>
    <tr>
      <th>sample24_xgb</th>
      <td>3.0</td>
      <td>0.028556</td>
      <td>7.0</td>
      <td>0.026980</td>
      <td>9.0</td>
      <td>0.026507</td>
    </tr>
    <tr>
      <th>sample25_xgb</th>
      <td>9.0</td>
      <td>0.027422</td>
      <td>7.0</td>
      <td>0.029439</td>
      <td>15.0</td>
      <td>0.023943</td>
    </tr>
    <tr>
      <th>sample26_xgb</th>
      <td>7.0</td>
      <td>0.041958</td>
      <td>8.0</td>
      <td>0.040797</td>
      <td>0.0</td>
      <td>0.317682</td>
    </tr>
    <tr>
      <th>sample27_xgb</th>
      <td>0.0</td>
      <td>0.220028</td>
      <td>12.0</td>
      <td>0.025972</td>
      <td>14.0</td>
      <td>0.025413</td>
    </tr>
    <tr>
      <th>sample28_xgb</th>
      <td>4.0</td>
      <td>0.028595</td>
      <td>19.0</td>
      <td>0.019224</td>
      <td>16.0</td>
      <td>0.022066</td>
    </tr>
    <tr>
      <th>sample29_xgb</th>
      <td>1.0</td>
      <td>0.197209</td>
      <td>23.0</td>
      <td>0.017794</td>
      <td>24.0</td>
      <td>0.017356</td>
    </tr>
    <tr>
      <th>sample30_xgb</th>
      <td>9.0</td>
      <td>0.029676</td>
      <td>5.0</td>
      <td>0.032763</td>
      <td>1.0</td>
      <td>0.194090</td>
    </tr>
    <tr>
      <th>sample31_xgb</th>
      <td>14.0</td>
      <td>0.023517</td>
      <td>9.0</td>
      <td>0.026523</td>
      <td>24.0</td>
      <td>0.018873</td>
    </tr>
    <tr>
      <th>sample32_xgb</th>
      <td>0.0</td>
      <td>0.193458</td>
      <td>13.0</td>
      <td>0.024052</td>
      <td>23.0</td>
      <td>0.017588</td>
    </tr>
  </tbody>
</table>
</div>



## 2) Neural-Network
- Multi-layer Perceptron classifier

### Multi-layer Perceptron classifier

#### before parameter tuning


```python
mlp_class = MLPClassifier(max_iter=100)
```


```python
def plot_feature_importances_mlp(model):
    n_features = cx.columns
    top_features = round(len(n_features)/2)
    model.fit(cx_train, cy_train)
    coef = []
    for i in range(len(model.coefs_[0])):
        coef.append(model.coefs_[0][i].mean())
    
    top_coefficients = np.argsort(coef).tolist()
    #top_negative_coefficients = np.argsort(coef)
    #top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in [coef[i] for i in top_coefficients]]
    plt.bar(range(len(n_features)), [coef[i] for i in top_coefficients], color=colors)
    feature_names = np.array(n_features)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()
    
    crucial_features = pd.DataFrame({'feature':n_features, 'importance':coef})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features = crucial_features[:7]
    return crucial_features
```


```python
temp_mlp = plot_feature_importances_mlp(mlp_class)
temp_mlp
```


![png](https://drive.google.com/uc?export=view&id=15rvnQ-uAsxpQGSYnZsGqb2WRaaZTDDVn)





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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>dis_level</td>
      <td>0.029991</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dis_range</td>
      <td>0.016137</td>
    </tr>
    <tr>
      <th>20</th>
      <td>work_dummy</td>
      <td>0.013527</td>
    </tr>
    <tr>
      <th>21</th>
      <td>workday_dummy</td>
      <td>0.011631</td>
    </tr>
    <tr>
      <th>10</th>
      <td>jobreturnopinion_dummy</td>
      <td>0.009582</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cureperiod_dummy</td>
      <td>0.007433</td>
    </tr>
    <tr>
      <th>16</th>
      <td>satisfaction_dummy</td>
      <td>0.007408</td>
    </tr>
  </tbody>
</table>
</div>




```python
#selected_variables_mlp = temp_mlp["feature"].to_list() + ["rtor_dummy"]
selected_variables_mlp = sorted(list(set(common_sample.columns) - set(jobcondition_features)))
```


```python
sample1_mlp = df[["js_dummy", "psy_dummy", "med_dummy", *selected_variables_mlp]] # 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample2_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", *selected_variables_mlp]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample3_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", *selected_variables_mlp]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample4_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", *selected_variables_mlp]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample5_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample6_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "income", *selected_variables_mlp]] # 임금/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample7_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", *selected_variables_mlp]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample8_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", *selected_variables_mlp]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample9_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", *selected_variables_mlp]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample10_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample11_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample12_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample13_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample14_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample15_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample16_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample17_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", *selected_variables_mlp]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample18_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", *selected_variables_mlp]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample19_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "income", *selected_variables_mlp]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample20_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample21_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample22_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample23_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample24_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample25_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample26_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample27_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", *selected_variables_mlp]] # 종사상지위/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample28_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "income", *selected_variables_mlp]] # 직종/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample29_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "workhour_dummy", "income", *selected_variables_mlp]] # 산업장규모/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample30_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "labor_dummy", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample31_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스

sample32_mlp = df[["js_dummy", "psy_dummy", "med_dummy", "emplev_dummy", "emplev_cont", "labor_dummy", "workhour_dummy", "income", *selected_variables_mlp]] # 하루평균근무시간/ 직업 재활 서비스 + 심리 치료 서비스 + 의료 서비스
```


```python
samples_mlp = [sample1_mlp, sample2_mlp, sample3_mlp, sample4_mlp, sample5_mlp, sample6_mlp, sample7_mlp, sample8_mlp, sample9_mlp, sample10_mlp
             , sample11_mlp, sample12_mlp, sample13_mlp, sample14_mlp, sample15_mlp, sample16_mlp,sample17_mlp, sample18_mlp, sample19_mlp, sample20_mlp, 
              sample21_mlp, sample22_mlp, sample23_mlp, sample24_mlp, sample25_mlp, sample26_mlp
             , sample27_mlp, sample28_mlp, sample29_mlp, sample30_mlp, sample31_mlp, sample32_mlp]

features_mlp=[]
X_mlp = []
y_mlp = []
X_train_mlp = []
X_test_mlp = []
y_train_mlp = []
y_test_mlp = []

for i,sample in enumerate(samples_mlp):
    features_mlp.append(sorted(list(set(sample.columns) - set(label_features))))
    X_mlp.append(sample[features_mlp[i]])
    y_mlp.append(sample[label_features])
    X_train_mlp.append(train_test_split(X_mlp[i], y_mlp[i], test_size=0.3, random_state=42)[0])
    X_test_mlp.append(train_test_split(X_mlp[i], y_mlp[i], test_size=0.3, random_state=42)[1])
    y_train_mlp.append(train_test_split(X_mlp[i], y_mlp[i], test_size=0.3, random_state=42)[2])
    y_test_mlp.append(train_test_split(X_mlp[i], y_mlp[i], test_size=0.3, random_state=42)[3])
    X_train_mlp[i] = X_train_mlp[i].reset_index(drop=True)
    X_test_mlp[i] = X_test_mlp[i].reset_index(drop=True)                   
    y_train_mlp[i] = y_train_mlp[i].reset_index(drop=True)
    y_test_mlp[i] = y_test_mlp[i].reset_index(drop=True)
```


```python
for i in range(2):
    print('---- sample{}_mlp ----'.format(i+1))
    accuracy = cross_val_score(mlp_class, X_train_mlp[i], y_train_mlp[i], scoring='accuracy', cv = 10).mean() * 100
    print("Accuracy of MLP is: " , accuracy)
```

    ---- sample1_mlp ----
    Accuracy of MLP is:  76.89393939393939
    ---- sample2_mlp ----
    Accuracy of MLP is:  77.80303030303031
    

#### randomsearch parameter tuning


```python
from scipy.stats import randint
from random import *
mlp_params = {'hidden_layer_sizes': [randint(100,600), randint(100,600), randint(100,600)],
              'activation': ['tanh', 'relu', 'logistic'],
              'solver': ['sgd', 'adam', 'lbfgs'],
              'alpha': np.arange(0.0001, 0.9),
              'learning_rate': ['constant','adaptive']}
params_list = [mlp_params]

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=5, scoring='accuracy', random_state=42)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
best_param_dict_mlp = dict()
samples_mlp = ['sample1_mlp', 'sample2_mlp', 'sample3_mlp', 'sample4_mlp', 'sample5_mlp', 'sample6_mlp', 'sample7_mlp', 'sample8_mlp',
              'sample9_mlp', 'sample10_mlp', 'sample11_mlp', 'sample12_mlp', 'sample13_mlp', 'sample14_mlp', 'sample15_mlp', 'sample16_mlp',
             'sample17_mlp', 'sample18_mlp', 'sample19_mlp', 'sample20_mlp', 'sample21_mlp', 'sample22_mlp', 'sample23_mlp', 'sample24_mlp', 
              'sample25_mlp', 'sample26_mlp', 'sample27_mlp', 'sample28_mlp', 'sample29_mlp', 'sample30_mlp', 'sample31_mlp', 'sample32_mlp']

samples_mlp1 = [sample1_mlp, sample2_mlp, sample3_mlp, sample4_mlp, sample5_mlp, sample6_mlp, sample7_mlp, sample8_mlp, sample9_mlp, sample10_mlp
             , sample11_mlp, sample12_mlp, sample13_mlp, sample14_mlp, sample15_mlp, sample16_mlp,sample17_mlp, sample18_mlp, sample19_mlp, sample20_mlp, 
              sample21_mlp, sample22_mlp, sample23_mlp, sample24_mlp, sample25_mlp, sample26_mlp
             , sample27_mlp, sample28_mlp, sample29_mlp, sample30_mlp, sample31_mlp, sample32_mlp]
for i,sample in enumerate(samples_mlp):
    print('---sample{}_mlp---'.format(i+1))
    best_params = hypertuning_rscv(mlp_class, mlp_params, 30, X_train_mlp[i], y_train_mlp[i])
    best_param_dict_mlp[sample] = best_params[0]
    print('best_params : ', best_params[0])
```

    ---sample1_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'tanh'}
    ---sample2_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample3_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample4_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample5_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample6_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample7_mlp---
    best_params :  {'solver': 'lbfgs', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 439, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample8_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample9_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample10_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample11_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample12_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample13_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample14_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample15_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample16_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample17_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample18_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample19_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample20_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample21_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample22_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample23_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample24_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'tanh'}
    ---sample25_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'tanh'}
    ---sample26_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'tanh'}
    ---sample27_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample28_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample29_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample30_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample31_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 333, 'alpha': 0.0001, 'activation': 'logistic'}
    ---sample32_mlp---
    best_params :  {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': 529, 'alpha': 0.0001, 'activation': 'logistic'}
    


```python
samples_mlp = ['sample1_mlp', 'sample2_mlp', 'sample3_mlp', 'sample4_mlp', 'sample5_mlp', 'sample6_mlp', 'sample7_mlp', 'sample8_mlp',
              'sample9_mlp', 'sample10_mlp', 'sample11_mlp', 'sample12_mlp', 'sample13_mlp', 'sample14_mlp', 'sample15_mlp', 'sample16_mlp',
             'sample17_mlp', 'sample18_mlp', 'sample19_mlp', 'sample20_mlp', 'sample21_mlp', 'sample22_mlp', 'sample23_mlp', 'sample24_mlp', 
              'sample25_mlp', 'sample26_mlp', 'sample27_mlp', 'sample28_mlp', 'sample29_mlp', 'sample30_mlp', 'sample31_mlp', 'sample32_mlp']

accuracy = []
precision = []
sensitivity = []
Auc = []

for i,sample in enumerate(samples_mlp):
    print('--------------sample{}_mlp--------------'.format(i+1))
    clf = MLPClassifier(random_state=42, **best_param_dict_mlp[sample])
    clf.fit(X_train_mlp[i], y_train_mlp[i])
    y_pred_mlp = clf.predict(X_test_mlp[i])
    #print(y_pred_mlp)
    #y_pred_proba_mlp = clf.predict_proba(X_test_mlp[i])
    print("accuracy_score: {}".format( cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='accuracy', cv = 5).mean() * 100))
    accuracy.append(cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='accuracy', cv = 5).mean() * 100)
    print("precision_score: {}".format( cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='precision', cv = 5).mean() * 100))
    precision.append(cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='precision', cv = 5).mean() * 100)
    print("sensitivity_score: {}".format( cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='recall', cv = 5).mean() * 100))
    sensitivity.append(cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='recall', cv = 5).mean() * 100)
    try:
        print("AUC: Area Under Curve: {}".format(cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='roc_auc', cv = 5).mean() * 100))
        Auc.append(cross_val_score(mlp_class, y_test_mlp[i], y_pred_mlp, scoring='roc_auc', cv = 5).mean() * 100)
    except ValueError:
        pass
    
score_mlp = pd.DataFrame({'accuracy':accuracy, 'precision':precision, 'sensitivity': sensitivity, 'Auc': Auc})
    #y_score = clf.decision_path(X_test[i])
    #precision, recall, _ = precision_recall_curve(y_test[i], y_score[1])
    #fpr, tpr, _ = roc_curve(y_test[i], y_score)
    
    #f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.set_size_inches((8, 4)) 
    #axes[0].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #axes[0].set_title('Recall-Precision Curve')
    
    #axes[1].plot(fpr, tpr)
    #axes[1].plot([0, 1], [0, 1], linestyle='--')
    #axes[1].set_title('ROC curve')
```

    --------------sample1_mlp--------------
    accuracy_score: 80.4145319049837
    precision_score: 62.155298269808966
    sensitivity_score: 77.78225806451613
    AUC: Area Under Curve: 79.6414677856456
    --------------sample2_mlp--------------
    accuracy_score: 76.3654712001242
    precision_score: 77.1516641622686
    sensitivity_score: 63.50340136054422
    AUC: Area Under Curve: 74.66841888270459
    --------------sample3_mlp--------------
    accuracy_score: 77.2411116286291
    precision_score: 74.35446065772481
    sensitivity_score: 65.61616161616162
    AUC: Area Under Curve: 75.26243681486649
    --------------sample4_mlp--------------
    accuracy_score: 77.41344511721782
    precision_score: 77.31047448801277
    sensitivity_score: 65.1063829787234
    AUC: Area Under Curve: 75.56472954377051
    --------------sample5_mlp--------------
    accuracy_score: 77.5904362676603
    precision_score: 75.28932178932179
    sensitivity_score: 65.93236714975845
    AUC: Area Under Curve: 75.63668655868145
    --------------sample6_mlp--------------
    accuracy_score: 75.12342803912435
    precision_score: 74.44672394473868
    sensitivity_score: 62.28723404255319
    AUC: Area Under Curve: 73.2284539584476
    --------------sample7_mlp--------------
    accuracy_score: 75.1311908088806
    precision_score: 69.67937736022843
    sensitivity_score: 63.424947145877375
    AUC: Area Under Curve: 72.86861975234068
    --------------sample8_mlp--------------
    accuracy_score: 78.11985716503649
    precision_score: 68.74804561646665
    sensitivity_score: 69.01282051282051
    AUC: Area Under Curve: 75.67515592515592
    --------------sample9_mlp--------------
    accuracy_score: 78.12140971898774
    precision_score: 69.80507495141642
    sensitivity_score: 68.67073170731707
    AUC: Area Under Curve: 76.50086011504321
    --------------sample10_mlp--------------
    accuracy_score: 73.36438441235833
    precision_score: 82.88945937783147
    sensitivity_score: 58.57142857142856
    AUC: Area Under Curve: 73.2569570477919
    --------------sample11_mlp--------------
    accuracy_score: 77.41499767116908
    precision_score: 75.31229235880399
    sensitivity_score: 65.61352657004831
    AUC: Area Under Curve: 75.33617504973003
    --------------sample12_mlp--------------
    accuracy_score: 77.41965533302282
    precision_score: 77.24852542643073
    sensitivity_score: 65.1063829787234
    AUC: Area Under Curve: 75.65143335546638
    --------------sample13_mlp--------------
    accuracy_score: 77.9459711224965
    precision_score: 72.80487804878048
    sensitivity_score: 67.27574750830566
    AUC: Area Under Curve: 76.07369800196527
    --------------sample14_mlp--------------
    accuracy_score: 77.59198882161155
    precision_score: 77.35918003565064
    sensitivity_score: 65.37465309898242
    AUC: Area Under Curve: 75.97629755519769
    --------------sample15_mlp--------------
    accuracy_score: 77.24266418258034
    precision_score: 72.18750808224492
    sensitivity_score: 66.19450317124735
    AUC: Area Under Curve: 75.1287428588444
    --------------sample16_mlp--------------
    accuracy_score: 78.65704083216892
    precision_score: 71.81571529245947
    sensitivity_score: 68.93147502903601
    AUC: Area Under Curve: 76.57380448559867
    --------------sample17_mlp--------------
    accuracy_score: 78.11985716503649
    precision_score: 71.37708091366628
    sensitivity_score: 68.0952380952381
    AUC: Area Under Curve: 76.22580332946185
    --------------sample18_mlp--------------
    accuracy_score: 77.76742741810278
    precision_score: 71.28334966224926
    sensitivity_score: 67.44483159117306
    AUC: Area Under Curve: 75.67072489135353
    --------------sample19_mlp--------------
    accuracy_score: 79.18180406769135
    precision_score: 63.96098638203902
    sensitivity_score: 73.3781512605042
    AUC: Area Under Curve: 77.53038547454035
    --------------sample20_mlp--------------
    accuracy_score: 78.82782176680641
    precision_score: 68.78349118661596
    sensitivity_score: 70.45883940620783
    AUC: Area Under Curve: 76.79878907247327
    --------------sample21_mlp--------------
    accuracy_score: 79.36190032603633
    precision_score: 70.67555582189729
    sensitivity_score: 70.7051282051282
    AUC: Area Under Curve: 77.39236417661076
    --------------sample22_mlp--------------
    accuracy_score: 77.5966464834653
    precision_score: 73.33225464932781
    sensitivity_score: 66.50105708245243
    AUC: Area Under Curve: 75.85503744654402
    --------------sample23_mlp--------------
    accuracy_score: 79.00326036329763
    precision_score: 68.88231148696265
    sensitivity_score: 70.80971659919028
    AUC: Area Under Curve: 76.914889788574
    --------------sample24_mlp--------------
    accuracy_score: 79.00481291724887
    precision_score: 65.13694065170317
    sensitivity_score: 72.42857142857143
    AUC: Area Under Curve: 77.24408924408924
    --------------sample25_mlp--------------
    accuracy_score: 80.94861046421362
    precision_score: 68.96642627555842
    sensitivity_score: 75.13513513513514
    AUC: Area Under Curve: 79.25374925374926
    --------------sample26_mlp--------------
    accuracy_score: 83.59882005899705
    precision_score: 0.0
    sensitivity_score: 0.0
    AUC: Area Under Curve: 79.15228253537914
    --------------sample27_mlp--------------
    accuracy_score: 77.94441856854526
    precision_score: 71.28334966224926
    sensitivity_score: 67.75842044134727
    AUC: Area Under Curve: 75.81993314902967
    --------------sample28_mlp--------------
    accuracy_score: 76.18226983387673
    precision_score: 73.16240186971893
    sensitivity_score: 64.15458937198068
    AUC: Area Under Curve: 74.25547030406365
    --------------sample29_mlp--------------
    accuracy_score: 78.6508306163639
    precision_score: 67.80593169135427
    sensitivity_score: 70.52631578947368
    AUC: Area Under Curve: 76.60046168051709
    --------------sample30_mlp--------------
    accuracy_score: 78.83403198261139
    precision_score: 66.64998358702272
    sensitivity_score: 71.35135135135135
    AUC: Area Under Curve: 77.02827400195821
    --------------sample31_mlp--------------
    accuracy_score: 75.1280857009781
    precision_score: 74.92412603404823
    sensitivity_score: 62.1631205673759
    AUC: Area Under Curve: 73.54867827208254
    --------------sample32_mlp--------------
    accuracy_score: 79.00791802515137
    precision_score: 66.18880812584727
    sensitivity_score: 71.98198198198197
    AUC: Area Under Curve: 77.15621465621466
    


```python
score_mlp
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
      <th>accuracy</th>
      <th>precision</th>
      <th>sensitivity</th>
      <th>Auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80.414532</td>
      <td>62.155298</td>
      <td>77.782258</td>
      <td>79.618602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.365471</td>
      <td>77.151664</td>
      <td>63.503401</td>
      <td>74.921621</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.241112</td>
      <td>74.354461</td>
      <td>65.616162</td>
      <td>75.224961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77.413445</td>
      <td>77.310474</td>
      <td>65.106383</td>
      <td>75.615154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77.590436</td>
      <td>75.289322</td>
      <td>65.932367</td>
      <td>75.586625</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75.123428</td>
      <td>74.446724</td>
      <td>62.287234</td>
      <td>73.460562</td>
    </tr>
    <tr>
      <th>6</th>
      <td>75.131191</td>
      <td>69.679377</td>
      <td>63.424947</td>
      <td>72.925098</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78.119857</td>
      <td>68.748046</td>
      <td>69.012821</td>
      <td>75.956687</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78.121410</td>
      <td>69.805075</td>
      <td>68.670732</td>
      <td>75.734292</td>
    </tr>
    <tr>
      <th>9</th>
      <td>73.364384</td>
      <td>82.889459</td>
      <td>58.571429</td>
      <td>73.256957</td>
    </tr>
    <tr>
      <th>10</th>
      <td>77.414998</td>
      <td>75.312292</td>
      <td>65.613527</td>
      <td>75.453822</td>
    </tr>
    <tr>
      <th>11</th>
      <td>77.419655</td>
      <td>77.248525</td>
      <td>65.106383</td>
      <td>75.549814</td>
    </tr>
    <tr>
      <th>12</th>
      <td>77.945971</td>
      <td>72.804878</td>
      <td>67.275748</td>
      <td>76.056619</td>
    </tr>
    <tr>
      <th>13</th>
      <td>77.591989</td>
      <td>77.359180</td>
      <td>65.374653</td>
      <td>75.776427</td>
    </tr>
    <tr>
      <th>14</th>
      <td>77.242664</td>
      <td>72.187508</td>
      <td>66.194503</td>
      <td>75.075942</td>
    </tr>
    <tr>
      <th>15</th>
      <td>78.657041</td>
      <td>71.815715</td>
      <td>68.931475</td>
      <td>76.862785</td>
    </tr>
    <tr>
      <th>16</th>
      <td>78.119857</td>
      <td>71.377081</td>
      <td>68.095238</td>
      <td>75.853175</td>
    </tr>
    <tr>
      <th>17</th>
      <td>77.767427</td>
      <td>71.283350</td>
      <td>67.444832</td>
      <td>75.627456</td>
    </tr>
    <tr>
      <th>18</th>
      <td>79.181804</td>
      <td>63.960986</td>
      <td>73.378151</td>
      <td>77.489326</td>
    </tr>
    <tr>
      <th>19</th>
      <td>78.827822</td>
      <td>68.783491</td>
      <td>70.458839</td>
      <td>76.571888</td>
    </tr>
    <tr>
      <th>20</th>
      <td>79.361900</td>
      <td>70.675556</td>
      <td>70.705128</td>
      <td>77.420084</td>
    </tr>
    <tr>
      <th>21</th>
      <td>77.596646</td>
      <td>73.332255</td>
      <td>66.501057</td>
      <td>75.688417</td>
    </tr>
    <tr>
      <th>22</th>
      <td>79.003260</td>
      <td>68.882311</td>
      <td>70.809717</td>
      <td>76.797841</td>
    </tr>
    <tr>
      <th>23</th>
      <td>79.004813</td>
      <td>65.136941</td>
      <td>72.428571</td>
      <td>77.362933</td>
    </tr>
    <tr>
      <th>24</th>
      <td>80.948610</td>
      <td>68.966426</td>
      <td>75.135135</td>
      <td>78.989986</td>
    </tr>
    <tr>
      <th>25</th>
      <td>83.598820</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>79.086108</td>
    </tr>
    <tr>
      <th>26</th>
      <td>77.944419</td>
      <td>71.283350</td>
      <td>67.758420</td>
      <td>75.931292</td>
    </tr>
    <tr>
      <th>27</th>
      <td>76.182270</td>
      <td>73.162402</td>
      <td>64.154589</td>
      <td>74.159562</td>
    </tr>
    <tr>
      <th>28</th>
      <td>78.650831</td>
      <td>67.805932</td>
      <td>70.526316</td>
      <td>76.400923</td>
    </tr>
    <tr>
      <th>29</th>
      <td>78.834032</td>
      <td>66.649984</td>
      <td>71.351351</td>
      <td>76.941217</td>
    </tr>
    <tr>
      <th>30</th>
      <td>75.128086</td>
      <td>74.924126</td>
      <td>62.163121</td>
      <td>73.606235</td>
    </tr>
    <tr>
      <th>31</th>
      <td>79.007918</td>
      <td>66.188808</td>
      <td>71.981982</td>
      <td>77.182442</td>
    </tr>
  </tbody>
</table>
</div>




```python
samples_mlp = ['sample1_mlp', 'sample2_mlp', 'sample3_mlp', 'sample4_mlp', 'sample5_mlp', 'sample6_mlp', 'sample7_mlp', 'sample8_mlp',
              'sample9_mlp', 'sample10_mlp', 'sample11_mlp', 'sample12_mlp', 'sample13_mlp', 'sample14_mlp', 'sample15_mlp', 'sample16_mlp',
             'sample17_mlp', 'sample18_mlp', 'sample19_mlp', 'sample20_mlp', 'sample21_mlp', 'sample22_mlp', 'sample23_mlp', 'sample24_mlp', 
              'sample25_mlp', 'sample26_mlp', 'sample27_mlp', 'sample28_mlp', 'sample29_mlp', 'sample30_mlp', 'sample31_mlp', 'sample32_mlp']

samples_mlp1 = [sample1_mlp, sample2_mlp, sample3_mlp, sample4_mlp, sample5_mlp, sample6_mlp, sample7_mlp, sample8_mlp, sample9_mlp, sample10_mlp
             , sample11_mlp, sample12_mlp, sample13_mlp, sample14_mlp, sample15_mlp, sample16_mlp,sample17_mlp, sample18_mlp, sample19_mlp, sample20_mlp, 
              sample21_mlp, sample22_mlp, sample23_mlp, sample24_mlp, sample25_mlp, sample26_mlp
             , sample27_mlp, sample28_mlp, sample29_mlp, sample30_mlp, sample31_mlp, sample32_mlp]

feature_order = []
psyservice_order = []
psyservice_score = []
jobservice_order = []
jobservice_score = []
medservice_order = []
medservice_score = []

for i,(sample,sample1) in enumerate(zip(samples_mlp, samples_mlp1)):
    print('--------------sample{}_mlp--------------'.format(i+1))
    clf = MLPClassifier(random_state=42, **best_param_dict_mlp[sample])
    clf.fit(X_train_mlp[i], y_train_mlp[i])
    
    coef = []
    for i in range(len(clf.coefs_[0])):
        coef.append(clf.coefs_[0][i].mean())
        
    crucial_features = pd.DataFrame({'feature':list(set(sample1.columns) - set(label_features)), 'importance':coef})
    crucial_features.sort_values(by=['importance'], ascending=False, inplace=True)
    crucial_features.reset_index(drop = True, inplace=True)

    feature = pd.DataFrame({sample: crucial_features['feature']})
    feature_order.append(feature)
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')])
    psyservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].index.to_list())
    psyservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'psy_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')])
    jobservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].index.to_list())
    jobservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'js_dummy')].to_list())
    
    print(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')])
    medservice_order.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].index.to_list())
    medservice_score.append(crucial_features['importance'].iloc[np.where(crucial_features['feature'] == 'med_dummy')].to_list())

feature_order_mlp = pd.concat(feature_order, axis =1 )
service_mlp = pd.DataFrame({'psy_order': psyservice_order, 'psy_score': psyservice_score, 'job_order': jobservice_order,
                          'job_score': jobservice_score, 'med_order': medservice_order, 'med_score': medservice_score}, index = samples_mlp)
for col in service_mlp.columns:
       service_mlp[col] = service_mlp[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
```

    --------------sample1_mlp--------------
    9   -0.000761
    Name: importance, dtype: float64
    6    0.00031
    Name: importance, dtype: float64
    23   -0.003573
    Name: importance, dtype: float64
    --------------sample2_mlp--------------
    24   -0.01289
    Name: importance, dtype: float64
    8   -0.000526
    Name: importance, dtype: float64
    18   -0.003148
    Name: importance, dtype: float64
    --------------sample3_mlp--------------
    20   -0.005548
    Name: importance, dtype: float64
    4    0.001631
    Name: importance, dtype: float64
    0    0.007597
    Name: importance, dtype: float64
    --------------sample4_mlp--------------
    19   -0.004696
    Name: importance, dtype: float64
    8   -0.001182
    Name: importance, dtype: float64
    18   -0.00452
    Name: importance, dtype: float64
    --------------sample5_mlp--------------
    13   -0.002494
    Name: importance, dtype: float64
    4    0.002015
    Name: importance, dtype: float64
    25   -0.0391
    Name: importance, dtype: float64
    --------------sample6_mlp--------------
    13    0.001781
    Name: importance, dtype: float64
    8    0.00714
    Name: importance, dtype: float64
    1    0.023186
    Name: importance, dtype: float64
    --------------sample7_mlp--------------
    16   -0.051726
    Name: importance, dtype: float64
    13   -0.008914
    Name: importance, dtype: float64
    3    0.053809
    Name: importance, dtype: float64
    --------------sample8_mlp--------------
    25   -0.015022
    Name: importance, dtype: float64
    9   -0.002958
    Name: importance, dtype: float64
    21   -0.006561
    Name: importance, dtype: float64
    --------------sample9_mlp--------------
    16   -0.003347
    Name: importance, dtype: float64
    9   -0.001821
    Name: importance, dtype: float64
    26   -0.019942
    Name: importance, dtype: float64
    --------------sample10_mlp--------------
    26   -0.001864
    Name: importance, dtype: float64
    23   -0.000397
    Name: importance, dtype: float64
    8    0.003969
    Name: importance, dtype: float64
    --------------sample11_mlp--------------
    19   -0.005503
    Name: importance, dtype: float64
    4    0.001209
    Name: importance, dtype: float64
    0    0.011734
    Name: importance, dtype: float64
    --------------sample12_mlp--------------
    18   -0.004536
    Name: importance, dtype: float64
    4    0.001575
    Name: importance, dtype: float64
    11   -0.001743
    Name: importance, dtype: float64
    --------------sample13_mlp--------------
    25   -0.011775
    Name: importance, dtype: float64
    14    0.003961
    Name: importance, dtype: float64
    8    0.005629
    Name: importance, dtype: float64
    --------------sample14_mlp--------------
    19   -0.003724
    Name: importance, dtype: float64
    4    0.00145
    Name: importance, dtype: float64
    26   -0.031434
    Name: importance, dtype: float64
    --------------sample15_mlp--------------
    18   -0.000513
    Name: importance, dtype: float64
    9    0.00303
    Name: importance, dtype: float64
    1    0.008329
    Name: importance, dtype: float64
    --------------sample16_mlp--------------
    6    0.012435
    Name: importance, dtype: float64
    13    0.005684
    Name: importance, dtype: float64
    26   -0.018244
    Name: importance, dtype: float64
    --------------sample17_mlp--------------
    16   -0.003132
    Name: importance, dtype: float64
    7    0.000638
    Name: importance, dtype: float64
    0    0.00435
    Name: importance, dtype: float64
    --------------sample18_mlp--------------
    26   -0.010906
    Name: importance, dtype: float64
    7   -0.001474
    Name: importance, dtype: float64
    20   -0.004207
    Name: importance, dtype: float64
    --------------sample19_mlp--------------
    27   -0.012491
    Name: importance, dtype: float64
    7    0.004931
    Name: importance, dtype: float64
    20   -0.002516
    Name: importance, dtype: float64
    --------------sample20_mlp--------------
    14   -0.00141
    Name: importance, dtype: float64
    5    0.001224
    Name: importance, dtype: float64
    27   -0.0204
    Name: importance, dtype: float64
    --------------sample21_mlp--------------
    20   -0.003301
    Name: importance, dtype: float64
    13    0.004829
    Name: importance, dtype: float64
    3    0.01875
    Name: importance, dtype: float64
    --------------sample22_mlp--------------
    15    0.001787
    Name: importance, dtype: float64
    14    0.002711
    Name: importance, dtype: float64
    26   -0.007793
    Name: importance, dtype: float64
    --------------sample23_mlp--------------
    17   -0.002861
    Name: importance, dtype: float64
    6    0.0009
    Name: importance, dtype: float64
    21   -0.004399
    Name: importance, dtype: float64
    --------------sample24_mlp--------------
    22   -0.00287
    Name: importance, dtype: float64
    8    0.000541
    Name: importance, dtype: float64
    24   -0.003627
    Name: importance, dtype: float64
    --------------sample25_mlp--------------
    9    0.000586
    Name: importance, dtype: float64
    5    0.00096
    Name: importance, dtype: float64
    7    0.000831
    Name: importance, dtype: float64
    --------------sample26_mlp--------------
    3    0.00187
    Name: importance, dtype: float64
    17   -0.001278
    Name: importance, dtype: float64
    20   -0.002157
    Name: importance, dtype: float64
    --------------sample27_mlp--------------
    26   -0.013425
    Name: importance, dtype: float64
    4    0.001721
    Name: importance, dtype: float64
    18   -0.002443
    Name: importance, dtype: float64
    --------------sample28_mlp--------------
    26   -0.008078
    Name: importance, dtype: float64
    8    0.005572
    Name: importance, dtype: float64
    7    0.006144
    Name: importance, dtype: float64
    --------------sample29_mlp--------------
    24   -0.007944
    Name: importance, dtype: float64
    6    0.006876
    Name: importance, dtype: float64
    0    0.018066
    Name: importance, dtype: float64
    --------------sample30_mlp--------------
    15    0.001043
    Name: importance, dtype: float64
    7    0.007513
    Name: importance, dtype: float64
    28   -0.011813
    Name: importance, dtype: float64
    --------------sample31_mlp--------------
    15    0.001707
    Name: importance, dtype: float64
    16    0.001683
    Name: importance, dtype: float64
    8    0.00528
    Name: importance, dtype: float64
    --------------sample32_mlp--------------
    28   -0.001448
    Name: importance, dtype: float64
    21    0.000595
    Name: importance, dtype: float64
    4    0.006749
    Name: importance, dtype: float64
    


```python
feature_order_mlp
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
      <th>sample1_mlp</th>
      <th>sample2_mlp</th>
      <th>sample3_mlp</th>
      <th>sample4_mlp</th>
      <th>sample5_mlp</th>
      <th>sample6_mlp</th>
      <th>sample7_mlp</th>
      <th>sample8_mlp</th>
      <th>sample9_mlp</th>
      <th>sample10_mlp</th>
      <th>...</th>
      <th>sample23_mlp</th>
      <th>sample24_mlp</th>
      <th>sample25_mlp</th>
      <th>sample26_mlp</th>
      <th>sample27_mlp</th>
      <th>sample28_mlp</th>
      <th>sample29_mlp</th>
      <th>sample30_mlp</th>
      <th>sample31_mlp</th>
      <th>sample32_mlp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>edu_dummy</td>
      <td>cureperiod_dummy</td>
      <td>med_dummy</td>
      <td>cureperiod_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>dis_range</td>
      <td>emplev_dummy</td>
      <td>labortime_dummy</td>
      <td>ralation_dummy</td>
      <td>...</td>
      <td>laborchange_dummy</td>
      <td>ralation_dummy</td>
      <td>spouse_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>labor_dummy</td>
      <td>emplev_dummy</td>
      <td>med_dummy</td>
      <td>emplev_dummy</td>
      <td>smoke_dummy</td>
      <td>ralation_dummy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cureperiod_dummy</td>
      <td>emplev_dummy</td>
      <td>cureperiod_dummy</td>
      <td>laborchange_dummy</td>
      <td>labortime_dummy</td>
      <td>med_dummy</td>
      <td>satisfaction_dummy</td>
      <td>cureperiod_dummy</td>
      <td>laborchange_dummy</td>
      <td>con_dummy</td>
      <td>...</td>
      <td>ralation_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>emplev_dummy</td>
      <td>edu_dummy</td>
      <td>doctorex_dummy</td>
      <td>workhour_dummy</td>
      <td>ralation_dummy</td>
      <td>jobreturnopinion_dummy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>doctorex_dummy</td>
      <td>dis_range</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>labor_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>...</td>
      <td>sex_dummy</td>
      <td>spouse_dummy</td>
      <td>income</td>
      <td>ralation_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>laborchange_dummy</td>
      <td>cureperiod_dummy</td>
      <td>income</td>
      <td>laborchange_dummy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>work_dummy</td>
      <td>doctorex_dummy</td>
      <td>dis_range</td>
      <td>sex_dummy</td>
      <td>dis_range</td>
      <td>workday_dummy</td>
      <td>med_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>labortime_dummy</td>
      <td>...</td>
      <td>labor_dummy</td>
      <td>smoke_dummy</td>
      <td>smoke_dummy</td>
      <td>psy_dummy</td>
      <td>ralation_dummy</td>
      <td>workday_dummy</td>
      <td>cureperiod_dummy</td>
      <td>smoke_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>jobreturnopinion_dummy</td>
      <td>alcohol_dummy</td>
      <td>js_dummy</td>
      <td>labor_dummy</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>emplev_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>spouse_dummy</td>
      <td>...</td>
      <td>alcohol_dummy</td>
      <td>emplev_cont</td>
      <td>laborchange_dummy</td>
      <td>workhour_dummy</td>
      <td>js_dummy</td>
      <td>doctorex_dummy</td>
      <td>workhour_dummy</td>
      <td>sex_dummy</td>
      <td>cureperiod_dummy</td>
      <td>med_dummy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>satisfaction_dummy</td>
      <td>age_dummy</td>
      <td>alcohol_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>ralation_dummy</td>
      <td>alcohol_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>laborchange_dummy</td>
      <td>...</td>
      <td>spouse_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>js_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_range</td>
      <td>dis_range</td>
      <td>dis_level</td>
      <td>sex_dummy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>js_dummy</td>
      <td>edu_dummy</td>
      <td>smoke_dummy</td>
      <td>age_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>laborchange_dummy</td>
      <td>smoke_dummy</td>
      <td>cureperiod_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>...</td>
      <td>js_dummy</td>
      <td>age_dummy</td>
      <td>ralation_dummy</td>
      <td>satisfaction_dummy</td>
      <td>doctorex_dummy</td>
      <td>dis_level</td>
      <td>js_dummy</td>
      <td>spouse_dummy</td>
      <td>con_dummy</td>
      <td>cureperiod_dummy</td>
    </tr>
    <tr>
      <th>7</th>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborchange_dummy</td>
      <td>edu_dummy</td>
      <td>cureperiod_dummy</td>
      <td>income</td>
      <td>smoke_dummy</td>
      <td>doctorex_dummy</td>
      <td>alcohol_dummy</td>
      <td>edu_dummy</td>
      <td>...</td>
      <td>age_dummy</td>
      <td>income</td>
      <td>med_dummy</td>
      <td>cureperiod_dummy</td>
      <td>smoke_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>js_dummy</td>
      <td>workhour_dummy</td>
      <td>smoke_dummy</td>
    </tr>
    <tr>
      <th>8</th>
      <td>recoveryex_dummy</td>
      <td>js_dummy</td>
      <td>dis_level</td>
      <td>js_dummy</td>
      <td>dis_level</td>
      <td>js_dummy</td>
      <td>sex_dummy</td>
      <td>edu_dummy</td>
      <td>spouse_dummy</td>
      <td>med_dummy</td>
      <td>...</td>
      <td>dis_level</td>
      <td>js_dummy</td>
      <td>satisfaction_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_range</td>
      <td>js_dummy</td>
      <td>spouse_dummy</td>
      <td>income</td>
      <td>med_dummy</td>
      <td>dis_level</td>
    </tr>
    <tr>
      <th>9</th>
      <td>psy_dummy</td>
      <td>laborchange_dummy</td>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>ralation_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>labortime_dummy</td>
      <td>js_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>...</td>
      <td>dis_range</td>
      <td>laborchange_dummy</td>
      <td>psy_dummy</td>
      <td>recoveryex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>dis_range</td>
      <td>ralation_dummy</td>
      <td>ralation_dummy</td>
      <td>dis_dummy</td>
      <td>labortime_dummy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ralation_dummy</td>
      <td>recoveryex_dummy</td>
      <td>ralation_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_dummy</td>
      <td>cureperiod_dummy</td>
      <td>laborchange_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>...</td>
      <td>recoveryex_dummy</td>
      <td>alcohol_dummy</td>
      <td>cureperiod_dummy</td>
      <td>alcohol_dummy</td>
      <td>dis_level</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_dummy</td>
    </tr>
    <tr>
      <th>11</th>
      <td>jobdoctorex_dummy</td>
      <td>smoke_dummy</td>
      <td>recoveryex_dummy</td>
      <td>smoke_dummy</td>
      <td>work_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>laborcontract_dummy</td>
      <td>ralation_dummy</td>
      <td>smoke_dummy</td>
      <td>...</td>
      <td>jobdoctorex_dummy</td>
      <td>sex_dummy</td>
      <td>dis_range</td>
      <td>income</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_dummy</td>
      <td>laborcontract_dummy</td>
      <td>workhour_dummy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>laborcontract_dummy</td>
      <td>ralation_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_level</td>
      <td>alcohol_dummy</td>
      <td>recoveryex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>recoveryex_dummy</td>
      <td>doctorex_dummy</td>
      <td>age_dummy</td>
      <td>...</td>
      <td>cureperiod_dummy</td>
      <td>cureperiod_dummy</td>
      <td>workhour_dummy</td>
      <td>labor_dummy</td>
      <td>emplev_cont</td>
      <td>labor_dummy</td>
      <td>emplev_cont</td>
      <td>recoveryex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>jobdoctorex_dummy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>spouse_dummy</td>
      <td>dis_level</td>
      <td>jobreturnopinion_dummy</td>
      <td>satisfaction_dummy</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>js_dummy</td>
      <td>ralation_dummy</td>
      <td>recoveryex_dummy</td>
      <td>emplev_dummy</td>
      <td>...</td>
      <td>emplev_cont</td>
      <td>satisfaction_dummy</td>
      <td>sex_dummy</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>sex_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>labortime_dummy</td>
      <td>satisfaction_dummy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>dis_dummy</td>
      <td>spouse_dummy</td>
      <td>edu_dummy</td>
      <td>alcohol_dummy</td>
      <td>con_dummy</td>
      <td>laborchange_dummy</td>
      <td>work_dummy</td>
      <td>sex_dummy</td>
      <td>emplev_dummy</td>
      <td>dis_dummy</td>
      <td>...</td>
      <td>smoke_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborchange_dummy</td>
      <td>spouse_dummy</td>
      <td>con_dummy</td>
      <td>sex_dummy</td>
      <td>work_dummy</td>
      <td>spouse_dummy</td>
      <td>work_dummy</td>
    </tr>
    <tr>
      <th>15</th>
      <td>workday_dummy</td>
      <td>satisfaction_dummy</td>
      <td>con_dummy</td>
      <td>spouse_dummy</td>
      <td>satisfaction_dummy</td>
      <td>satisfaction_dummy</td>
      <td>doctorex_dummy</td>
      <td>spouse_dummy</td>
      <td>dis_level</td>
      <td>dis_level</td>
      <td>...</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
      <td>doctorex_dummy</td>
      <td>dis_dummy</td>
      <td>dis_dummy</td>
      <td>laborcontract_dummy</td>
      <td>satisfaction_dummy</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>doctorex_dummy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>age_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>satisfaction_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>dis_dummy</td>
      <td>age_dummy</td>
      <td>psy_dummy</td>
      <td>dis_level</td>
      <td>psy_dummy</td>
      <td>work_dummy</td>
      <td>...</td>
      <td>dis_dummy</td>
      <td>laborcontract_dummy</td>
      <td>alcohol_dummy</td>
      <td>sex_dummy</td>
      <td>sex_dummy</td>
      <td>satisfaction_dummy</td>
      <td>emplev_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>js_dummy</td>
      <td>age_dummy</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dis_level</td>
      <td>sex_dummy</td>
      <td>work_dummy</td>
      <td>ralation_dummy</td>
      <td>laborcontract_dummy</td>
      <td>laborcontract_dummy</td>
      <td>ralation_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>...</td>
      <td>psy_dummy</td>
      <td>dis_dummy</td>
      <td>recoveryex_dummy</td>
      <td>js_dummy</td>
      <td>cureperiod_dummy</td>
      <td>smoke_dummy</td>
      <td>income</td>
      <td>satisfaction_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
    </tr>
    <tr>
      <th>18</th>
      <td>doctorex_dummy</td>
      <td>med_dummy</td>
      <td>laborcontract_dummy</td>
      <td>med_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>con_dummy</td>
      <td>edu_dummy</td>
      <td>doctorex_dummy</td>
      <td>...</td>
      <td>satisfaction_dummy</td>
      <td>dis_range</td>
      <td>work_dummy</td>
      <td>doctorex_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>edu_dummy</td>
      <td>laborchange_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
    </tr>
    <tr>
      <th>19</th>
      <td>laborchange_dummy</td>
      <td>con_dummy</td>
      <td>dis_dummy</td>
      <td>psy_dummy</td>
      <td>laborchange_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>con_dummy</td>
      <td>satisfaction_dummy</td>
      <td>smoke_dummy</td>
      <td>recoveryex_dummy</td>
      <td>...</td>
      <td>con_dummy</td>
      <td>recoveryex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>dis_range</td>
      <td>con_dummy</td>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>doctorex_dummy</td>
      <td>doctorex_dummy</td>
      <td>recoveryex_dummy</td>
    </tr>
    <tr>
      <th>20</th>
      <td>alcohol_dummy</td>
      <td>dis_dummy</td>
      <td>psy_dummy</td>
      <td>dis_dummy</td>
      <td>sex_dummy</td>
      <td>con_dummy</td>
      <td>workday_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>laborcontract_dummy</td>
      <td>workday_dummy</td>
      <td>...</td>
      <td>workhour_dummy</td>
      <td>workday_dummy</td>
      <td>dis_dummy</td>
      <td>med_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>alcohol_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>edu_dummy</td>
      <td>sex_dummy</td>
      <td>dis_range</td>
    </tr>
    <tr>
      <th>21</th>
      <td>sex_dummy</td>
      <td>workday_dummy</td>
      <td>sex_dummy</td>
      <td>con_dummy</td>
      <td>workday_dummy</td>
      <td>spouse_dummy</td>
      <td>spouse_dummy</td>
      <td>med_dummy</td>
      <td>con_dummy</td>
      <td>income</td>
      <td>...</td>
      <td>med_dummy</td>
      <td>doctorex_dummy</td>
      <td>con_dummy</td>
      <td>work_dummy</td>
      <td>workhour_dummy</td>
      <td>spouse_dummy</td>
      <td>age_dummy</td>
      <td>labor_dummy</td>
      <td>emplev_cont</td>
      <td>js_dummy</td>
    </tr>
    <tr>
      <th>22</th>
      <td>con_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>workday_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>workhour_dummy</td>
      <td>labortime_dummy</td>
      <td>recoveryex_dummy</td>
      <td>dis_dummy</td>
      <td>workhour_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>...</td>
      <td>doctorex_dummy</td>
      <td>psy_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>smoke_dummy</td>
      <td>satisfaction_dummy</td>
      <td>laborchange_dummy</td>
      <td>con_dummy</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>workday_dummy</td>
    </tr>
    <tr>
      <th>23</th>
      <td>med_dummy</td>
      <td>labortime_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>workday_dummy</td>
      <td>edu_dummy</td>
      <td>cureperiod_dummy</td>
      <td>dis_dummy</td>
      <td>workday_dummy</td>
      <td>dis_dummy</td>
      <td>js_dummy</td>
      <td>...</td>
      <td>workday_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>age_dummy</td>
      <td>alcohol_dummy</td>
      <td>cureperiod_dummy</td>
      <td>alcohol_dummy</td>
      <td>con_dummy</td>
      <td>labor_dummy</td>
      <td>emplev_dummy</td>
    </tr>
    <tr>
      <th>24</th>
      <td>smoke_dummy</td>
      <td>psy_dummy</td>
      <td>emplev_cont</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborcontract_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>laborcontract_dummy</td>
      <td>...</td>
      <td>laborcontract_dummy</td>
      <td>med_dummy</td>
      <td>age_dummy</td>
      <td>jobreturnopinion_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>jobdoctorex_dummy</td>
      <td>psy_dummy</td>
      <td>laborcontract_dummy</td>
      <td>alcohol_dummy</td>
      <td>labor_dummy</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>work_dummy</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>psy_dummy</td>
      <td>workday_dummy</td>
      <td>alcohol_dummy</td>
      <td>...</td>
      <td>edu_dummy</td>
      <td>labor_dummy</td>
      <td>edu_dummy</td>
      <td>edu_dummy</td>
      <td>workday_dummy</td>
      <td>income</td>
      <td>smoke_dummy</td>
      <td>labortime_dummy</td>
      <td>dis_range</td>
      <td>edu_dummy</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>age_dummy</td>
      <td>work_dummy</td>
      <td>med_dummy</td>
      <td>psy_dummy</td>
      <td>...</td>
      <td>labortime_dummy</td>
      <td>edu_dummy</td>
      <td>labortime_dummy</td>
      <td>laborcontract_dummy</td>
      <td>psy_dummy</td>
      <td>psy_dummy</td>
      <td>work_dummy</td>
      <td>age_dummy</td>
      <td>age_dummy</td>
      <td>spouse_dummy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>work_dummy</td>
      <td>labortime_dummy</td>
      <td>workday_dummy</td>
      <td>labortime_dummy</td>
      <td>labortime_dummy</td>
      <td>age_dummy</td>
      <td>labortime_dummy</td>
      <td>alcohol_dummy</td>
      <td>laborchange_dummy</td>
      <td>alcohol_dummy</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>work_dummy</td>
      <td>emplev_cont</td>
      <td>workday_dummy</td>
      <td>med_dummy</td>
      <td>workday_dummy</td>
      <td>psy_dummy</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>income</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 32 columns</p>
</div>




```python
service_mlp
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
      <th>psy_order</th>
      <th>psy_score</th>
      <th>job_order</th>
      <th>job_score</th>
      <th>med_order</th>
      <th>med_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample1_mlp</th>
      <td>9.0</td>
      <td>-0.000761</td>
      <td>6.0</td>
      <td>0.000310</td>
      <td>23.0</td>
      <td>-0.003573</td>
    </tr>
    <tr>
      <th>sample2_mlp</th>
      <td>24.0</td>
      <td>-0.012890</td>
      <td>8.0</td>
      <td>-0.000526</td>
      <td>18.0</td>
      <td>-0.003148</td>
    </tr>
    <tr>
      <th>sample3_mlp</th>
      <td>20.0</td>
      <td>-0.005548</td>
      <td>4.0</td>
      <td>0.001631</td>
      <td>0.0</td>
      <td>0.007597</td>
    </tr>
    <tr>
      <th>sample4_mlp</th>
      <td>19.0</td>
      <td>-0.004696</td>
      <td>8.0</td>
      <td>-0.001182</td>
      <td>18.0</td>
      <td>-0.004520</td>
    </tr>
    <tr>
      <th>sample5_mlp</th>
      <td>13.0</td>
      <td>-0.002494</td>
      <td>4.0</td>
      <td>0.002015</td>
      <td>25.0</td>
      <td>-0.039100</td>
    </tr>
    <tr>
      <th>sample6_mlp</th>
      <td>13.0</td>
      <td>0.001781</td>
      <td>8.0</td>
      <td>0.007140</td>
      <td>1.0</td>
      <td>0.023186</td>
    </tr>
    <tr>
      <th>sample7_mlp</th>
      <td>16.0</td>
      <td>-0.051726</td>
      <td>13.0</td>
      <td>-0.008914</td>
      <td>3.0</td>
      <td>0.053809</td>
    </tr>
    <tr>
      <th>sample8_mlp</th>
      <td>25.0</td>
      <td>-0.015022</td>
      <td>9.0</td>
      <td>-0.002958</td>
      <td>21.0</td>
      <td>-0.006561</td>
    </tr>
    <tr>
      <th>sample9_mlp</th>
      <td>16.0</td>
      <td>-0.003347</td>
      <td>9.0</td>
      <td>-0.001821</td>
      <td>26.0</td>
      <td>-0.019942</td>
    </tr>
    <tr>
      <th>sample10_mlp</th>
      <td>26.0</td>
      <td>-0.001864</td>
      <td>23.0</td>
      <td>-0.000397</td>
      <td>8.0</td>
      <td>0.003969</td>
    </tr>
    <tr>
      <th>sample11_mlp</th>
      <td>19.0</td>
      <td>-0.005503</td>
      <td>4.0</td>
      <td>0.001209</td>
      <td>0.0</td>
      <td>0.011734</td>
    </tr>
    <tr>
      <th>sample12_mlp</th>
      <td>18.0</td>
      <td>-0.004536</td>
      <td>4.0</td>
      <td>0.001575</td>
      <td>11.0</td>
      <td>-0.001743</td>
    </tr>
    <tr>
      <th>sample13_mlp</th>
      <td>25.0</td>
      <td>-0.011775</td>
      <td>14.0</td>
      <td>0.003961</td>
      <td>8.0</td>
      <td>0.005629</td>
    </tr>
    <tr>
      <th>sample14_mlp</th>
      <td>19.0</td>
      <td>-0.003724</td>
      <td>4.0</td>
      <td>0.001450</td>
      <td>26.0</td>
      <td>-0.031434</td>
    </tr>
    <tr>
      <th>sample15_mlp</th>
      <td>18.0</td>
      <td>-0.000513</td>
      <td>9.0</td>
      <td>0.003030</td>
      <td>1.0</td>
      <td>0.008329</td>
    </tr>
    <tr>
      <th>sample16_mlp</th>
      <td>6.0</td>
      <td>0.012435</td>
      <td>13.0</td>
      <td>0.005684</td>
      <td>26.0</td>
      <td>-0.018244</td>
    </tr>
    <tr>
      <th>sample17_mlp</th>
      <td>16.0</td>
      <td>-0.003132</td>
      <td>7.0</td>
      <td>0.000638</td>
      <td>0.0</td>
      <td>0.004350</td>
    </tr>
    <tr>
      <th>sample18_mlp</th>
      <td>26.0</td>
      <td>-0.010906</td>
      <td>7.0</td>
      <td>-0.001474</td>
      <td>20.0</td>
      <td>-0.004207</td>
    </tr>
    <tr>
      <th>sample19_mlp</th>
      <td>27.0</td>
      <td>-0.012491</td>
      <td>7.0</td>
      <td>0.004931</td>
      <td>20.0</td>
      <td>-0.002516</td>
    </tr>
    <tr>
      <th>sample20_mlp</th>
      <td>14.0</td>
      <td>-0.001410</td>
      <td>5.0</td>
      <td>0.001224</td>
      <td>27.0</td>
      <td>-0.020400</td>
    </tr>
    <tr>
      <th>sample21_mlp</th>
      <td>20.0</td>
      <td>-0.003301</td>
      <td>13.0</td>
      <td>0.004829</td>
      <td>3.0</td>
      <td>0.018750</td>
    </tr>
    <tr>
      <th>sample22_mlp</th>
      <td>15.0</td>
      <td>0.001787</td>
      <td>14.0</td>
      <td>0.002711</td>
      <td>26.0</td>
      <td>-0.007793</td>
    </tr>
    <tr>
      <th>sample23_mlp</th>
      <td>17.0</td>
      <td>-0.002861</td>
      <td>6.0</td>
      <td>0.000900</td>
      <td>21.0</td>
      <td>-0.004399</td>
    </tr>
    <tr>
      <th>sample24_mlp</th>
      <td>22.0</td>
      <td>-0.002870</td>
      <td>8.0</td>
      <td>0.000541</td>
      <td>24.0</td>
      <td>-0.003627</td>
    </tr>
    <tr>
      <th>sample25_mlp</th>
      <td>9.0</td>
      <td>0.000586</td>
      <td>5.0</td>
      <td>0.000960</td>
      <td>7.0</td>
      <td>0.000831</td>
    </tr>
    <tr>
      <th>sample26_mlp</th>
      <td>3.0</td>
      <td>0.001870</td>
      <td>17.0</td>
      <td>-0.001278</td>
      <td>20.0</td>
      <td>-0.002157</td>
    </tr>
    <tr>
      <th>sample27_mlp</th>
      <td>26.0</td>
      <td>-0.013425</td>
      <td>4.0</td>
      <td>0.001721</td>
      <td>18.0</td>
      <td>-0.002443</td>
    </tr>
    <tr>
      <th>sample28_mlp</th>
      <td>26.0</td>
      <td>-0.008078</td>
      <td>8.0</td>
      <td>0.005572</td>
      <td>7.0</td>
      <td>0.006144</td>
    </tr>
    <tr>
      <th>sample29_mlp</th>
      <td>24.0</td>
      <td>-0.007944</td>
      <td>6.0</td>
      <td>0.006876</td>
      <td>0.0</td>
      <td>0.018066</td>
    </tr>
    <tr>
      <th>sample30_mlp</th>
      <td>15.0</td>
      <td>0.001043</td>
      <td>7.0</td>
      <td>0.007513</td>
      <td>28.0</td>
      <td>-0.011813</td>
    </tr>
    <tr>
      <th>sample31_mlp</th>
      <td>15.0</td>
      <td>0.001707</td>
      <td>16.0</td>
      <td>0.001683</td>
      <td>8.0</td>
      <td>0.005280</td>
    </tr>
    <tr>
      <th>sample32_mlp</th>
      <td>28.0</td>
      <td>-0.001448</td>
      <td>21.0</td>
      <td>0.000595</td>
      <td>4.0</td>
      <td>0.006749</td>
    </tr>
  </tbody>
</table>
</div>



## to CSV


```python
scores = [score_dt, score_rf, score_sv, score_lgb, score_xgb, score_mlp]
feature_orders = [feature_order_dt, feature_order_rf, feature_order_sv, feature_order_lgb, feature_order_xgb, feature_order_mlp]
services = [service_dt, service_rf, service_sv, service_lgb, service_xgb, service_mlp]

scores1 = ['score_dt', 'score_rf', 'score_sv', 'score_lgb', 'score_xgb', 'score_mlp']
feature_orders1 = ['feature_order_dt', 'feature_order_rf', 'feature_order_sv', 'feature_order_lgb', 'feature_order_xgb', 'feature_order_mlp']
services1 = ['service_dt', 'service_rf', 'service_sv', 'service_lgb', 'service_xgb', 'service_mlp']

for score, score1, feature_order, feature_order1, service, service1 in zip(scores, scores1, feature_orders, feature_orders1, services, services1):
    score.to_csv('C:/Users/Dawis Kim/Dropbox/산재보험패널/1st_cohort/STATA/result/'+str(score1)+'.csv')
    feature_order.to_csv('C:/Users/Dawis Kim/Dropbox/산재보험패널/1st_cohort/STATA/result/'+str(feature_order1)+'.csv')
    service.to_csv('C:/Users/Dawis Kim/Dropbox/산재보험패널/1st_cohort/STATA/result/'+str(service1)+'.csv')
```
