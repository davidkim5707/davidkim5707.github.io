---
layout: post
title: machine learning_recommending system
categories:
  - Programming
tags:
  - Python
  - recommend
  - FM
last_modified_at: 2020-04-11
use_math: true
---
source& copyright: lecture note in DataScienceLab in Yonsei university  

### Recommend System using FM model


### *RecSys 2015 Challenge Dataset Data


### *purpose: predict what kind of product would be chosen by consumers



```python
!pip install tensorflow
!pip install tffm
```

    Collecting tensorflow
      Using cached https://files.pythonhosted.org/packages/f7/08/25e47a53692c2e0dcd2211a493ddfe9007a5cd92e175d6dffa6169a0b392/tensorflow-1.14.0-cp37-cp37m-win_amd64.whl
    Collecting google-pasta>=0.1.6 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/d0/33/376510eb8d6246f3c30545f416b2263eee461e40940c2a4413c711bdf62d/google_pasta-0.1.7-py3-none-any.whl
    Collecting keras-preprocessing>=1.0.5 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/28/6a/8c1f62c37212d9fc441a7e26736df51ce6f0e38455816445471f10da4f0a/Keras_Preprocessing-1.1.0-py2.py3-none-any.whl
    Requirement already satisfied: numpy<2.0,>=1.14.5 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.16.2)
    Collecting keras-applications>=1.0.6 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/71/e3/19762fdfc62877ae9102edf6342d71b28fbfd9dea3d2f96a882ce099b03f/Keras_Applications-1.0.8-py3-none-any.whl
    Collecting tensorflow-estimator<1.15.0rc0,>=1.14.0rc0 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/3c/d5/21860a5b11caf0678fbc8319341b0ae21a07156911132e0e71bffed0510d/tensorflow_estimator-1.14.0-py2.py3-none-any.whl
    Collecting grpcio>=1.8.6 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/7a/f5/fe046577387a3589ab3092096ca423fcf9a8c7ac876f56c6f3b4c9b9e533/grpcio-1.22.0-cp37-cp37m-win_amd64.whl
    Collecting protobuf>=3.6.1 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/46/8b/5e77963dac4a944a0c6b198c004fac4c85d7adc54221c288fc6ca9078072/protobuf-3.9.1-cp37-cp37m-win_amd64.whl
    Collecting gast>=0.2.0 (from tensorflow)
    Requirement already satisfied: six>=1.10.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.12.0)
    Requirement already satisfied: wrapt>=1.11.1 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.11.1)
    Collecting termcolor>=1.1.0 (from tensorflow)
    Collecting tensorboard<1.15.0,>=1.14.0 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/91/2d/2ed263449a078cd9c8a9ba50ebd50123adf1f8cfbea1492f9084169b89d9/tensorboard-1.14.0-py3-none-any.whl
    Collecting absl-py>=0.7.0 (from tensorflow)
    Requirement already satisfied: wheel>=0.26 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (0.33.1)
    Collecting astor>=0.6.0 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/d1/4f/950dfae467b384fc96bc6469de25d832534f6b4441033c39f914efd13418/astor-0.8.0-py2.py3-none-any.whl
    Requirement already satisfied: h5py in c:\programdata\anaconda3\lib\site-packages (from keras-applications>=1.0.6->tensorflow) (2.9.0)
    Requirement already satisfied: setuptools in c:\programdata\anaconda3\lib\site-packages (from protobuf>=3.6.1->tensorflow) (40.8.0)
    Collecting markdown>=2.6.8 (from tensorboard<1.15.0,>=1.14.0->tensorflow)
      Using cached https://files.pythonhosted.org/packages/c0/4e/fd492e91abdc2d2fcb70ef453064d980688762079397f779758e055f6575/Markdown-3.1.1-py2.py3-none-any.whl
    Requirement already satisfied: werkzeug>=0.11.15 in c:\programdata\anaconda3\lib\site-packages (from tensorboard<1.15.0,>=1.14.0->tensorflow) (0.14.1)
    Installing collected packages: google-pasta, keras-preprocessing, keras-applications, tensorflow-estimator, grpcio, protobuf, gast, termcolor, absl-py, markdown, tensorboard, astor, tensorflow
    Successfully installed absl-py-0.7.1 astor-0.8.0 gast-0.2.2 google-pasta-0.1.7 grpcio-1.22.0 keras-applications-1.0.8 keras-preprocessing-1.1.0 markdown-3.1.1 protobuf-3.9.1 tensorboard-1.14.0 tensorflow-1.14.0 tensorflow-estimator-1.14.0 termcolor-1.1.0
    

    ERROR: tensorboard 1.14.0 has requirement setuptools>=41.0.0, but you'll have setuptools 40.8.0 which is incompatible.
    

    Collecting tffm
      Downloading https://files.pythonhosted.org/packages/58/ad/a9b6a5a389969c4ea0e177607c72b52f22458c2422b717714df944355896/tffm-1.0.1.tar.gz
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (from tffm) (0.20.3)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from tffm) (1.16.2)
    Requirement already satisfied: tqdm in c:\programdata\anaconda3\lib\site-packages (from tffm) (4.31.1)
    Requirement already satisfied: scipy>=0.13.3 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn->tffm) (1.2.1)
    Building wheels for collected packages: tffm
      Building wheel for tffm (setup.py): started
      Building wheel for tffm (setup.py): finished with status 'done'
      Created wheel for tffm: filename=tffm-1.0.1-cp37-none-any.whl size=11867 sha256=e7ae4bf7f68fefd28f473552fb9ea2d22a11a628052b178424cdb104970cd86d
      Stored in directory: C:\Users\김다윗\AppData\Local\pip\Cache\wheels\12\65\ea\6ffb58f9871d6f309690cddc75cea139d3a88f55da54f32081
    Successfully built tffm
    Installing collected packages: tffm
    Successfully installed tffm-1.0.1
    


```python
import tensorflow as tf
from tffm import TFFMRegressor
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

# 데이터 가져오기


```python
buys = open('yoochoose-buys.dat', 'r')
clicks = open('yoochoose-clicks.dat', 'r')
```


```python
initial_buys_df = pd.read_csv(buys, names=['Session ID', 'Timestamp', 'Item ID', 'Category', 'Quantity'],
                              dtype={'Session ID': 'float32', 'Timestamp': 'str', 'Item ID': 'float32',
                                     'Category': 'str'})

initial_buys_df.set_index('Session ID', inplace=True)

initial_clicks_df = pd.read_csv(clicks, names=['Session ID', 'Timestamp', 'Item ID', 'Category'],
                                dtype={'Category': 'str'})

initial_clicks_df.set_index('Session ID', inplace=True)
```


```python
initial_buys_df.tail()
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
      <th>Timestamp</th>
      <th>Item ID</th>
      <th>Category</th>
      <th>Quantity</th>
    </tr>
    <tr>
      <th>Session ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11368701.0</th>
      <td>2014-09-26T07:52:51.357Z</td>
      <td>214849808.0</td>
      <td>554</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11368691.0</th>
      <td>2014-09-25T09:37:44.206Z</td>
      <td>214700000.0</td>
      <td>6806</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11523941.0</th>
      <td>2014-09-25T06:14:47.965Z</td>
      <td>214578016.0</td>
      <td>14556</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11423202.0</th>
      <td>2014-09-26T18:49:34.024Z</td>
      <td>214849168.0</td>
      <td>1046</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11423202.0</th>
      <td>2014-09-26T18:49:34.026Z</td>
      <td>214560496.0</td>
      <td>5549</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
initial_clicks_df.head()
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
      <th>Timestamp</th>
      <th>Item ID</th>
      <th>Category</th>
    </tr>
    <tr>
      <th>Session ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2014-04-07T10:51:09.277Z</td>
      <td>214536502</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-04-07T10:54:09.868Z</td>
      <td>214536500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-04-07T10:54:46.998Z</td>
      <td>214536506</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-04-07T10:57:00.306Z</td>
      <td>214577561</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-04-07T13:56:37.614Z</td>
      <td>214662742</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(initial_clicks_df), len(initial_buys_df)
```




    (33003944, 1150753)



# 데이터 전처리
### 1) 필요없는 열 제거
### 2) 일부 데이터만 추출 (용량, 속도관계로)
### 3) 필요한 열 추가
### 4) One-hot encoding (벡터형태로 바꾸기 위해)
### 5) 데이터 통합

 


```python
# 여기선 Timestamp를 사용하지 않을 것이므로 column 삭제
# 즉, 여기선 사용자 ID와 구매/클릭 이력만 사용할 것

initial_buys_df = initial_buys_df.drop('Timestamp', 1)
initial_clicks_df = initial_clicks_df.drop('Timestamp', 1)

```


```python
# 데이터가 굉장히 큼! 여기선 간단하게 보여주기 위해 구매/클릭 수 상위 10,000명의 데이터만 가져옴

x = Counter(initial_buys_df.index).most_common(10000) # most_common(n): 상위 n개 데이터만 가져옴
top_k = dict(x).keys()
initial_buys_df = initial_buys_df[initial_buys_df.index.isin(top_k)]
initial_clicks_df = initial_clicks_df[initial_clicks_df.index.isin(top_k)]
```


```python
# index를 나타내는 열 추가. index(Session, 즉, 클릭)도 벡터에 포함시키기 위해

initial_buys_df['_Session ID'] = initial_buys_df.index
```


```python
initial_buys_df.head()
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
      <th>Item ID</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>_Session ID</th>
    </tr>
    <tr>
      <th>Session ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>420471.0</th>
      <td>214717888.0</td>
      <td>2092</td>
      <td>1</td>
      <td>420471.0</td>
    </tr>
    <tr>
      <th>420471.0</th>
      <td>214821024.0</td>
      <td>1570</td>
      <td>1</td>
      <td>420471.0</td>
    </tr>
    <tr>
      <th>420471.0</th>
      <td>214829280.0</td>
      <td>837</td>
      <td>1</td>
      <td>420471.0</td>
    </tr>
    <tr>
      <th>420471.0</th>
      <td>214819552.0</td>
      <td>418</td>
      <td>1</td>
      <td>420471.0</td>
    </tr>
    <tr>
      <th>420471.0</th>
      <td>214746384.0</td>
      <td>784</td>
      <td>1</td>
      <td>420471.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# One-hot encoding(벡터화)

transformed_buys = pd.get_dummies(initial_buys_df)
transformed_clicks = pd.get_dummies(initial_clicks_df)
```


```python
# 아이템과 카테고리에 대한 과거 데이터를 추리기
# Aggregate historical data for Items and Categories

filtered_buys = transformed_buys.filter(regex="Item.*|Category.*")
filtered_clicks = transformed_clicks.filter(regex="Item.*|Category.*")

historical_buy_data = filtered_buys.groupby(filtered_buys.index).sum()
historical_buy_data = historical_buy_data.rename(columns=lambda column_name: 'buy history:' + column_name)

historical_click_data = filtered_clicks.groupby(filtered_clicks.index).sum()
historical_click_data = historical_click_data.rename(columns=lambda column_name: 'click history:' + column_name)

```


```python
# 각 사용자id를 기준으로 과거 데이터와 원본 데이터 병합
# Merge historical data of every user_id

merged1 = pd.merge(transformed_buys, historical_buy_data, left_index=True, right_index=True)
merged2 = pd.merge(merged1, historical_click_data, left_index=True, right_index=True)
```

# TFFM라이브러리를 사용하여 학습모델 구성


```python
model = TFFMRegressor(
    order=2, 
    rank=7,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.1), #다른 알고리즘을 써도 됌 
    n_epochs=100,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)


merged2.drop(['Item ID', '_Session ID', 'click history:Item ID', 'buy history:Item ID'], 1, inplace=True)
X = np.array(merged2)
X = np.nan_to_num(X)
y = np.array(merged2['Quantity'].as_matrix())

```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:15: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      from ipykernel import kernelapp as app
    

# 학습 데이터, 테스트 데이터 나누기


```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
```


```python
# What happens if we only have access to categories and no historical click/purchase data?
# Let's delete historical click and purchasing data for the cold_start test set

for column in X_te_cs.columns:
    if ('buy' in column or 'click' in column) and ('Category' not in column):
        X_te_cs[column] = 0
```


```python
# Compute the mean squared error for both test sets

model.fit(X_tr, y_tr, show_progress=True)
predictions = model.predict(X_te)
cold_start_predictions = model.predict(X_te_cs)
print('MSE: {}'.format(mean_squared_error(y_te, predictions)))
```

    100%|█████████████████████████████████████████████████████████████████████████████| 100/100 [01:31<00:00,  1.15epoch/s]
    

    MSE: 0.7006379736490871
    
