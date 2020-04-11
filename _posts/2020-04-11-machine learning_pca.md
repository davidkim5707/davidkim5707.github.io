---
layout: post
title: machine learning_pca
categories:
  - Programming
tags:
  - Python
  - pca
last_modified_at: 2020-04-11
use_math: true
---
source& copyright: lecture note in DataScienceLab in Yonsei university  

### PCA
Regression & Penalized model 1

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.datasets import load_iris
```

#### 데이터 불러오기


```python
# 데이터 불러오기
Iris = load_iris()

# dataframe으로
df = pd.DataFrame(data = np.c_[Iris['data'], Iris['target']], columns = Iris['feature_names']+['target'])
df['target'] = df['target'].map({0:'setosa', 1:'versicolor', 2: 'virginica'})
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
```


```python
df.head()
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
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



#### Covariance Matrix


```python
x = df.loc[:,['sepal length', 'sepal width', 'petal length', 'petal width']].values
y = df.loc[:,['target']].values
```

공분산 행렬


```python
features = x.T
covariance_matrix = np.cov(features)
print(covariance_matrix)
```

    [[ 0.68569351 -0.042434    1.27431544  0.51627069]
     [-0.042434    0.18997942 -0.32965638 -0.12163937]
     [ 1.27431544 -0.32965638  3.11627785  1.2956094 ]
     [ 0.51627069 -0.12163937  1.2956094   0.58100626]]
    

고유값, 고유벡터


```python
eig_val, eig_vec = np.linalg.eig(covariance_matrix)
```


```python
print(eig_vec) # 고유벡터
```

    [[ 0.36138659 -0.65658877 -0.58202985  0.31548719]
     [-0.08452251 -0.73016143  0.59791083 -0.3197231 ]
     [ 0.85667061  0.17337266  0.07623608 -0.47983899]
     [ 0.3582892   0.07548102  0.54583143  0.75365743]]
    


```python
print(eig_val) # 고유값
```

    [4.22824171 0.24267075 0.0782095  0.02383509]
    

#### PCA(2차원)


```python
pca = PCA(n_components = 2) #components 조정 https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/ 참고
```


```python
principalComponents = pca.fit_transform(x)
principaldf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principaldf.head(5)
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
      <th>principal component 1</th>
      <th>principal component 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.684126</td>
      <td>0.319397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.714142</td>
      <td>-0.177001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.888991</td>
      <td>-0.144949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.745343</td>
      <td>-0.318299</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.728717</td>
      <td>0.326755</td>
    </tr>
  </tbody>
</table>
</div>




```python
finaldf = pd.concat([principaldf, df[['target']]], axis = 1)
finaldf.head()
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
      <th>principal component 1</th>
      <th>principal component 2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.684126</td>
      <td>0.319397</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.714142</td>
      <td>-0.177001</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.888991</td>
      <td>-0.144949</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.745343</td>
      <td>-0.318299</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.728717</td>
      <td>0.326755</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



#### 모델 유의성


```python
pca.explained_variance_ratio_
```




    array([0.92461872, 0.05306648])



pc1은 92%, pc2는 5% 정도 설명 가능

#### 그래프


```python
plt.figure(figsize = (15,10))
sns.scatterplot(x = 'principal component 1', y = 'principal component 2', hue = 'target', style = 'target', data = finaldf)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x257d4b79160>




![png](https://drive.google.com/uc?export=view&id=1soTqPU6EEaW7LuCxwGk9RvFijgr0-B5d)


feature scaling : from sklearn.preprocessing import StandardScaler

# Regression


```python
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(123)
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')
```




    [<matplotlib.lines.Line2D at 0x257d50cec88>]




![png](https://drive.google.com/uc?export=view&id=1UQPCmFx9FU1vaIoswKffd-cuKQn1FZd3)



```python
data.head()
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.047198</td>
      <td>0.703181</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.117011</td>
      <td>1.048396</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.186824</td>
      <td>0.969631</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.256637</td>
      <td>0.725112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.326450</td>
      <td>0.883506</td>
    </tr>
  </tbody>
</table>
</div>



sin함수에 noise 추가한 데이터


```python
for i in range(2,16):  #power of 1 is already there
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i
```


```python
data.head()
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
      <th>x</th>
      <th>y</th>
      <th>x_2</th>
      <th>x_3</th>
      <th>x_4</th>
      <th>x_5</th>
      <th>x_6</th>
      <th>x_7</th>
      <th>x_8</th>
      <th>x_9</th>
      <th>x_10</th>
      <th>x_11</th>
      <th>x_12</th>
      <th>x_13</th>
      <th>x_14</th>
      <th>x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.047198</td>
      <td>0.703181</td>
      <td>1.096623</td>
      <td>1.148381</td>
      <td>1.202581</td>
      <td>1.259340</td>
      <td>1.318778</td>
      <td>1.381021</td>
      <td>1.446202</td>
      <td>1.514459</td>
      <td>1.585938</td>
      <td>1.660790</td>
      <td>1.739176</td>
      <td>1.821260</td>
      <td>1.907219</td>
      <td>1.997235</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.117011</td>
      <td>1.048396</td>
      <td>1.247713</td>
      <td>1.393709</td>
      <td>1.556788</td>
      <td>1.738948</td>
      <td>1.942424</td>
      <td>2.169709</td>
      <td>2.423588</td>
      <td>2.707173</td>
      <td>3.023942</td>
      <td>3.377775</td>
      <td>3.773011</td>
      <td>4.214494</td>
      <td>4.707635</td>
      <td>5.258479</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.186824</td>
      <td>0.969631</td>
      <td>1.408551</td>
      <td>1.671702</td>
      <td>1.984016</td>
      <td>2.354677</td>
      <td>2.794587</td>
      <td>3.316683</td>
      <td>3.936319</td>
      <td>4.671717</td>
      <td>5.544505</td>
      <td>6.580351</td>
      <td>7.809718</td>
      <td>9.268760</td>
      <td>11.000386</td>
      <td>13.055521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.256637</td>
      <td>0.725112</td>
      <td>1.579137</td>
      <td>1.984402</td>
      <td>2.493673</td>
      <td>3.133642</td>
      <td>3.937850</td>
      <td>4.948448</td>
      <td>6.218404</td>
      <td>7.814277</td>
      <td>9.819710</td>
      <td>12.339811</td>
      <td>15.506664</td>
      <td>19.486248</td>
      <td>24.487142</td>
      <td>30.771450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.326450</td>
      <td>0.883506</td>
      <td>1.759470</td>
      <td>2.333850</td>
      <td>3.095735</td>
      <td>4.106339</td>
      <td>5.446854</td>
      <td>7.224981</td>
      <td>9.583578</td>
      <td>12.712139</td>
      <td>16.862020</td>
      <td>22.366630</td>
      <td>29.668222</td>
      <td>39.353420</td>
      <td>52.200353</td>
      <td>69.241170</td>
    </tr>
  </tbody>
</table>
</div>




```python
def linear_regression(data, power, models_to_plot):
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%power)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret
```


```python
#Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
```


![png](https://drive.google.com/uc?export=view&id=1W-Ql9JTpX0rLjmaBMB1dgiXCH-BXwR--)


차수가 높을수록 복잡한 모델


```python
coef_matrix_simple
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
      <th>rss</th>
      <th>intercept</th>
      <th>coef_x_1</th>
      <th>coef_x_2</th>
      <th>coef_x_3</th>
      <th>coef_x_4</th>
      <th>coef_x_5</th>
      <th>coef_x_6</th>
      <th>coef_x_7</th>
      <th>coef_x_8</th>
      <th>coef_x_9</th>
      <th>coef_x_10</th>
      <th>coef_x_11</th>
      <th>coef_x_12</th>
      <th>coef_x_13</th>
      <th>coef_x_14</th>
      <th>coef_x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>model_pow_1</th>
      <td>4.68889</td>
      <td>1.88584</td>
      <td>-0.597884</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_2</th>
      <td>4.67436</td>
      <td>1.98333</td>
      <td>-0.671849</td>
      <td>0.0119041</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_3</th>
      <td>1.71286</td>
      <td>-1.51254</td>
      <td>3.54702</td>
      <td>-1.48178</td>
      <td>0.160265</td>
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
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_4</th>
      <td>1.69327</td>
      <td>-2.23729</td>
      <td>4.74278</td>
      <td>-2.15209</td>
      <td>0.314083</td>
      <td>-0.012378</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_5</th>
      <td>1.6361</td>
      <td>0.942072</td>
      <td>-1.89927</td>
      <td>2.97864</td>
      <td>-1.53356</td>
      <td>0.30077</td>
      <td>-0.0201596</td>
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
    </tr>
    <tr>
      <th>model_pow_6</th>
      <td>1.52303</td>
      <td>12.4807</td>
      <td>-31.0481</td>
      <td>31.7359</td>
      <td>-15.7842</td>
      <td>4.06454</td>
      <td>-0.525585</td>
      <td>0.0271149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_7</th>
      <td>1.50693</td>
      <td>1.1977</td>
      <td>2.37111</td>
      <td>-8.42158</td>
      <td>9.66626</td>
      <td>-5.15933</td>
      <td>1.39437</td>
      <td>-0.18632</td>
      <td>0.00981454</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_8</th>
      <td>1.43751</td>
      <td>-59.7195</td>
      <td>209.279</td>
      <td>-301.627</td>
      <td>236.502</td>
      <td>-110.226</td>
      <td>31.3214</td>
      <td>-5.32207</td>
      <td>0.496814</td>
      <td>-0.0195948</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_9</th>
      <td>1.43676</td>
      <td>-76.1603</td>
      <td>272.251</td>
      <td>-404.436</td>
      <td>330.522</td>
      <td>-163.402</td>
      <td>50.6541</td>
      <td>-9.8509</td>
      <td>1.15762</td>
      <td>-0.0742219</td>
      <td>0.00195374</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_10</th>
      <td>1.42052</td>
      <td>-277.739</td>
      <td>1131.55</td>
      <td>-1992.48</td>
      <td>2007.49</td>
      <td>-1285.48</td>
      <td>548.562</td>
      <td>-158.513</td>
      <td>30.7054</td>
      <td>-3.82302</td>
      <td>0.276621</td>
      <td>-0.00884116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_11</th>
      <td>1.38217</td>
      <td>537.536</td>
      <td>-2695.75</td>
      <td>5904.35</td>
      <td>-7451.39</td>
      <td>6029.93</td>
      <td>-3291.9</td>
      <td>1240.01</td>
      <td>-323.1</td>
      <td>57.2118</td>
      <td>-6.57195</td>
      <td>0.441702</td>
      <td>-0.013184</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_12</th>
      <td>1.34977</td>
      <td>-1443.59</td>
      <td>7457.62</td>
      <td>-17227</td>
      <td>23539.3</td>
      <td>-21185.4</td>
      <td>13227.8</td>
      <td>-5874.97</td>
      <td>1870.42</td>
      <td>-423.83</td>
      <td>66.7062</td>
      <td>-6.92786</td>
      <td>0.426703</td>
      <td>-0.0117995</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_13</th>
      <td>1.27568</td>
      <td>-9403.15</td>
      <td>51668.5</td>
      <td>-127431</td>
      <td>186835</td>
      <td>-181752</td>
      <td>123950</td>
      <td>-61010.9</td>
      <td>21965.1</td>
      <td>-5789.6</td>
      <td>1104.62</td>
      <td>-148.446</td>
      <td>13.3206</td>
      <td>-0.716146</td>
      <td>0.01744</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_14</th>
      <td>1.27235</td>
      <td>-4897.32</td>
      <td>24711.9</td>
      <td>-54466.9</td>
      <td>68366.3</td>
      <td>-52789.3</td>
      <td>24321.9</td>
      <td>-4642.76</td>
      <td>-1781.95</td>
      <td>1702.5</td>
      <td>-658.542</td>
      <td>156.512</td>
      <td>-24.3049</td>
      <td>2.41708</td>
      <td>-0.140331</td>
      <td>0.00362745</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>model_pow_15</th>
      <td>1.24963</td>
      <td>-36526.8</td>
      <td>227441</td>
      <td>-646484</td>
      <td>1.11349e+06</td>
      <td>-1.30062e+06</td>
      <td>1.09217e+06</td>
      <td>-681670</td>
      <td>322264</td>
      <td>-116434</td>
      <td>32171.6</td>
      <td>-6746.69</td>
      <td>1055.12</td>
      <td>-119.18</td>
      <td>9.18307</td>
      <td>-0.431788</td>
      <td>0.00934362</td>
    </tr>
  </tbody>
</table>
</div>



1~15차 모델의 계수

# Ridge


```python
##https://ratsgo.github.io/machine%20learning/2017/05/22/RLR/ 참고
##https://greatjoy.tistory.com/59 코드 참고

def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret
```

alpha(강의자료 lambda)값이 커질수록 x값에 대해 덜 민감한, 더 단순한 모델, RSS 증가, 계수 =/= 0


```python
#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
```

    c:\users\kim98\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\ridge.py:125: LinAlgWarning: Ill-conditioned matrix (rcond=3.83911e-17): result may not be accurate.
      overwrite_a=True).T
    


![png](https://drive.google.com/uc?export=view&id=1tV_bM__ulx2Z36yg4RhODUDGFIkf_wH1)



```python
coef_matrix_ridge
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
      <th>rss</th>
      <th>intercept</th>
      <th>coef_x_1</th>
      <th>coef_x_2</th>
      <th>coef_x_3</th>
      <th>coef_x_4</th>
      <th>coef_x_5</th>
      <th>coef_x_6</th>
      <th>coef_x_7</th>
      <th>coef_x_8</th>
      <th>coef_x_9</th>
      <th>coef_x_10</th>
      <th>coef_x_11</th>
      <th>coef_x_12</th>
      <th>coef_x_13</th>
      <th>coef_x_14</th>
      <th>coef_x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_1e-15</th>
      <td>1.36842</td>
      <td>11.3637</td>
      <td>-109.011</td>
      <td>295.389</td>
      <td>-370.598</td>
      <td>248.301</td>
      <td>-87.4872</td>
      <td>11.6019</td>
      <td>1.63427</td>
      <td>-0.485482</td>
      <td>-0.0505551</td>
      <td>0.00933948</td>
      <td>0.00490828</td>
      <td>-0.000647964</td>
      <td>-0.000176742</td>
      <td>4.13935e-05</td>
      <td>-2.37445e-06</td>
    </tr>
    <tr>
      <th>alpha_1e-10</th>
      <td>1.45497</td>
      <td>-2.54323</td>
      <td>10.4022</td>
      <td>-12.5123</td>
      <td>7.05422</td>
      <td>-1.55166</td>
      <td>-0.0631853</td>
      <td>0.0441214</td>
      <td>0.00496497</td>
      <td>-0.000629424</td>
      <td>-0.000207847</td>
      <td>-1.59951e-05</td>
      <td>1.58641e-06</td>
      <td>4.73012e-07</td>
      <td>9.17361e-08</td>
      <td>2.79826e-08</td>
      <td>-6.45385e-09</td>
    </tr>
    <tr>
      <th>alpha_1e-08</th>
      <td>1.46605</td>
      <td>3.37297</td>
      <td>-5.3967</td>
      <td>3.48973</td>
      <td>-0.398007</td>
      <td>-0.191974</td>
      <td>0.000973383</td>
      <td>0.00809652</td>
      <td>0.00153802</td>
      <td>2.45488e-05</td>
      <td>-5.43229e-05</td>
      <td>-1.47665e-05</td>
      <td>-1.65239e-06</td>
      <td>1.91529e-07</td>
      <td>1.26013e-07</td>
      <td>2.10654e-08</td>
      <td>-4.98599e-09</td>
    </tr>
    <tr>
      <th>alpha_0.0001</th>
      <td>1.66576</td>
      <td>-0.252645</td>
      <td>1.42825</td>
      <td>-0.319223</td>
      <td>-0.0533222</td>
      <td>-0.00207982</td>
      <td>0.000961015</td>
      <td>0.000299055</td>
      <td>5.17486e-05</td>
      <td>5.42736e-06</td>
      <td>-4.89356e-08</td>
      <td>-1.91962e-07</td>
      <td>-5.72921e-08</td>
      <td>-1.10596e-08</td>
      <td>-1.3082e-09</td>
      <td>6.37933e-11</td>
      <td>1.00969e-10</td>
    </tr>
    <tr>
      <th>alpha_0.001</th>
      <td>1.97896</td>
      <td>0.623998</td>
      <td>0.506875</td>
      <td>-0.13544</td>
      <td>-0.0264657</td>
      <td>-0.00251488</td>
      <td>4.20218e-05</td>
      <td>7.66614e-05</td>
      <td>2.02965e-05</td>
      <td>3.77083e-06</td>
      <td>5.51617e-07</td>
      <td>5.9486e-08</td>
      <td>2.2642e-09</td>
      <td>-1.05569e-09</td>
      <td>-3.99199e-10</td>
      <td>-9.3171e-11</td>
      <td>-1.66959e-11</td>
    </tr>
    <tr>
      <th>alpha_0.01</th>
      <td>2.52318</td>
      <td>1.27345</td>
      <td>-0.0757816</td>
      <td>-0.057742</td>
      <td>-0.0107403</td>
      <td>-0.00134583</td>
      <td>-9.36545e-05</td>
      <td>9.133e-06</td>
      <td>5.37049e-06</td>
      <td>1.40331e-06</td>
      <td>2.86205e-07</td>
      <td>4.96999e-08</td>
      <td>7.29585e-09</td>
      <td>7.96288e-10</td>
      <td>1.35614e-11</td>
      <td>-2.71338e-11</td>
      <td>-1.12808e-11</td>
    </tr>
    <tr>
      <th>alpha_1</th>
      <td>6.88341</td>
      <td>0.939331</td>
      <td>-0.139562</td>
      <td>-0.0192806</td>
      <td>-0.00297725</td>
      <td>-0.000454754</td>
      <td>-6.68248e-05</td>
      <td>-9.25334e-06</td>
      <td>-1.15946e-06</td>
      <td>-1.16884e-07</td>
      <td>-4.36908e-09</td>
      <td>2.25334e-09</td>
      <td>9.63303e-10</td>
      <td>2.76392e-10</td>
      <td>6.88942e-11</td>
      <td>1.59799e-11</td>
      <td>3.5489e-12</td>
    </tr>
    <tr>
      <th>alpha_5</th>
      <td>15.1172</td>
      <td>0.518723</td>
      <td>-0.0582006</td>
      <td>-0.00837821</td>
      <td>-0.00138295</td>
      <td>-0.000232191</td>
      <td>-3.88475e-05</td>
      <td>-6.45088e-06</td>
      <td>-1.06181e-06</td>
      <td>-1.72978e-07</td>
      <td>-2.78197e-08</td>
      <td>-4.3993e-09</td>
      <td>-6.79691e-10</td>
      <td>-1.01523e-10</td>
      <td>-1.43857e-11</td>
      <td>-1.85899e-12</td>
      <td>-1.96694e-13</td>
    </tr>
    <tr>
      <th>alpha_10</th>
      <td>19.1028</td>
      <td>0.374275</td>
      <td>-0.0365327</td>
      <td>-0.00538996</td>
      <td>-0.000918146</td>
      <td>-0.000160143</td>
      <td>-2.80234e-05</td>
      <td>-4.90295e-06</td>
      <td>-8.57487e-07</td>
      <td>-1.49942e-07</td>
      <td>-2.62176e-08</td>
      <td>-4.58355e-09</td>
      <td>-8.01062e-10</td>
      <td>-1.39912e-10</td>
      <td>-2.44122e-11</td>
      <td>-4.25339e-12</td>
      <td>-7.39658e-13</td>
    </tr>
    <tr>
      <th>alpha_20</th>
      <td>23.4027</td>
      <td>0.255211</td>
      <td>-0.0219041</td>
      <td>-0.0032993</td>
      <td>-0.000576093</td>
      <td>-0.000103338</td>
      <td>-1.86481e-05</td>
      <td>-3.37252e-06</td>
      <td>-6.11003e-07</td>
      <td>-1.10907e-07</td>
      <td>-2.01724e-08</td>
      <td>-3.67658e-09</td>
      <td>-6.71424e-10</td>
      <td>-1.22851e-10</td>
      <td>-2.25185e-11</td>
      <td>-4.13458e-12</td>
      <td>-7.60339e-13</td>
    </tr>
  </tbody>
</table>
</div>



# Lasso


```python
def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret
```


```python
#Initialize predictors to all 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)
```

    c:\users\kim98\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    c:\users\kim98\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    c:\users\kim98\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    c:\users\kim98\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    


![png](https://drive.google.com/uc?export=view&id=1WlxEp6EhiQy-quTvk77EGV5MVB0_KZrB)



```python
coef_matrix_lasso
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
      <th>rss</th>
      <th>intercept</th>
      <th>coef_x_1</th>
      <th>coef_x_2</th>
      <th>coef_x_3</th>
      <th>coef_x_4</th>
      <th>coef_x_5</th>
      <th>coef_x_6</th>
      <th>coef_x_7</th>
      <th>coef_x_8</th>
      <th>coef_x_9</th>
      <th>coef_x_10</th>
      <th>coef_x_11</th>
      <th>coef_x_12</th>
      <th>coef_x_13</th>
      <th>coef_x_14</th>
      <th>coef_x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_1e-15</th>
      <td>1.60382</td>
      <td>-0.998601</td>
      <td>2.25165</td>
      <td>-0.467254</td>
      <td>-0.107276</td>
      <td>0.0035944</td>
      <td>0.0031348</td>
      <td>0.000532336</td>
      <td>4.10856e-05</td>
      <td>-3.59937e-06</td>
      <td>-1.9946e-06</td>
      <td>-4.36419e-07</td>
      <td>-6.52648e-08</td>
      <td>-5.93735e-09</td>
      <td>2.66977e-10</td>
      <td>2.839e-10</td>
      <td>8.38043e-11</td>
    </tr>
    <tr>
      <th>alpha_1e-10</th>
      <td>1.60381</td>
      <td>-0.998591</td>
      <td>2.25164</td>
      <td>-0.46725</td>
      <td>-0.107275</td>
      <td>0.00359393</td>
      <td>0.00313482</td>
      <td>0.000532333</td>
      <td>4.10844e-05</td>
      <td>-3.59842e-06</td>
      <td>-1.99463e-06</td>
      <td>-4.36422e-07</td>
      <td>-6.52645e-08</td>
      <td>-5.93714e-09</td>
      <td>2.66701e-10</td>
      <td>2.8391e-10</td>
      <td>8.38083e-11</td>
    </tr>
    <tr>
      <th>alpha_1e-08</th>
      <td>1.60378</td>
      <td>-0.997543</td>
      <td>2.25028</td>
      <td>-0.466899</td>
      <td>-0.107158</td>
      <td>0.00354732</td>
      <td>0.00313684</td>
      <td>0.000532049</td>
      <td>4.09607e-05</td>
      <td>-3.50486e-06</td>
      <td>-1.99789e-06</td>
      <td>-4.36705e-07</td>
      <td>-6.52351e-08</td>
      <td>-5.91586e-09</td>
      <td>2.39359e-10</td>
      <td>2.84976e-10</td>
      <td>8.42054e-11</td>
    </tr>
    <tr>
      <th>alpha_1e-05</th>
      <td>1.63169</td>
      <td>-0.931047</td>
      <td>2.40115</td>
      <td>-0.739295</td>
      <td>-0.00340879</td>
      <td>-0</td>
      <td>0.000860296</td>
      <td>0.000538858</td>
      <td>0</td>
      <td>0</td>
      <td>-0</td>
      <td>-7.0793e-08</td>
      <td>-1.17802e-07</td>
      <td>-0</td>
      <td>-0</td>
      <td>0</td>
      <td>7.69599e-11</td>
    </tr>
    <tr>
      <th>alpha_0.0001</th>
      <td>1.92859</td>
      <td>0.386579</td>
      <td>0.846562</td>
      <td>-0.274709</td>
      <td>-0.0158377</td>
      <td>-0</td>
      <td>0</td>
      <td>1.49988e-05</td>
      <td>5.03244e-05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-2.36725e-11</td>
    </tr>
    <tr>
      <th>alpha_0.001</th>
      <td>2.72778</td>
      <td>1.27109</td>
      <td>-0</td>
      <td>-0.127089</td>
      <td>-0</td>
      <td>-0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.8675e-08</td>
      <td>1.78698e-08</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>alpha_0.01</th>
      <td>5.04889</td>
      <td>1.6868</td>
      <td>-0.533817</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
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
      <th>alpha_1</th>
      <td>36.0406</td>
      <td>0.0283987</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>alpha_5</th>
      <td>36.0406</td>
      <td>0.0283987</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>alpha_10</th>
      <td>36.0406</td>
      <td>0.0283987</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
    </tr>
  </tbody>
</table>
</div>



alpha(강의자료 lambda)값이 커질수록 x값에 대해 덜 민감한, 더 단순한 모델, RSS 증가, 계수 = 0
