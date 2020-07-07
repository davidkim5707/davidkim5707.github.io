---
layout: post
title: Item-based Collaborative Filtering
categories:
  - Programming
  - Collaborative Filtering
tags:
  - programming
last_modified_at: 2020-07-12
use_math: true
---

by Lee seung-won from datascience-lab in Yonsei.  

### Item-based Collaborative Filtering

```python
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
```


```python
def na_handling(df, name_of_strategy):

        #list of stategies -> mean, mode, 0
        if name_of_strategy=='0':
            df.fillna(0, inplace=True)
            return df
        elif name_of_strategy=='mean':
            for i in range(0,409):
                df.loc[i].fillna(df.loc[i].mean(), inplace=True)
            return df
        elif name_of_strategy=='mode':
            for i in range(0,409):
                df.loc[i].fillna(df.loc[i].mode(), inplace=True)
            return df
        else:
            print("Wrong specified strategy")
            
ratings = pd.read_csv('/Users/michelle/Downloads/movie_rating.csv', encoding='euc-kr')
ratings = ratings.drop('Unnamed: 0', axis=1)
ratings = ratings.drop('id', axis=1)

na_handling(ratings, '0')

df_ibs = pd.DataFrame(index = ratings.columns, columns = ratings.columns)

for i in range(0,len(df_ibs.columns)) :
    for j in range(0,len(df_ibs.columns)) :
        df_ibs.iloc[i,j] = 1-cosine(ratings.iloc[:,i], ratings.iloc[:,j])
        
df_neighbours = pd.DataFrame(index = df_ibs.columns, columns = range(1,11))
 
# Loop through our similarity dataframe and fill in neighbouring item names
for i in range(0, len(df_ibs.columns)):
    df_neighbours.iloc[i,:10] = df_ibs.iloc[0:,i].sort_values(ascending=False)[:10].index
```
