---
layout: post
title:  "nlp_Word Embedding using crawled movie reviews."
description: natural language processing.
date:   2020-03-15
use_math: true
---
### Word Embedding using crawled movie reviews.
natural language processing

## Crawling


```python
import requests 
from bs4 import BeautifulSoup as bs
```


```python
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
```


```python
def get_data(name):
  title_list = []
  genre_list = []
  created_at_list = []
  review_text_list = []
  score_list = []

  resulted_df = []

  url_code = 'https://movie.naver.com/movie/search/result.nhn?query='+str(name)+'&section=all&ie=utf8'
  res_code = requests.get(url_code)
  soup_code = bs(res_code.content, 'html.parser')
  temp = soup_code.find('ul', {'class': 'search_list_1'})
  temp1 = temp.findAll('a')
  code = re.sub(r"[^0-9]+", " ",temp1[0].attrs['href']).strip()
  time.sleep(1)

  url_pre ='https://movie.naver.com/movie/bi/mi/point.nhn?code='+str(code)
  res_pre = requests.get(url_pre) 
  soup_pre = bs(res_pre.content, 'html.parser')
  score_result_pre = soup_pre.find('strong', {'class': 'h_movie2'})
  date = re.sub(r"[^0-9]+", " ",score_result_pre.text).strip()

  score_result_pre1 = soup_pre.find('dl', {'class': 'info_spec'})
  genre = score_result_pre1.findAll('a')[0].text
  time.sleep(1)

  for i in range(100):
    url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='+str(code)+'&type=after&onlyActualPointYn=Y&onlySpoilerPointYn=N&order=sympathyScore&page='+str(i+1)

    res = requests.get(url)
    soup = bs(res.content, 'html.parser')
    score_result = soup.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')
  
    for li in lis:
        title_list.append(name)
        genre_list.append(genre)
        created_at_list.append(date)
        review_text_list.append(re.sub(r"[^^0-9a-zA-Zㄱ-힗]+", " ",li.findAll('span')[3].getText()).strip())
        score_list.append(li.findAll('em')[0].text)
    time.sleep(0.5)      
  appended_df = pd.DataFrame({'title': title_list, 'year':created_at_list, 'genre': genre_list, 'score':score_list, 'review':review_text_list})
  

  resulted_df.append(appended_df)
  final_df = pd.concat(resulted_df)
  time.sleep(2)
  return final_df

```

Using above function defined as 'get_data', we can crawl the naver movie review data.


```python
names = ['내부자들', '가장 보통의 연애', '그것만이 내 세상', '남한산성', '해적: 바다로 간 산적', '돈', '뺑반', '도어락', '봉오동 전투', '곤지암']
final = []

for name in names:
  final.append(get_data(name))
  print('---{}complete-----'.format(str(name)))
  
final_netizen = pd.concat(final)
final_netizen

```

I crawled the 10000-sized netizen reviews from 10 movies.

## Proprocessing data

### mount google drive


```python
from google.colab import drive 
drive.mount('/content/gdrive/')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive/
    

### load data


```python
import pandas as pd

a = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/naver_netizen_token_dawis.csv', encoding='euc-kr')
b = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/review_token_남승희.csv')
c = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/tokenized_review_임서하.csv')
c.drop('Unnamed: 0', axis=1, inplace=True)
d = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/movie_tokenized_김연강.csv', encoding='euc-kr')
```


```python
data = pd.concat([a,b,c,d])
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
data.drop('token', axis=1, inplace=True)
data
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
      <th>title</th>
      <th>year</th>
      <th>genre</th>
      <th>score</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>7</td>
      <td>그래도이병헌은이병헌이다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>스캔들때문에 배우생활 접기에는 정말 아까운 배우다 이병헌 백윤식이고 조승우고 영화끝...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>이병헌은 진짜 눈빛이 살아있다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>9</td>
      <td>생각보다 재미있었고 이병헌씨는 정말연기는 정말 잘하시는듯 영화에서 제목이 반전임</td>
    </tr>
    <tr>
      <th>4</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>9</td>
      <td>진부하지만 강렬하다</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39399</th>
      <td>국제시장</td>
      <td>2014</td>
      <td>드라마</td>
      <td>9</td>
      <td>웃픈영화입니다 가슴이 먹먹해지는</td>
    </tr>
    <tr>
      <th>39400</th>
      <td>국제시장</td>
      <td>2014</td>
      <td>드라마</td>
      <td>10</td>
      <td>2014년 최고의 영화를 봤네요 너무 재미있고 감동있게 봤습니다 아버지 시대를 배경...</td>
    </tr>
    <tr>
      <th>39401</th>
      <td>국제시장</td>
      <td>2014</td>
      <td>드라마</td>
      <td>10</td>
      <td>보다가 소름돋는부분이 너무만앗어여ㅠㅠㅠ굿</td>
    </tr>
    <tr>
      <th>39402</th>
      <td>국제시장</td>
      <td>2014</td>
      <td>드라마</td>
      <td>10</td>
      <td>슬프고 웃기다 슬프고 재밋다가 슬프고</td>
    </tr>
    <tr>
      <th>39403</th>
      <td>국제시장</td>
      <td>2014</td>
      <td>드라마</td>
      <td>10</td>
      <td>자식들을 위해 희생하시는 아버지 모성애보다 덜 눈물샘을 자극하지만 영화가 끝난휴 아...</td>
    </tr>
  </tbody>
</table>
<p>39404 rows × 5 columns</p>
</div>




```python
data['review'] = data['review'].astype(str)
for i in range(len(data['review'])):
  data['review'][i] = data['review'][i].encode('utf-8')
```

since '**euc-kr**' type provokes some problems in implementing tokenizing, I encode the data type to '**utf-8**'

### install konlpy


```python
!pip install konlpy
```


```python
from konlpy.tag import Okt  
okt=Okt()  
```

### tokenize the reviews and elimination of stop words


```python
data["token"] = 1
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
for i in range(len(data["review"])):
  data["token"][i] =  okt.morphs(data["review"][i], stem=True)
  data["token"][i] = [word for word in data["token"][i] if not word in stopwords]
```


```python
for i in range(len(data['review'])):
  data['review'][i] = data['review'][i].decode('utf-8')
```

Decode the data back to original version.


```python
tokenized_data = []
for i in range(len(data['token'])):
  tokenized_data.append(data['token'][i])
```


```python
import matplotlib.pyplot as plt
print('Maximum length of the review :',max(len(l) for l in tokenized_data))
print('Average length of the review :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```

    Maximum length of the review : 72
    Average length of the review : 10.056999289412243
    


![png](https://drive.google.com/uc?export=view&id=1DyiEqGZTKApQXIIOFJzG81aN-LcOY70e)


## Word Embedding

### Word2vec


```python
from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
```


```python
model.wv.vectors.shape
```




    (4617, 100)




```python
print(model.wv.most_similar("이병헌"))
```

    [('황정민', 0.9475492238998413), ('송강호', 0.9394603967666626), ('류준열', 0.9342226982116699), ('유아인', 0.9299998879432678), ('조정석', 0.9269593954086304), ('손예진', 0.9230944514274597), ('윤계상', 0.9214309453964233), ('씨', 0.9209078550338745), ('공효진', 0.9182060360908508), ('김윤석', 0.915282130241394)]
    

    /usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    

In the process of getting some *relevent words* with "이병헌" by using **Word2Vec**, the result is revealed as above.

### GloVe


```python
pip install glove_python
```


```python
from glove import Corpus, Glove

corpus = Corpus() 
corpus.fit(tokenized_data, window=5)
# make Co-occurrence Matrix from tokenized_data

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# Thread : 4, epoch = 20
```

    Performing 20 training epochs with 4 threads
    Epoch 0
    Epoch 1
    Epoch 2
    Epoch 3
    Epoch 4
    Epoch 5
    Epoch 6
    Epoch 7
    Epoch 8
    Epoch 9
    Epoch 10
    Epoch 11
    Epoch 12
    Epoch 13
    Epoch 14
    Epoch 15
    Epoch 16
    Epoch 17
    Epoch 18
    Epoch 19
    


```python
model_result1=glove.most_similar("이병헌")
print(model_result1)
```

    [('황정민', 0.9404819115988845), ('씨', 0.9316294682363065), ('송강호', 0.9235389442938021), ('유아인', 0.9082707360454588)]
    

In the process of getting some *relevent words* with "이병헌" by using **GloVe**, the result is revealed as above.

### Elmo


```python
!pip install tensorflow-hub
```


```python
import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K

sess = tf.Session()
K.set_session(sess)
# initiate the session

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# download elmo from tensorflow-hub

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
```


```python
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
```


```python
data['detoken']=data['token'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
```

To implement Elmo, I detokenize the tokenized data to make sentences without stop words.


```python
data["title"] = data["title"].astype('category').cat.codes
y_data = list(data['title'])
X_data = list(data['detoken'])
```


```python
print(len(X_data))
n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)
print(n_of_train)
print(n_of_test)
```

    39404
    31523
    7881
    


```python
import numpy as np
X_train = np.asarray(X_data[:n_of_train]) #X_data saving former 31523 datas from total data
y_train = np.asarray(y_data[:n_of_train]) #y_data saving former 31523 datas from total data
X_test = np.asarray(X_data[n_of_train:]) #X_data saving latter 7881 datas from total data
y_test = np.asarray(y_data[n_of_train:]) #y_data saving latter 7881 datas from total data
```


```python
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]
# function for making the data move from keras to tensorflow and to keras again.
```


```python
from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text = Input(shape=(1,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
history = model.fit(X_train, y_train, epochs=1, batch_size=60)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    

    Epoch 1/1
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    

    31523/31523 [==============================] - 5104s 162ms/step - loss: nan - acc: 0.0052
    


```python
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```

    7881/7881 [==============================] - 1370s 174ms/step
    
     테스트 정확도: 0.0000
    
Unfortunately, the score is zero, which means that the data using in the Elmo process is needed to be refined through elaborate procedures. However, one noticing point in here is that, when I used only 10000-sized crawled data, the score was 0.5. It was also a low score, but make the result derived from using almost 40000-sized crawled data seem to be rather weird.   