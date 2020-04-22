---
layout: post
title: Make tagging using crawled movie reviews.
categories:
  - Programming
tags:
  - programming
  - word count
last_modified_at: 2020-04-22
use_math: true
---
### Word counting using crawled movie reviews.
natural language processing

## Mount Google Drive


```python
from google.colab import drive 
drive.mount('/content/gdrive/')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive/
    

## Load Data


```python
import pandas as pd

data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/naver_netizen_token_dawis.csv', encoding='euc-kr')

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
      <th>9995</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>9</td>
      <td>와 공포영화 보면서 빨리 끝났으면 하고 생각한 건 처음인듯 402호 402호 앙대 ...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>10</td>
      <td>정말 최고임 고등학생들만 아니라면 집중이 더 잘됐을텐데</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>8</td>
      <td>끝나기 전 30분이 고비써클귀신이 다했다</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>9</td>
      <td>재밌었는데 중후반까지 그냥 공포 코믹 같았구 후반이 진짜 하이라이트 ㅇㅇ</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>10</td>
      <td>한국 공포영화의 새로운 느낌을 받았다 결과는 뻔했지만 그 과정들이 매우 심장을 쫄깃...</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>




```python
data['review'] = data['review'].astype(str)
for i in range(len(data['review'])):
  data['review'][i] = data['review'][i].encode('utf-8')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    

## Install konlpy & Import Okt


```python
!pip install konlpy
```

    Requirement already satisfied: konlpy in /usr/local/lib/python3.6/dist-packages (0.5.2)
    Requirement already satisfied: colorama in /usr/local/lib/python3.6/dist-packages (from konlpy) (0.4.3)
    Requirement already satisfied: beautifulsoup4==4.6.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.6.0)
    Requirement already satisfied: tweepy>=3.7.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (3.8.0)
    Requirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.6/dist-packages (from konlpy) (1.18.2)
    Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.2.6)
    Requirement already satisfied: JPype1>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (0.7.3)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.3.0)
    Requirement already satisfied: requests>=2.11.1 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (2.21.0)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.12.0)
    Requirement already satisfied: PySocks>=1.5.7 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.7.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy) (3.1.0)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests>=2.11.1->tweepy>=3.7.0->konlpy) (2.8)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests>=2.11.1->tweepy>=3.7.0->konlpy) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests>=2.11.1->tweepy>=3.7.0->konlpy) (2020.4.5.1)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests>=2.11.1->tweepy>=3.7.0->konlpy) (3.0.4)
    


```python
from konlpy.tag import Okt  
okt=Okt()  
```

## Convert review data to list


```python
a = list(data.loc[data['title']=='내부자들']['review'])
b = list(data.loc[data['title']=='가장 보통의 연애']['review'])
c = list(data.loc[data['title']=='그것만이 내 세상']['review'])
d = list(data.loc[data['title']=='남한산성']['review'])
e = list(data.loc[data['title']=='해적: 바다로 간 산적']['review'])
f = list(data.loc[data['title']=='돈']['review'])
g = list(data.loc[data['title']=='뺑반']['review'])
h = list(data.loc[data['title']=='도어락']['review'])
i = list(data.loc[data['title']=='봉오동 전투']['review'])
j = list(data.loc[data['title']=='곤지암']['review'])

```


```python
movie_lists = [a,b,c,d,e,f,g,h,i,j]
for movie_list in movie_lists:
  print(movie_list)
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    

## Stemming & Lemmatization


```python
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
movie_lists = [a,b,c,d,e,f,g,h,i,j]
final = []
for movie_list in movie_lists:
  clean_words = []
  for review in movie_list:
    for word in okt.pos(review, stem=True): #어간 추출
     if word[1] in ['Noun', 'Adjective']:
       clean_words.append(word[0])
       clean_words = [word for word in clean_words if not word in stopwords]
      
  final.append(clean_words)
```


```python
final = pd.DataFrame(final)
final['total'] = final[final.columns].astype(str).apply(lambda x: ' '.join(x), axis = 1)
final = final[['total']]
final
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
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>이병헌 이병헌 스캔들 때문 배우 생활 접기 정말 아깝다 이병헌 백윤식 조승우 영화 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>연애 직장 것 공감 미치다 현실 대사 뼈 후려 살짝 아프다 발견 최근 코미디 로맨스...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>누군가 말 이병헌 연기 없다 이번 말 역시 증명 영화 윤여정 연기 한지민 연기 좋다...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>솔직하다 또 우려 똑같다 사극 정말 기대 이상 한국 영화 가장 완성 높다 명작 생각...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해적 고맙다 요즘 우울하다 시원하다 바다 화려하다 액션 웃음 일단 일부러 극장 보람...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>재미 도감 서스펜스 삼박자 잘만 오락 영화 돈 대해 많다 생각 여운 길다 남아 류준...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>이영화 돈 뺑소니 류준 연기 점점 기 맥혀 아주 내 지갑 뺑소니 당한 기분 직업 정...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>현실 공감 각심 생기 공효진 연기 화보 내내 몰입 몸 강 추강 추 낯선 사람 내 집...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>홍범도 장군 전투 아니다 독립군 모두 전투 훌륭하다 영화 눈물 외 싸움 같다 주인공...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>재밋게봣는데 낮 급식 존나 시끄럽다 개팰 번하다 호들갑 진짜 해설 진짜 개 무서움 ...</td>
    </tr>
  </tbody>
</table>
</div>



## Word Counting & Extracting top 20 words


```python
from collections import Counter
final["count"]=final['total'].apply(lambda x: Counter(x.split(' ')).most_common(21)) #extract top 20 words
for i in range(len(final)):
  last = []
  for j in range(len(final['count'][i])):
      last.append(final['count'][i][j][0])
  final['count'][i] = last
  final['count'][i] = [word for word in final['count'][i] if not word=='None'] #eraze None value
final
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
      <th>total</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>이병헌 이병헌 스캔들 때문 배우 생활 접기 정말 아깝다 이병헌 백윤식 조승우 영화 ...</td>
      <td>[이병헌, 연기, 영화, 재밌다, 조승우, 배우, 최고, 정말, 진짜, 좋다, 역시...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>연애 직장 것 공감 미치다 현실 대사 뼈 후려 살짝 아프다 발견 최근 코미디 로맨스...</td>
      <td>[영화, 재밌다, 좋다, 공효진, 연기, 연애, 있다, 공감, 김래원, 현실, 배우...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>누군가 말 이병헌 연기 없다 이번 말 역시 증명 영화 윤여정 연기 한지민 연기 좋다...</td>
      <td>[영화, 연기, 이병헌, 감동, 좋다, 재밌다, 배우, 박정민, 역시, 정말, 있다...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>솔직하다 또 우려 똑같다 사극 정말 기대 이상 한국 영화 가장 완성 높다 명작 생각...</td>
      <td>[영화, 좋다, 역사, 연기, 배우, 있다, 생각, 없다, 지루하다, 재밌다, 같다...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해적 고맙다 요즘 우울하다 시원하다 바다 화려하다 액션 웃음 일단 일부러 극장 보람...</td>
      <td>[영화, 재밌다, 재미있다, 좋다, 김남길, 있다, 유해진, 연기, 같다, 정말, ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>재미 도감 서스펜스 삼박자 잘만 오락 영화 돈 대해 많다 생각 여운 길다 남아 류준...</td>
      <td>[영화, 재밌다, 연기, 류준열, 좋다, 돈, 재미있다, 있다, 배우, 없다, 생각...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>이영화 돈 뺑소니 류준 연기 점점 기 맥혀 아주 내 지갑 뺑소니 당한 기분 직업 정...</td>
      <td>[연기, 영화, 재밌다, 류준열, 배우, 조정석, 좋다, 진짜, 스토리, 볼, 있다...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>현실 공감 각심 생기 공효진 연기 화보 내내 몰입 몸 강 추강 추 낯선 사람 내 집...</td>
      <td>[무섭다, 영화, 있다, 재밌다, 현실, 공효진, 볼, 답답하다, 같다, 생각, 연...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>홍범도 장군 전투 아니다 독립군 모두 전투 훌륭하다 영화 눈물 외 싸움 같다 주인공...</td>
      <td>[영화, 있다, 좋다, 꼭, 독립운동가, 존경, 독립, 이다, 정말, 재밌다, 연기...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>재밋게봣는데 낮 급식 존나 시끄럽다 개팰 번하다 호들갑 진짜 해설 진짜 개 무서움 ...</td>
      <td>[무섭다, 영화, 공포영화, 진짜, 없다, 같다, 귀신, 그냥, 생각, 공포, 재밌...</td>
    </tr>
  </tbody>
</table>
</div>



## Merge the data


```python
data['count'] =1
for i in range(len(data)):
  if data["title"][i]=='내부자들':
    data['count'][i] = final['count'][0]
  if data["title"][i]=='가장 보통의 연애':
    data['count'][i] = final['count'][1]
  if data["title"][i]=='그것만이 내 세상':
    data['count'][i] = final['count'][2]
  if data["title"][i]=='남한산성':
    data['count'][i] = final['count'][3]
  if data["title"][i]== '해적: 바다로 간 산적':
    data['count'][i] = final['count'][4]
  if data["title"][i]=='돈':
    data['count'][i] = final['count'][5]
  if data["title"][i]=='뺑반':
    data['count'][i] = final['count'][6]
  if data["title"][i]== '도어락':
    data['count'][i] = final['count'][7]
  if data["title"][i]== '봉오동 전투':
    data['count'][i] = final['count'][8]
  if data["title"][i]==  '곤지암':
    data['count'][i] = final['count'][9]
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    /usr/local/lib/python3.6/dist-packages/pandas/core/indexing.py:671: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_with_indexer(indexer, value)
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      app.launch_new_instance()
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:22: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    


```python
for i in range(len(data['review'])):
  data['review'][i] = data['review'][i].decode('utf-8')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    

## Tagging


```python
data['tag'] = data[['year', 'genre', 'count']].astype(str).apply(lambda x: ', '.join(x), axis = 1) #year+genre+top 20 words
data.drop(["count"], axis =1, inplace = True)

data['tag'] = data['tag'].map(lambda x : x.replace('[', ''))
data['tag'] = data['tag'].map(lambda x : x.replace("'", ''))
data['tag'] = data['tag'].map(lambda x : x.replace(']', ''))
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
      <th>tag</th>
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
      <td>2015, 범죄, 이병헌, 연기, 영화, 재밌다, 조승우, 배우, 최고, 정말, 진...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>스캔들때문에 배우생활 접기에는 정말 아까운 배우다 이병헌 백윤식이고 조승우고 영화끝...</td>
      <td>2015, 범죄, 이병헌, 연기, 영화, 재밌다, 조승우, 배우, 최고, 정말, 진...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>이병헌은 진짜 눈빛이 살아있다</td>
      <td>2015, 범죄, 이병헌, 연기, 영화, 재밌다, 조승우, 배우, 최고, 정말, 진...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>9</td>
      <td>생각보다 재미있었고 이병헌씨는 정말연기는 정말 잘하시는듯 영화에서 제목이 반전임</td>
      <td>2015, 범죄, 이병헌, 연기, 영화, 재밌다, 조승우, 배우, 최고, 정말, 진...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>9</td>
      <td>진부하지만 강렬하다</td>
      <td>2015, 범죄, 이병헌, 연기, 영화, 재밌다, 조승우, 배우, 최고, 정말, 진...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>9</td>
      <td>와 공포영화 보면서 빨리 끝났으면 하고 생각한 건 처음인듯 402호 402호 앙대 ...</td>
      <td>2017, 공포, 무섭다, 영화, 공포영화, 진짜, 없다, 같다, 귀신, 그냥, 생...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>10</td>
      <td>정말 최고임 고등학생들만 아니라면 집중이 더 잘됐을텐데</td>
      <td>2017, 공포, 무섭다, 영화, 공포영화, 진짜, 없다, 같다, 귀신, 그냥, 생...</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>8</td>
      <td>끝나기 전 30분이 고비써클귀신이 다했다</td>
      <td>2017, 공포, 무섭다, 영화, 공포영화, 진짜, 없다, 같다, 귀신, 그냥, 생...</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>9</td>
      <td>재밌었는데 중후반까지 그냥 공포 코믹 같았구 후반이 진짜 하이라이트 ㅇㅇ</td>
      <td>2017, 공포, 무섭다, 영화, 공포영화, 진짜, 없다, 같다, 귀신, 그냥, 생...</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>10</td>
      <td>한국 공포영화의 새로운 느낌을 받았다 결과는 뻔했지만 그 과정들이 매우 심장을 쫄깃...</td>
      <td>2017, 공포, 무섭다, 영화, 공포영화, 진짜, 없다, 같다, 귀신, 그냥, 생...</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 6 columns</p>
</div>




```python
data.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/tag_dawis.csv', index=False)
```


```python

```
