---
layout: post
title: University site crawling.
categories:
  - Programming
tags:
  - programming
  - crawling
last_modified_at: 2020-05-17
use_math: true
---
### Crawling
university site url/region crawling  

```python
import requests 
from bs4 import BeautifulSoup as bs
import time
import re
import pandas as pd
from datetime import datetime
```


```python
from google.colab import drive 
drive.mount('/content/gdrive/')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive/
    


```python
data2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/d.xlsx.csv', encoding='euc-kr')
```


```python
data2
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
      <th>대학</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>강릉원주대학교</td>
    </tr>
    <tr>
      <th>1</th>
      <td>경북대학교</td>
    </tr>
    <tr>
      <th>2</th>
      <td>경희대학교</td>
    </tr>
    <tr>
      <th>3</th>
      <td>광주보건대학교</td>
    </tr>
    <tr>
      <th>4</th>
      <td>국민대학교</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>152</th>
      <td>한동대학교</td>
    </tr>
    <tr>
      <th>153</th>
      <td>한림대학교</td>
    </tr>
    <tr>
      <th>154</th>
      <td>한서대학교</td>
    </tr>
    <tr>
      <th>155</th>
      <td>한성대학교</td>
    </tr>
    <tr>
      <th>156</th>
      <td>한일장신대학교</td>
    </tr>
  </tbody>
</table>
<p>157 rows × 1 columns</p>
</div>




```python
a=[]
for i in range(len(data2)):
  a.append(data2.대학[i])
```


```python
region=[]
for name in a:
  url = 'https://www.academyinfo.go.kr/search/search.do'
  params = {
        'kwd': name,
        'category': 'TOTAL',
        'pageNum': '1',
        'pageSize': '3',
        'schlSize': ''}
  headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
  res = requests.post(url=url, params=params, headers=headers)
  soup = bs(res.text,'html.parser')
  temp = soup.select('#contentsWrap > div > div > div.college-search-result > div.college-data-wrap > div.info-box')
  temp1 = temp[0].findAll('span')
  region.append(temp1[2].getText().split()[2])
  print('---{}complete-----'.format(str(name)))
  time.sleep(0.2)
appended_df = pd.DataFrame({'title': a, 'region':region})
```


```python
appended_df
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
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>강릉원주대학교</td>
      <td>강원도</td>
    </tr>
    <tr>
      <th>1</th>
      <td>경북대학교</td>
      <td>대구광역시</td>
    </tr>
    <tr>
      <th>2</th>
      <td>경희대학교</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>3</th>
      <td>광주보건대학교</td>
      <td>광주광역시</td>
    </tr>
    <tr>
      <th>4</th>
      <td>국민대학교</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>152</th>
      <td>한동대학교</td>
      <td>경상북도</td>
    </tr>
    <tr>
      <th>153</th>
      <td>한림대학교</td>
      <td>강원도</td>
    </tr>
    <tr>
      <th>154</th>
      <td>한서대학교</td>
      <td>충청남도</td>
    </tr>
    <tr>
      <th>155</th>
      <td>한성대학교</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>156</th>
      <td>한일장신대학교</td>
      <td>전라북도</td>
    </tr>
  </tbody>
</table>
<p>157 rows × 2 columns</p>
</div>




```python
appended_df.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/d_region.csv', encoding='euc-kr')
```


```python
data2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/c.csv', encoding='euc-kr')
a=[]
for i in range(len(data2)):
  a.append(data2.대학교[i])
```


```python
url1=[]
b=a[:100]
for name in b:
  url = 'https://www.academyinfo.go.kr/search/search.do'
  params = {
        'kwd': name,
        'category': 'TOTAL',
        'pageNum': '1',
        'pageSize': '3',
        'schlSize': ''}
  headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
  res = requests.post(url=url, params=params, headers=headers)
  soup = bs(res.text,'html.parser')
  temp = soup.select('#contentsWrap > div > div > div.college-search-result > div.college-data-wrap > div.info-box')
  temp1 = temp[0].findAll('a')
  for link in temp1:
    if 'href' in link.attrs: 
      url1.append(link.attrs['href'].replace("//",""))
  time.sleep(0.1)
  print('---{}complete-----'.format(str(name)))
appended_dfa = pd.DataFrame({'title': b, 'url':url1})
```

    ---한국교원대학교complete-----
    ---경인교육대학교complete-----
    ---공주교육대학교 complete-----
    ---광주교육대학교complete-----
    ---대구교육대학교complete-----
    ---부산교육대학교complete-----
    ---서울교육대학교complete-----
    ---전주교육대학교complete-----
    ---진주교육대학교complete-----
    ---청주교육대학교 complete-----
    ---춘천교육대학교complete-----
    ---제주대학교complete-----
    ---이화여자대학교complete-----
    ---한국교원대학교complete-----
    ---경인교육대학교complete-----
    ---공주교육대학교 complete-----
    ---광주교육대학교complete-----
    ---대구교육대학교complete-----
    ---부산교육대학교complete-----
    ---서울교육대학교complete-----
    ---전주교육대학교complete-----
    ---진주교육대학교complete-----
    ---청주교육대학교 complete-----
    ---춘천교육대학교complete-----
    ---이화여자대학교complete-----
    ---한국복지대학교complete-----
    ---경북대학교complete-----
    ---경상대학교complete-----
    ---공주대학교complete-----
    ---목포대학교complete-----
    ---부산대학교complete-----
    ---서울대학교 complete-----
    ---순천대학교complete-----
    ---안동대학교complete-----
    ---전남대학교complete-----
    ---전북대학교complete-----
    ---제주대학교complete-----
    ---충남대학교complete-----
    ---충북대학교complete-----
    ---한국교원대학교complete-----
    ---인천대학교complete-----
    ---강남대학교complete-----
    ---건국대학교complete-----
    ---경남대학교complete-----
    ---계명대학교complete-----
    ---고려대학교complete-----
    ---가톨릭관동대학교complete-----
    ---한국복지대학교complete-----
    ---대구가톨릭대학교complete-----
    ---대구대학교complete-----
    ---동국대학교complete-----
    ---목원대학교complete-----
    ---상명대학교complete-----
    ---서원대학교complete-----
    ---성결대학교complete-----
    ---성균관대학교complete-----
    ---성신여자대학교complete-----
    ---신라대학교complete-----
    ---영남대학교complete-----
    ---우석대학교complete-----
    ---원광대학교complete-----
    ---이화여자대학교complete-----
    ---인하대학교complete-----
    ---전주대학교complete-----
    ---조선대학교complete-----
    ---중앙대학교complete-----
    ---청주대학교complete-----
    ---한국복지대학교complete-----
    ---한남대학교complete-----
    ---한양대학교complete-----
    ---홍익대학교complete-----
    ---부경대학교complete-----
    ---경성대학교complete-----
    ---고신대학교complete-----
    ---국민대학교complete-----
    ---한국복지대학교complete-----
    ---동아대학교complete-----
    ---배재대학교complete-----
    ---서울신학대학교complete-----
    ---서울여자대학교complete-----
    ---세한대학교complete-----
    ---숙명여자대학교complete-----
    ---안양대학교complete-----
    ---연세대학교complete-----
    ---장로회신학대학교complete-----
    ---총신대학교complete-----
    ---강릉원주대학교complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---경북대학교complete-----
    ---경상대학교complete-----
    ---공주대학교complete-----
    ---군산대학교complete-----
    ---목포대학교complete-----
    ---부경대학교complete-----
    ---부산대학교complete-----
    ---서울과학기술대학교complete-----
    ---서울대학교 complete-----
    ---순천대학교complete-----
    


```python
appended_dfa.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/c_url1.csv', encoding='euc-kr')
```


```python
url1=[]
c=a[100:200]
for name in c:
  url = 'https://www.academyinfo.go.kr/search/search.do'
  params = {
        'kwd': name,
        'category': 'TOTAL',
        'pageNum': '1',
        'pageSize': '3',
        'schlSize': ''}
  headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
  res = requests.post(url=url, params=params, headers=headers)
  soup = bs(res.text,'html.parser')
  temp = soup.select('#contentsWrap > div > div > div.college-search-result > div.college-data-wrap > div.info-box')
  temp1 = temp[0].findAll('a')
  for link in temp1:
    if 'href' in link.attrs: 
      url1.append(link.attrs['href'].replace("//",""))
  time.sleep(0.1)
  print('---{}complete-----'.format(str(name)))
appended_dfb = pd.DataFrame({'title': c, 'url':url1})
```

    ---안동대학교complete-----
    ---전남대학교complete-----
    ---한국복지대학교complete-----
    ---전북대학교complete-----
    ---제주대학교complete-----
    ---창원대학교complete-----
    ---충남대학교complete-----
    ---충북대학교complete-----
    ---한경대학교complete-----
    ---한국교통대학교complete-----
    ---한국체육대학교complete-----
    ---한국해양대학교complete-----
    ---한밭대학교complete-----
    ---서울시립대학교complete-----
    ---가천대학교complete-----
    ---가톨릭대학교 complete-----
    ---강남대학교complete-----
    ---건국대학교complete-----
    ---한국복지대학교complete-----
    ---건양대학교complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---경남대학교complete-----
    ---경성대학교complete-----
    ---경일대학교complete-----
    ---경희대학교complete-----
    ---계명대학교complete-----
    ---고려대학교complete-----
    ---한국복지대학교complete-----
    ---고신대학교complete-----
    ---가톨릭관동대학교complete-----
    ---광주대학교complete-----
    ---광주여자대학교complete-----
    ---국민대학교complete-----
    ---남부대학교complete-----
    ---남서울대학교complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---대구가톨릭대학교complete-----
    ---대구대학교complete-----
    ---대구한의대학교complete-----
    ---대전대학교complete-----
    ---대진대학교complete-----
    ---덕성여자대학교complete-----
    ---동국대학교complete-----
    ---한국복지대학교complete-----
    ---동덕여자대학교complete-----
    ---동명대학교complete-----
    ---동서대학교complete-----
    ---동신대학교complete-----
    ---동아대학교complete-----
    ---동의대학교complete-----
    ---명지대학교complete-----
    ---목원대학교complete-----
    ---배재대학교complete-----
    ---백석대학교complete-----
    ---부산외국어대학교complete-----
    ---삼육대학교complete-----
    ---상명대학교complete-----
    ---상지대학교complete-----
    ---서강대학교complete-----
    ---서울신학대학교complete-----
    ---서울여자대학교complete-----
    ---서원대학교complete-----
    ---성결대학교complete-----
    ---성균관대학교complete-----
    ---성신여자대학교complete-----
    ---세명대학교complete-----
    ---세종대학교complete-----
    ---수원대학교complete-----
    ---숙명여자대학교complete-----
    ---순천향대학교complete-----
    ---숭실대학교complete-----
    ---신라대학교complete-----
    ---아주대학교complete-----
    ---안양대학교complete-----
    ---연세대학교complete-----
    ---한국복지대학교complete-----
    ---영남대학교complete-----
    ---영남신학대학교complete-----
    ---유원대학교complete-----
    ---용인대학교complete-----
    ---우석대학교complete-----
    ---우송대학교complete-----
    ---울산대학교complete-----
    ---원광대학교complete-----
    ---이화여자대학교complete-----
    ---인제대학교complete-----
    ---전주대학교complete-----
    ---조선대학교complete-----
    ---중부대학교complete-----
    ---중앙대학교complete-----
    ---한국복지대학교complete-----
    ---청주대학교complete-----
    ---초당대학교complete-----
    ---한국국제대학교complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---한남대학교complete-----
    ---한라대학교complete-----
    


```python
appended_dfb.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/c_url2.csv', encoding='euc-kr')
```


```python
url1=[]
d=a[200:300]
for name in d:
  url = 'https://www.academyinfo.go.kr/search/search.do'
  params = {
        'kwd': name,
        'category': 'TOTAL',
        'pageNum': '1',
        'pageSize': '3',
        'schlSize': ''}
  headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
  res = requests.post(url=url, params=params, headers=headers)
  soup = bs(res.text,'html.parser')
  temp = soup.select('#contentsWrap > div > div > div.college-search-result > div.college-data-wrap > div.info-box')
  temp1 = temp[0].findAll('a')
  for link in temp1:
    if 'href' in link.attrs: 
      url1.append(link.attrs['href'].replace("//",""))
  time.sleep(0.1)
  print('---{}complete-----'.format(str(name)))
appended_dfc = pd.DataFrame({'title': d, 'url':url1})
```

    ---한림대학교complete-----
    ---한성대학교complete-----
    ---한양대학교complete-----
    ---한국복지대학교complete-----
    ---한중대학교complete-----
    ---협성대학교complete-----
    ---호남대학교complete-----
    ---호서대학교complete-----
    ---홍익대학교complete-----
    ---청운대학교complete-----
    ---강릉원주대학교 교육대학원complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---경북대학교 교육대학원complete-----
    ---경상대학교 교육대학원complete-----
    ---공주대학교 교육대학원complete-----
    ---군산대학교 교육대학원complete-----
    ---목포대학교 교육대학원complete-----
    ---부경대학교 교육대학원complete-----
    ---부산대학교 교육대학원complete-----
    ---서울대학교complete-----
    ---순천대학교 교육대학원complete-----
    ---안동대학교 교육대학원complete-----
    ---전남대학교 교육대학원complete-----
    ---전북대학교 교육대학원complete-----
    ---제주대학교 교육대학원complete-----
    ---창원대학교 교육대학원complete-----
    ---충남대학교 교육대학원complete-----
    ---충북대학교 교육대학원complete-----
    ---한국교원대학교 교육대학원complete-----
    ---서울시립대학교 교육대학원complete-----
    ---인천대학교 교육대학원complete-----
    ---가천대학교 교육대학원complete-----
    ---건국대학교 교육대학원complete-----
    ---한국복지대학교complete-----
    ---경남대학교 교육대학원complete-----
    ---경희대학교 교육대학원complete-----
    ---계명대학교 교육대학원complete-----
    ---고려대학교 교육대학원complete-----
    ---광주여자대학교 교육대학원complete-----
    ---국민대학교 교육대학원complete-----
    ---남부대학교 교육대학원complete-----
    ---단국대학교 교육대학원complete-----
    ---대구가톨릭대학교 교육대학원complete-----
    ---대구대학교 교육대학원complete-----
    ---대진대학교 교육대학원complete-----
    ---덕성여자대학교 교육대학원complete-----
    ---동국대학교 교육대학원complete-----
    ---동아대학교 교육대학원complete-----
    ---명지대학교 교육대학원complete-----
    ---부산외국어대학교 교육대학원complete-----
    ---상명대학교 교육대학원complete-----
    ---서강대학교 교육대학원complete-----
    ---성결대학교 교육대학원complete-----
    ---성균관대학교 교육대학원complete-----
    ---성신여자대학교 교육대학원complete-----
    ---세종대학교 교육대학원complete-----
    ---수원대학교 교육대학원complete-----
    ---숙명여자대학교 교육대학원complete-----
    ---순천향대학교 교육대학원complete-----
    ---신라대학교 교육대학원complete-----
    ---아주대학교 교육대학원complete-----
    ---연세대학교 교육대학원complete-----
    ---영남대학교 교육대학원complete-----
    ---용인대학교 교육대학원complete-----
    ---우석대학교 교육대학원complete-----
    ---울산대학교 교육대학원complete-----
    ---원광대학교 교육대학원complete-----
    ---이화여자대학교 교육대학원complete-----
    ---인제대학교 교육대학원complete-----
    ---인하대학교 교육대학원complete-----
    ---장로회신학대학교 교육대학원complete-----
    ---전주대학교 교육대학원complete-----
    ---조선대학교 교육대학원complete-----
    ---중앙대학교 교육대학원complete-----
    ---총신대학교 교육대학원complete-----
    ---한국복지대학교complete-----
    ---한남대학교 교육대학원complete-----
    ---한서대학교 교육대학원complete-----
    ---한양대학교 교육대학원complete-----
    ---호남대학교 교육대학원complete-----
    ---강릉원주대학교 교육대학원complete-----
    ---한국복지대학교complete-----
    ---한국복지대학교complete-----
    ---경북대학교 교육대학원complete-----
    ---경상대학교 교육대학원complete-----
    ---공주대학교 교육대학원complete-----
    ---군산대학교 교육대학원complete-----
    ---금오공과대학교 교육대학원complete-----
    ---목포대학교 교육대학원complete-----
    ---부경대학교 교육대학원complete-----
    ---부산대학교 교육대학원complete-----
    ---순천대학교 교육대학원complete-----
    ---안동대학교 교육대학원complete-----
    ---전남대학교 교육대학원complete-----
    ---전북대학교 교육대학원complete-----
    ---제주대학교 교육대학원complete-----
    ---창원대학교 교육대학원complete-----
    ---충남대학교 교육대학원complete-----
    ---충북대학교 교육대학원complete-----
    


```python
appended_dfc.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/c_url3.csv', encoding='euc-kr')
```


```python
url1=[]
e=a[300:]
for name in e:
  url = 'https://www.academyinfo.go.kr/search/search.do'
  params = {
        'kwd': name,
        'category': 'TOTAL',
        'pageNum': '1',
        'pageSize': '3',
        'schlSize': ''}
  headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
  res = requests.post(url=url, params=params, headers=headers)
  soup = bs(res.text,'html.parser')
  temp = soup.select('#contentsWrap > div > div > div.college-search-result > div.college-data-wrap > div.info-box')
  temp1 = temp[0].findAll('a')
  for link in temp1:
    if 'href' in link.attrs: 
      url1.append(link.attrs['href'].replace("//",""))
  time.sleep(0.1)
  print('---{}complete-----'.format(str(name)))
appended_dfd = pd.DataFrame({'title': e, 'url':url1})
```

    ---한국교통대학교 교육대학원complete-----
    ---한국체육대학교 교육대학원complete-----
    ---한국해양대학교 교육대학원complete-----
    ---한국교원대학교 교육대학원complete-----
    ---서울시립대학교 교육대학원complete-----
    ---인천대학교 교육대학원complete-----
    ---가천대학교 교육대학원complete-----
    ---가톨릭대학교  교육대학원complete-----
    ---강남대학교 교육대학원complete-----
    ---건국대학교 교육대학원complete-----
    ---한국복지대학교complete-----
    ---경남대학교 교육대학원complete-----
    ---경성대학교 교육대학원complete-----
    ---경희대학교 교육대학원complete-----
    ---계명대학교 교육대학원complete-----
    ---고려대학교 교육대학원complete-----
    ---고신대학교 교육대학원complete-----
    ---가톨릭관동대학교 교육대학원complete-----
    ---광운대학교 교육대학원complete-----
    ---광주여자대학교 교육대학원complete-----
    ---국민대학교 교육대학원complete-----
    ---극동대학교 교육대학원complete-----
    ---나사렛대학교 교육대학원complete-----
    ---남부대학교 교육대학원complete-----
    ---단국대학교 교육대학원complete-----
    ---대구가톨릭대학교 교육대학원complete-----
    ---대구대학교 교육대학원complete-----
    ---대전대학교 교육대학원complete-----
    ---대진대학교 교육대학원complete-----
    ---덕성여자대학교 교육대학원complete-----
    ---동국대학교 교육대학원complete-----
    ---동덕여자대학교 교육대학원complete-----
    ---동신대학교 교육대학원complete-----
    ---동아대학교 교육대학원complete-----
    ---동양대학교 교육대학원complete-----
    ---동의대학교 교육대학원complete-----
    ---명지대학교 교육대학원complete-----
    ---배재대학교 교육대학원complete-----
    ---백석대학교 교육대학원complete-----
    ---부산외국어대학교 교육대학원complete-----
    ---상명대학교 교육대학원complete-----
    ---상지대학교 교육대학원complete-----
    ---서강대학교 교육대학원complete-----
    ---서울여자대학교 교육대학원complete-----
    ---서원대학교 교육대학원complete-----
    ---선문대학교 교육대학원complete-----
    ---성결대학교 교육대학원complete-----
    ---성공회대학교 교육대학원complete-----
    ---성균관대학교 교육대학원complete-----
    ---성신여자대학교 교육대학원complete-----
    ---세종대학교 교육대학원complete-----
    ---수원대학교 교육대학원complete-----
    ---숙명여자대학교 교육대학원complete-----
    ---순천향대학교 교육대학원complete-----
    ---숭실대학교 교육대학원complete-----
    ---신라대학교 교육대학원complete-----
    ---아세아연합신학대학교 교육대학원complete-----
    ---아주대학교 교육대학원complete-----
    ---안양대학교 교육대학원complete-----
    ---연세대학교 교육대학원complete-----
    ---영남대학교 교육대학원complete-----
    ---용인대학교 교육대학원complete-----
    ---우석대학교 교육대학원complete-----
    ---울산대학교 교육대학원complete-----
    ---원광대학교 교육대학원complete-----
    ---위덕대학교 교육대학원complete-----
    ---이화여자대학교 교육대학원complete-----
    ---인제대학교 교육대학원complete-----
    ---인하대학교 교육대학원complete-----
    ---장로회신학대학교 교육대학원complete-----
    ---전주대학교 교육대학원complete-----
    ---조선대학교 교육대학원complete-----
    ---중부대학교 교육대학원complete-----
    ---중앙대학교 교육대학원complete-----
    ---총신대학교 교육대학원complete-----
    ---추계예술대학교 교육대학원complete-----
    ---한국복지대학교complete-----
    ---한남대학교 교육대학원complete-----
    ---한동대학교 교육대학원complete-----
    ---한서대학교 교육대학원complete-----
    ---한성대학교 교육대학원complete-----
    ---한신대학교 교육대학원complete-----
    ---한양대학교 교육대학원complete-----
    ---협성대학교 교육대학원complete-----
    ---호남대학교 교육대학원complete-----
    ---호서대학교 교육대학원complete-----
    ---홍익대학교 교육대학원complete-----
    


```python
appended_dfd.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/c_ur4.csv', encoding='euc-kr')
```
