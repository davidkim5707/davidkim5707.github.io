---
layout: post
title: Make tagging using crawled movie reviews(version.2).
categories:
  - Programming
tags:
  - programming
  - word count
last_modified_at: 2020-05-01
use_math: true
---
### Word counting using crawled movie reviews.
natural language processing

```python
import requests 
#from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
```

## Crawling the data from Naver movie


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
    url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='+str(code)+'&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=sympathyScore&page='+str(i+1)

    res = requests.get(url)
    soup = bs(res.content, 'html.parser')
    score_result = soup.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')
  
    for li in lis:
        title_list.append(name)
        genre_list.append(genre)
        created_at_list.append(date)
        if li.findAll('span')[2].getText()=="관람객":
          review_text_list.append(re.sub(r"[^^0-9a-zA-Zㄱ-힗]+", " ",li.findAll('span')[3].getText()).strip())
        else:
          review_text_list.append(re.sub(r"[^^0-9a-zA-Zㄱ-힗]+", " ",li.findAll('span')[2].getText()).strip())
        score_list.append(li.findAll('em')[0].text)
    time.sleep(0.5)      
  appended_df = pd.DataFrame({'title': title_list, 'year':created_at_list, 'genre': genre_list, 'score':score_list, 'review':review_text_list})
  

  resulted_df.append(appended_df)
  final_df = pd.concat(resulted_df)
  time.sleep(2)
  return final_df

```


```python
names = ['1917(1917)', '서치 아웃', '덕구', '가버나움', '어벤져스: 엔드게임', '인셉션', '작은 아씨들(Little Women)', '그린 북', '포드 V 페라리', '보헤미안 랩소디']
data = []

for name in names:
  data.append(get_data(name))
  print('---{}complete-----'.format(str(name)))
  
data2 = pd.concat(data)
data2
```

    ---1917(1917)complete-----
    ---서치 아웃complete-----
    ---덕구complete-----
    ---가버나움complete-----
    ---어벤져스: 엔드게임complete-----
    ---인셉션complete-----
    ---작은 아씨들(Little Women)complete-----
    ---그린 북complete-----
    ---포드 V 페라리complete-----
    ---보헤미안 랩소디complete-----
    




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
      <td>1917(1917)</td>
      <td>1917 2019</td>
      <td>드라마</td>
      <td>10</td>
      <td>이 영화는 미쳤다 넷플릭스가 일상화된 시대에 극장이 존재해야하는 이유를 증명해준다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1917(1917)</td>
      <td>1917 2019</td>
      <td>드라마</td>
      <td>8</td>
      <td>충무로 이거 어케하는거냐</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1917(1917)</td>
      <td>1917 2019</td>
      <td>드라마</td>
      <td>10</td>
      <td>아카데미에서 촬영상 음향효과상 시각효과상을 받은 이유가 고스란히 녹아있는 영화 IM...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1917(1917)</td>
      <td>1917 2019</td>
      <td>드라마</td>
      <td>10</td>
      <td>처절한 전쟁 속에서 한 남자를 영웅으로 만든 것은 훈장도 장군의 명령도 아닌 바로 ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1917(1917)</td>
      <td>1917 2019</td>
      <td>드라마</td>
      <td>10</td>
      <td>촬영감독의 영혼까지 갈아넣은 마스터피스</td>
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
      <th>995</th>
      <td>보헤미안 랩소디</td>
      <td>2018</td>
      <td>드라마</td>
      <td>10</td>
      <td>내일 또 보러간다 몇 번 더 보러갈지 나도 이제 모르겠다</td>
    </tr>
    <tr>
      <th>996</th>
      <td>보헤미안 랩소디</td>
      <td>2018</td>
      <td>드라마</td>
      <td>10</td>
      <td>퀸의 노래를 안들어본사람은 있어도 한번들어본 사람은 없다</td>
    </tr>
    <tr>
      <th>997</th>
      <td>보헤미안 랩소디</td>
      <td>2018</td>
      <td>드라마</td>
      <td>10</td>
      <td>최고 처음으로 혼자가서 두번째다시 본 영화</td>
    </tr>
    <tr>
      <th>998</th>
      <td>보헤미안 랩소디</td>
      <td>2018</td>
      <td>드라마</td>
      <td>10</td>
      <td>별100개가 안찍혀서 10개줌</td>
    </tr>
    <tr>
      <th>999</th>
      <td>보헤미안 랩소디</td>
      <td>2018</td>
      <td>드라마</td>
      <td>10</td>
      <td>영화보고 와서 퀸 공부중입니다 같은 세대였다면 얼마나 좋았을까요</td>
    </tr>
  </tbody>
</table>
<p>9615 rows × 5 columns</p>
</div>




```python
names = ['위대한 쇼맨', '알라딘', '어벤져스: 인피니티 워', '컨저링2', '샌 안드레아스', '레버넌트', '본 얼티메이텀', '인비저블맨(invisible)', '어바웃 타임', '조조 래빗']
data = []

for name in names:
  data.append(get_data(name))
  print('---{}complete-----'.format(str(name)))
  
data3 = pd.concat(data)
data3
```

    ---위대한 쇼맨complete-----
    ---알라딘complete-----
    ---어벤져스: 인피니티 워complete-----
    ---컨저링2complete-----
    ---샌 안드레아스complete-----
    ---레버넌트complete-----
    ---본 얼티메이텀complete-----
    ---인비저블맨(invisible)complete-----
    ---어바웃 타임complete-----
    ---조조 래빗complete-----
    




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
      <td>위대한 쇼맨</td>
      <td>2017</td>
      <td>드라마</td>
      <td>10</td>
      <td>말이 필요없음 이런걸 영화라고 함</td>
    </tr>
    <tr>
      <th>1</th>
      <td>위대한 쇼맨</td>
      <td>2017</td>
      <td>드라마</td>
      <td>10</td>
      <td>레미제라블이후로 휴잭맨의인생영화였다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>위대한 쇼맨</td>
      <td>2017</td>
      <td>드라마</td>
      <td>9</td>
      <td>휴잭맨의 매력이 한껏 돋보인 영화입니다 다른 출연자들도 너무 멋지구요 사람은 누구나...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>위대한 쇼맨</td>
      <td>2017</td>
      <td>드라마</td>
      <td>9</td>
      <td>지금 미국에선 평론가들 평 안좋고 관객평은 좋다던데 우째 딱 영화 처럼 된듯ㅋㅋㅋ ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>위대한 쇼맨</td>
      <td>2017</td>
      <td>드라마</td>
      <td>10</td>
      <td>정말 인생 최고의 영화 너무 좋아서 울컥하고 남들 없었음 ㅠ 기립박수 칠뻔 강추</td>
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
      <th>995</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>개인적으론 올해 최고의 영화 ^^</td>
    </tr>
    <tr>
      <th>996</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>꼭 모두에게 권하고싶은 영화에요 ㅜㅜ</td>
    </tr>
    <tr>
      <th>997</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>스포일러가 포함된 감상평입니다 감상평 보기</td>
    </tr>
    <tr>
      <th>998</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>7</td>
      <td>귀여운 영화 후반으로 갈수록 애늙으니 같은 주인공 요키 너무 귀여움 영화는 역시 2...</td>
    </tr>
    <tr>
      <th>999</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>7</td>
      <td>세상과 맞서싸우려는 토끼에게 경외를 표하노니</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>




```python
names = ['내부자들', '가장 보통의 연애', '그것만이 내 세상', '남한산성', '해적: 바다로 간 산적', '돈', '뺑반', '도어락', '봉오동 전투', '곤지암']
data = []

for name in names:
  data.append(get_data(name))
  print('---{}complete-----'.format(str(name)))
  
data1 = pd.concat(data)
data1

```

    ---내부자들complete-----
    ---가장 보통의 연애complete-----
    ---그것만이 내 세상complete-----
    ---남한산성complete-----
    ---해적: 바다로 간 산적complete-----
    ---돈complete-----
    ---뺑반complete-----
    ---도어락complete-----
    ---봉오동 전투complete-----
    ---곤지암complete-----
    




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
      <td>10</td>
      <td>사람 이병헌은 별로인데 연기자 이병헌은 최고다 인정</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>이병헌의 연기적 재능은 악마적 재능인듯 진짜 포스 ㅎㄷㄷ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>그분 의 사생활 때문에 폄하하기엔 영화가 너무 잘빠졌다 ㅋㅋ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>다들 이병헌이 싫다하지만 이병헌 연기는 뭐라 못함 걍 재밌음 더러운데 재밌음 사회적으로 시사하는 바도 많고 참 씁쓸함 대한민국 현실</td>
    </tr>
    <tr>
      <th>4</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>7</td>
      <td>그래도이병헌은이병헌이다</td>
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
      <th>995</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>5</td>
      <td>암걸릴뻔했다 1 영화 스토리상 하지말라는거 꼭 쳐하는 애들때문에 욕나왔음2 밤늦게갔는데도 급식들이 떼창을했음ㅡㅡ진짜 머갈통 후려팰뻔</td>
    </tr>
    <tr>
      <th>996</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>8</td>
      <td>스토리와 개연성이 부족한건 사실이다 카메라 앵글도 정신없다 그러나 무섭긴 무섭다 깜짝깜짝 놀라느라 쉴틈을 안준다 왠만한 공포영화 안 무서워하는데 이건 몸에 땀 주고 보았음 역대급 공포영화에 꼽힐만 한듯</td>
    </tr>
    <tr>
      <th>997</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>8</td>
      <td>무서웠어요 ㅠㅠ 귀신은 한국귀신이 제일 무서운듯</td>
    </tr>
    <tr>
      <th>998</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>10</td>
      <td>갑자기 외계인 나와서당황</td>
    </tr>
    <tr>
      <th>999</th>
      <td>곤지암</td>
      <td>2017</td>
      <td>공포</td>
      <td>6</td>
      <td>초반엔 좀지루하지만 후반부엔 무섭네요ㅋㅋㅋ</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>



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
#data1 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/naver_netizen_token_dawis.csv', encoding='euc-kr')

data = pd.concat([data1, data2, data3])
data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
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
      <td>10</td>
      <td>사람 이병헌은 별로인데 연기자 이병헌은 최고다 인정</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>이병헌의 연기적 재능은 악마적 재능인듯 진짜 포스 ㅎㄷㄷ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>그분 의 사생활 때문에 폄하하기엔 영화가 너무 잘빠졌다 ㅋㅋ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>다들 이병헌이 싫다하지만 이병헌 연기는 뭐라 못함 걍 재밌음 더러운데 재밌음 사회적으로 시사하는 바도 많고 참 씁쓸함 대한민국 현실</td>
    </tr>
    <tr>
      <th>4</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>7</td>
      <td>그래도이병헌은이병헌이다</td>
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
      <th>29610</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>개인적으론 올해 최고의 영화 ^^</td>
    </tr>
    <tr>
      <th>29611</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>꼭 모두에게 권하고싶은 영화에요 ㅜㅜ</td>
    </tr>
    <tr>
      <th>29612</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>스포일러가 포함된 감상평입니다 감상평 보기</td>
    </tr>
    <tr>
      <th>29613</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>7</td>
      <td>귀여운 영화 후반으로 갈수록 애늙으니 같은 주인공 요키 너무 귀여움 영화는 역시 2차세계대전입니다</td>
    </tr>
    <tr>
      <th>29614</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>7</td>
      <td>세상과 맞서싸우려는 토끼에게 경외를 표하노니</td>
    </tr>
  </tbody>
</table>
<p>29615 rows × 5 columns</p>
</div>




```python
data['review'] = data['review'].astype(str)
for i in range(len(data['review'])):
  data['review'][i] = data['review'][i].encode('utf-8')
```

## Install konlpy & Import Okt


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

a1 = list(data.loc[data['title']=='1917(1917)']['review'])
b1 = list(data.loc[data['title']=='서치 아웃']['review'])
c1 = list(data.loc[data['title']=='덕구']['review'])
d1 = list(data.loc[data['title']=='가버나움']['review'])
e1 = list(data.loc[data['title']=='어벤져스: 엔드게임']['review'])
f1 = list(data.loc[data['title']=='인셉션']['review'])
g1 = list(data.loc[data['title']=='작은 아씨들(Little Women)']['review'])
h1 = list(data.loc[data['title']=='그린 북']['review'])
i1 = list(data.loc[data['title']=='포드 V 페라리']['review'])
j1 = list(data.loc[data['title']=='보헤미안 랩소디']['review'])

a2 = list(data.loc[data['title']=='위대한 쇼맨']['review'])
b2 = list(data.loc[data['title']=='알라딘']['review'])
c2 = list(data.loc[data['title']=='어벤져스: 인피니티 워']['review'])
d2 = list(data.loc[data['title']=='컨저링2']['review'])
e2 = list(data.loc[data['title']=='샌 안드레아스']['review'])
f2 = list(data.loc[data['title']=='레버넌트']['review'])
g2 = list(data.loc[data['title']=='본 얼티메이텀']['review'])
h2 = list(data.loc[data['title']=='인비저블맨(invisible)']['review'])
i2 = list(data.loc[data['title']=='어바웃 타임']['review'])
j2 = list(data.loc[data['title']=='조조 래빗']['review'])
```


```python
movie_lists = [a,b,c,d,e,f,g,h,i,j,a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,a2,b2,c2,d2,e2,f2,g2,h2,i2,j2]
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
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다' ,'영화', '생각', '같다', '그냥', '내용', '편', '점', '영화중', '영화관', '대해', '것', '이다', '진짜', '정말', '하나', 
'해지', '여왜', '있다', '아니다', '내내', '자인', '리얼', '볼', '보다', '보고', '이영화', '그렇다', '꼭', '더', '아이', '로', '이렇다', '없다', '평', '감상', '번', '때', '조조', '보기', '말', '수', '내', '그', '바', '스포일러', '상영', '재', '관', '감', '안', '왜', '개', '거']
movie_lists = [a,b,c,d,e,f,g,h,i,j,a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,a2,b2,c2,d2,e2,f2,g2,h2,i2,j2]
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
import pandas as pd
final = pd.DataFrame(final)
final['total'] = final[final.columns].astype(str).apply(lambda x: ' '.join(x), axis = 1)
final = final[['total']]
final
```

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

for i in range(len(final)):
  for word in final["count"][i]:
          if word =="재미있다":
             word == "재미"
          if word =="재밌다":
             word == "재미"
          if word =="1917 2019":
             word == "2019"
          if word =="2 2016":
             word == "2016" 

final
```

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
  if data["title"][i]=='1917(1917)':
    data['count'][i] = final['count'][10]
  if data["title"][i]=='서치 아웃':
    data['count'][i] = final['count'][11]
  if data["title"][i]=='덕구':
    data['count'][i] = final['count'][12]
  if data["title"][i]== '가버나움':
    data['count'][i] = final['count'][13]
  if data["title"][i]=='어벤져스: 엔드게임':
    data['count'][i] = final['count'][14]
  if data["title"][i]=='인셉션':
    data['count'][i] = final['count'][15]
  if data["title"][i]== '작은 아씨들(Little Women)':
    data['count'][i] = final['count'][16]
  if data["title"][i]== '그린 북':
    data['count'][i] = final['count'][17]
  if data["title"][i]==  '포드 V 페라리':
    data['count'][i] = final['count'][18]
  if data["title"][i]=='보헤미안 랩소디':
    data['count'][i] = final['count'][19]
  if data["title"][i]=='위대한 쇼맨':
    data['count'][i] = final['count'][20]
  if data["title"][i]=='알라딘':
    data['count'][i] = final['count'][21]
  if data["title"][i]=='어벤져스: 인피니티 워':
    data['count'][i] = final['count'][22]
  if data["title"][i]=='컨저링2':
    data['count'][i] = final['count'][23]
  if data["title"][i]== '샌 안드레아스':
    data['count'][i] = final['count'][24]
  if data["title"][i]=='레버넌트':
    data['count'][i] = final['count'][25]
  if data["title"][i]=='본 얼티메이텀':
    data['count'][i] = final['count'][26]
  if data["title"][i]== '인비저블맨(invisible)':
    data['count'][i] = final['count'][27]
  if data["title"][i]== '어바웃 타임':
    data['count'][i] = final['count'][28]
  if data["title"][i]==  '조조 래빗':
    data['count'][i] = final['count'][29]
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    /usr/local/lib/python3.6/dist-packages/pandas/core/indexing.py:671: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_with_indexer(indexer, value)
    


```python
data['count'][0]
```




    ['이병헌',
     '연기',
     '배우',
     '최고',
     '조승우',
     '재밌다',
     '좋다',
     '역시',
     '현실',
     '연기력',
     '재미있다',
     '스토리',
     '모히또',
     '부자',
     '사람',
     '병헌',
     '몰디브',
     '대박',
     '평점',
     '백윤식']




```python
for i in range(len(data['review'])):
  data['review'][i] = data['review'][i].decode('utf-8')
```

## Tagging


```python
#data.drop(["tag"], axis=1, inplace=True)
for i in range(len(data)):
  if data.title[i] == "1917(1917)":
      data.year[i] = "2019"
for i in range(len(data)):
  if data.title[i] =="컨저링2":
      data.year[i] = "2016"
#data.loc[data.title=="1917(1917)", :].year = data.loc[data.title=="1917(1917)", :].year.replace("1917 2019", "2019", inplace=True)
#data.loc[data.title=="컨저링2", :].year = data.loc[data.title=="컨저링2", :].year.replace("2 2016", "2016", inplace=True)

for i in range(len(data)):
    data["count"][i] = [word for word in data["count"][i] if len(word)>1]

data['tag'] = data[['year', 'genre', 'count']].astype(str).apply(lambda x: ', '.join(x), axis = 1) #year+genre+top 20 words
#data.drop(["count"], axis =1, inplace = True)

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
      <th>count</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>사람 이병헌은 별로인데 연기자 이병헌은 최고다 인정</td>
      <td>[이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식]</td>
      <td>2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>이병헌의 연기적 재능은 악마적 재능인듯 진짜 포스 ㅎㄷㄷ</td>
      <td>[이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식]</td>
      <td>2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식</td>
    </tr>
    <tr>
      <th>2</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>그분 의 사생활 때문에 폄하하기엔 영화가 너무 잘빠졌다 ㅋㅋ</td>
      <td>[이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식]</td>
      <td>2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식</td>
    </tr>
    <tr>
      <th>3</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>10</td>
      <td>다들 이병헌이 싫다하지만 이병헌 연기는 뭐라 못함 걍 재밌음 더러운데 재밌음 사회적으로 시사하는 바도 많고 참 씁쓸함 대한민국 현실</td>
      <td>[이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식]</td>
      <td>2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식</td>
    </tr>
    <tr>
      <th>4</th>
      <td>내부자들</td>
      <td>2015</td>
      <td>범죄</td>
      <td>7</td>
      <td>그래도이병헌은이병헌이다</td>
      <td>[이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식]</td>
      <td>2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29610</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>개인적으론 올해 최고의 영화 ^^</td>
      <td>[좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다]</td>
      <td>2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다</td>
    </tr>
    <tr>
      <th>29611</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>꼭 모두에게 권하고싶은 영화에요 ㅜㅜ</td>
      <td>[좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다]</td>
      <td>2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다</td>
    </tr>
    <tr>
      <th>29612</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>10</td>
      <td>스포일러가 포함된 감상평입니다 감상평 보기</td>
      <td>[좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다]</td>
      <td>2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다</td>
    </tr>
    <tr>
      <th>29613</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>7</td>
      <td>귀여운 영화 후반으로 갈수록 애늙으니 같은 주인공 요키 너무 귀여움 영화는 역시 2차세계대전입니다</td>
      <td>[좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다]</td>
      <td>2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다</td>
    </tr>
    <tr>
      <th>29614</th>
      <td>조조 래빗</td>
      <td>2019</td>
      <td>코미디</td>
      <td>7</td>
      <td>세상과 맞서싸우려는 토끼에게 경외를 표하노니</td>
      <td>[좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다]</td>
      <td>2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다</td>
    </tr>
  </tbody>
</table>
<p>29615 rows × 7 columns</p>
</div>



## Making tagged data only


```python
data4 = data[["title","tag"]]
result_df = data4.drop_duplicates(subset=['title'], keep='first')
result_df.reset_index(inplace=True)
result_df.drop("index", axis=1, inplace=True)
pd.set_option('max_colwidth', None)
result_df
```

    /usr/local/lib/python3.6/dist-packages/pandas/core/frame.py:3997: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,
    




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
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>내부자들</td>
      <td>2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식</td>
    </tr>
    <tr>
      <th>1</th>
      <td>가장 보통의 연애</td>
      <td>2019, 멜로/로맨스, 연애, 공감, 보통, 공효진, 김래원, 좋다, 배우, 재밌다, 평점, 연기, 스토리, 아깝다, 가장, 현실, 사람, 안되다, 별로, 시간, 대사</td>
    </tr>
    <tr>
      <th>2</th>
      <td>그것만이 내 세상</td>
      <td>2017, 코미디, 이병헌, 연기, 감동, 좋다, 배우, 박정민, 역시, 최고, 재밌다, 피아노, 눈물, 연기력, 한지민, 윤여정, 뻔하다, 재미, 병헌, 스토리, 가족, 슬프다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>남한산성</td>
      <td>2017, 드라마, 역사, 지루하다, 배우, 연기, 좋다, 재미없다, 아깝다, 평점, 연기력, 최고, 재미, 재밌다, 사람, 이병헌, 사극, 감동, 노잼, 시간, 지루함</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해적: 바다로 간 산적</td>
      <td>2014, 모험, 재밌다, 김남길, 해적, 유해진, 좋다, 명량, 재미있다, 연기, 코믹, 손예진, 배우, 유쾌하다, 최고, 재미, 완전, 시원하다, 평점, 웃음, 가족</td>
    </tr>
    <tr>
      <th>5</th>
      <td>돈</td>
      <td>2018, 범죄, 재밌다, 연기, 류준열, 좋다, 배우, 주식, 시간, 스토리, 재미있다, 류준, 유지태, 평점, 알바, 재미, 사람, 아깝다, 작전, 부자, 결말</td>
    </tr>
    <tr>
      <th>6</th>
      <td>뺑반</td>
      <td>2018, 범죄, 연기, 배우, 재밌다, 류준열, 좋다, 조정석, 스토리, 아깝다, 공효진, 개연, 지루하다, 액션, 알바, 평점, 시간, 연기력, 직업, 최고, 뺑반</td>
    </tr>
    <tr>
      <th>7</th>
      <td>도어락</td>
      <td>2018, 스릴러, 무섭다, 답답하다, 현실, 공효진, 주인공, 혼자, 재밌다, 범인, 연기, 고구마, 사람, 공포, 좋다, 스릴러, 경찰, 아깝다, 여자, 뻔하다, 도어</td>
    </tr>
    <tr>
      <th>8</th>
      <td>봉오동 전투</td>
      <td>2019, 액션, 좋다, 배우, 연기, 우리, 독립군, 가슴, 역사, 감동, 독립, 독립운동가, 야하다, 재밌다, 지금, 만세, 유해진, 일본, 최고, 나라, 존경, 대한민국</td>
    </tr>
    <tr>
      <th>9</th>
      <td>곤지암</td>
      <td>2017, 공포, 무섭다, 공포영화, 아깝다, 공포, 소리, 사람, 귀신, 재미없다, 평점, 노잼, 알바, 처음, 곤지암, 급식, 무서움, 한국, 시끄럽다</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1917(1917)</td>
      <td>2019, 드라마, 전쟁, 테이크, 연출, 촬영, 스토리, 좋다, 전쟁영화, 기생충, 몰입, 최고, 느낌, 작품, 주인공, 연기, 장면, 긴장감, 기법, 시간, 포함</td>
    </tr>
    <tr>
      <th>11</th>
      <td>서치 아웃</td>
      <td>2019, 스릴러, 연기력, 재밌다, 최악, 이시언, 연기, 예능, 제발, 스릴, 한국, 장난, 자체, 그만하다, 배우, 넘치다, 다행, 영화배우, 어마어마하다, 공포, 용구성</td>
    </tr>
    <tr>
      <th>12</th>
      <td>덕구</td>
      <td>2017, 드라마, 이순재, 연기, 눈물, 감동, 배우, 좋다, 할아버지, 슬프다, 선생님, 최고, 덕구, 따뜻하다, 처음, 마음, 펑펑, 가슴, 사랑</td>
    </tr>
    <tr>
      <th>13</th>
      <td>가버나움</td>
      <td>2018, 드라마, 마음, 가슴, 좋다, 현실, 아프다, 눈물, 마지막, 많다, 부모, 사람, 연기, 어른, 우리, 감동, 인생, 모습, 난민, 세상, 슬프다</td>
    </tr>
    <tr>
      <th>14</th>
      <td>어벤져스: 엔드게임</td>
      <td>2019, 액션, 마블, 어벤져스, 재밌다, 아이언맨, 최고, 마지막, 사랑, 좋다, 사람, 엔드게임, 감동, 재미있다, 히어로, 눈물, 시리즈, 인생, 아쉽다, 슬프다, 장면</td>
    </tr>
    <tr>
      <th>15</th>
      <td>인셉션</td>
      <td>2010, 액션, 최고, 놀란, 인셉션, 인생, 이해, 개봉, 감독, 다시, 명작, 사람, 재밌다, 지금, 작품, 대단하다, 현실, 처음, 천재, 마지막</td>
    </tr>
    <tr>
      <th>16</th>
      <td>작은 아씨들(Little Women)</td>
      <td>2019, 드라마, 좋다, 감동, 연기, 배우, 사랑, 잔잔하다, 자매, 재밌다, 따뜻하다, 원작, 이야기, 포함, 아름답다, 스토리, 연출, 에이미, 지루하다, 인생</td>
    </tr>
    <tr>
      <th>17</th>
      <td>그린 북</td>
      <td>2018, 드라마, 좋다, 감동, 사람, 최고, 따뜻하다, 편견, 연기, 많다, 차별, 잔잔하다, 흑인, 마음, 재미, 여운, 작품, 배우, 재밌다, 인종차별, 셜리, 음악</td>
    </tr>
    <tr>
      <th>18</th>
      <td>포드 V 페라리</td>
      <td>2019, 액션, 재밌다, 최고, 연기, 베일, 좋다, 포드, 레이싱, 페라리, 시간, 사람, 크리스찬, 포함, 스토리, 자동차, 감동, 심장, 엔진, 데이먼, 소리</td>
    </tr>
    <tr>
      <th>19</th>
      <td>보헤미안 랩소디</td>
      <td>2018, 드라마, 감동, 노래, 프레디, 눈물, 마지막, 최고, 음악, 좋다, 머큐리, 인생, 장면, 라이브, 소름, 공연, 한번, 사람, 처음, 이드</td>
    </tr>
    <tr>
      <th>20</th>
      <td>위대한 쇼맨</td>
      <td>2017, 드라마, 좋다, 최고, 노래, 인생, 뮤지컬, 감동, 음악, 사람, 재밌다, 스토리, 처음, 소름, 휴잭맨, 미화, 눈물, 라라, 랜드, 즐겁다</td>
    </tr>
    <tr>
      <th>21</th>
      <td>알라딘</td>
      <td>2019, 모험, 알라딘, 좋다, 재밌다, 노래, 자스민, 디즈니, 최고, 윌스미스, 지니, 감동, 재미있다, 아름답다, 실사, 연기, 음악, 처음, 공주, 배우, 이쁘다</td>
    </tr>
    <tr>
      <th>22</th>
      <td>어벤져스: 인피니티 워</td>
      <td>2018, 액션, 마블, 토르, 노스, 번역가, 어벤져스, 번역, 충격, 최고, 재밌다, 사람, 결말, 오역, 히어로, 역대, 마지막, 박지훈, 좋다, 제발, 때문, 등장</td>
    </tr>
    <tr>
      <th>23</th>
      <td>컨저링2</td>
      <td>2016, 공포, 무섭다, 공포영화, 컨저링, 재밌다, 공포, 수녀, 좋다, 귀신, 역시, 사람, 스토리, 스완, 장면, 처음, 식이, 무서움, 재미있다, 평론가, 혼자</td>
    </tr>
    <tr>
      <th>24</th>
      <td>샌 안드레아스</td>
      <td>2015, 액션, 재난영화, 재밌다, 스토리, 뻔하다, 가족, 스케일, 재난, 지진, 좋다, 주인공, 평점, 재미있다, 박평, 최고, 사람, 평론가, 정도, 장면, 재미, 해운대</td>
    </tr>
    <tr>
      <th>25</th>
      <td>레버넌트</td>
      <td>2015, 모험, 디카프리오, 연기, 지루하다, 좋다, 평점, 생존, 스토리, 복수, 시간, 레오, 고생, 재미없다, 아깝다, 노잼, 배우, 잔인하다, 다큐, 재미</td>
    </tr>
    <tr>
      <th>26</th>
      <td>본 얼티메이텀</td>
      <td>2007, 액션, 시리즈, 최고, 액션, 첩보, 데이먼, 재밌다, 엔딩, 액션영화, 다시, 마지막, 스토리, 소름, 제이슨, 역시, 평점, 좋다, 첩보물, 정도</td>
    </tr>
    <tr>
      <th>27</th>
      <td>인비저블맨(invisible)</td>
      <td>2020, 공포, 무섭다, 공포, 재밌다, 연기, 좋다, 스토리, 포함, 여주, 공포영화, 주인공, 스릴러, 긴장감, 평점, 반전, 사람, 결말, 몰입, 처음, 투명인간, 연출</td>
    </tr>
    <tr>
      <th>28</th>
      <td>어바웃 타임</td>
      <td>2013, 멜로/로맨스, 감동, 평점, 좋다, 인생, 시간, 사랑, 최고, 따뜻하다, 다시, 교훈, 소중하다, 아버지, 지루하다, 마지막, 가족, 사람, 시간여행, 가슴, 재밌다, 행복하다</td>
    </tr>
    <tr>
      <th>29</th>
      <td>조조 래빗</td>
      <td>2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다</td>
    </tr>
  </tbody>
</table>
</div>




```python
result_df['tag'] = result_df['tag'].astype(str)
for i in range(len(result_df)):
  result_df['tag'][i] = result_df['tag'][i].encode('utf-8')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    /usr/local/lib/python3.6/dist-packages/IPython/core/interactiveshell.py:2882: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      exec(code_obj, self.user_global_ns, self.user_ns)
    


```python
for i in range(len(result_df)):
  clean_words = []
  for word in okt.pos(result_df['tag'][i], stem=True): #어간 추출
    if word[1] in ['Number', 'Noun', 'Adjective']:
      clean_words.append(word[0])
      result_df['tag'][i] = clean_words

result_df
```

    /usr/local/lib/python3.6/dist-packages/IPython/core/interactiveshell.py:2882: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      exec(code_obj, self.user_global_ns, self.user_ns)
    




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
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>내부자들</td>
      <td>[2015, 범죄, 이병헌, 연기, 배우, 최고, 조승우, 재밌다, 좋다, 역시, 현실, 연기력, 재미있다, 스토리, 모히또, 부자, 사람, 병헌, 몰디브, 대박, 평점, 백윤식]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>가장 보통의 연애</td>
      <td>[2019, 멜로, 로맨스, 연애, 공감, 보통, 공효진, 김래원, 좋다, 배우, 재밌다, 평점, 연기, 스토리, 아깝다, 가장, 현실, 사람, 안되다, 별로, 시간, 대사]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>그것만이 내 세상</td>
      <td>[2017, 코미디, 이병헌, 연기, 감동, 좋다, 배우, 박정민, 역시, 최고, 재밌다, 피아노, 눈물, 연기력, 한지민, 윤여정, 뻔하다, 재미, 병헌, 스토리, 가족, 슬프다]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>남한산성</td>
      <td>[2017, 드라마, 역사, 지루하다, 배우, 연기, 좋다, 재미없다, 아깝다, 평점, 연기력, 최고, 재미, 재밌다, 사람, 이병헌, 사극, 감동, 노잼, 시간, 지루함]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해적: 바다로 간 산적</td>
      <td>[2014, 모험, 재밌다, 김남길, 해적, 유해진, 좋다, 명량, 재미있다, 연기, 코믹, 손예진, 배우, 유쾌하다, 최고, 재미, 완전, 시원하다, 평점, 웃음, 가족]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>돈</td>
      <td>[2018, 범죄, 재밌다, 연기, 류준열, 좋다, 배우, 주식, 시간, 스토리, 재미있다, 류준, 유지태, 평점, 알바, 재미, 사람, 아깝다, 작전, 부자, 결말]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>뺑반</td>
      <td>[2018, 범죄, 연기, 배우, 재밌다, 류준열, 좋다, 조정석, 스토리, 아깝다, 공효진, 개연, 지루하다, 액션, 알바, 평점, 시간, 연기력, 직업, 최고, 뺑반]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>도어락</td>
      <td>[2018, 스릴러, 무섭다, 답답하다, 현실, 공효진, 주인공, 혼자, 재밌다, 범인, 연기, 고구마, 사람, 공포, 좋다, 스릴러, 경찰, 아깝다, 여자, 뻔하다, 도어]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>봉오동 전투</td>
      <td>[2019, 액션, 좋다, 배우, 연기, 우리, 독립군, 가슴, 역사, 감동, 독립, 독립운동가, 야하다, 재밌다, 지금, 만세, 유해진, 일본, 최고, 나라, 존경, 대한민국]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>곤지암</td>
      <td>[2017, 공포, 무섭다, 공포영화, 아깝다, 공포, 소리, 사람, 귀신, 재미없다, 평점, 노잼, 알바, 처음, 곤지암, 급식, 무서움, 한국, 시끄럽다]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1917(1917)</td>
      <td>[2019, 드라마, 전쟁, 테이크, 연출, 촬영, 스토리, 좋다, 전쟁영화, 기생충, 몰입, 최고, 느낌, 작품, 주인공, 연기, 장면, 긴장감, 기법, 시간, 포함]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>서치 아웃</td>
      <td>[2019, 스릴러, 연기력, 재밌다, 최악, 이시언, 연기, 예능, 제발, 스릴, 한국, 장난, 자체, 그만하다, 배우, 넘치다, 다행, 영화배우, 어마어마하다, 공포, 용구성]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>덕구</td>
      <td>[2017, 드라마, 이순재, 연기, 눈물, 감동, 배우, 좋다, 할아버지, 슬프다, 선생님, 최고, 덕구, 따뜻하다, 처음, 마음, 펑펑, 가슴, 사랑]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>가버나움</td>
      <td>[2018, 드라마, 마음, 가슴, 좋다, 현실, 아프다, 눈물, 마지막, 많다, 부모, 사람, 연기, 어른, 우리, 감동, 인생, 모습, 난민, 세상, 슬프다]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>어벤져스: 엔드게임</td>
      <td>[2019, 액션, 마블, 어벤져스, 재밌다, 아이언맨, 최고, 마지막, 사랑, 좋다, 사람, 엔드게임, 감동, 재미있다, 히어로, 눈물, 시리즈, 인생, 아쉽다, 슬프다, 장면]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>인셉션</td>
      <td>[2010, 액션, 최고, 놀란, 인셉션, 인생, 이해, 개봉, 감독, 다시, 명작, 사람, 재밌다, 지금, 작품, 대단하다, 현실, 처음, 천재, 마지막]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>작은 아씨들(Little Women)</td>
      <td>[2019, 드라마, 좋다, 감동, 연기, 배우, 사랑, 잔잔하다, 자매, 재밌다, 따뜻하다, 원작, 이야기, 포함, 아름답다, 스토리, 연출, 에이미, 지루하다, 인생]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>그린 북</td>
      <td>[2018, 드라마, 좋다, 감동, 사람, 최고, 따뜻하다, 편견, 연기, 많다, 차별, 잔잔하다, 흑인, 마음, 재미, 여운, 작품, 배우, 재밌다, 인종차별, 셜리, 음악]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>포드 V 페라리</td>
      <td>[2019, 액션, 재밌다, 최고, 연기, 베일, 좋다, 포드, 레이싱, 페라리, 시간, 사람, 크리스찬, 포함, 스토리, 자동차, 감동, 심장, 엔진, 데이먼, 소리]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>보헤미안 랩소디</td>
      <td>[2018, 드라마, 감동, 노래, 프레디, 눈물, 마지막, 최고, 음악, 좋다, 머큐리, 인생, 장면, 라이브, 소름, 공연, 한번, 사람, 처음, 이드]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>위대한 쇼맨</td>
      <td>[2017, 드라마, 좋다, 최고, 노래, 인생, 뮤지컬, 감동, 음악, 사람, 재밌다, 스토리, 처음, 소름, 휴잭맨, 미화, 눈물, 라라, 랜드, 즐겁다]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>알라딘</td>
      <td>[2019, 모험, 알라딘, 좋다, 재밌다, 노래, 자스민, 디즈니, 최고, 윌스미스, 지니, 감동, 재미있다, 아름답다, 실사, 연기, 음악, 처음, 공주, 배우, 이쁘다]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>어벤져스: 인피니티 워</td>
      <td>[2018, 액션, 마블, 토르, 노스, 번역가, 어벤져스, 번역, 충격, 최고, 재밌다, 사람, 결말, 오역, 히어로, 역대, 마지막, 박지훈, 좋다, 제발, 때문, 등장]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>컨저링2</td>
      <td>[2016, 공포, 무섭다, 공포영화, 컨저링, 재밌다, 공포, 수녀, 좋다, 귀신, 역시, 사람, 스토리, 스완, 장면, 처음, 식이, 무서움, 재미있다, 평론가, 혼자]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>샌 안드레아스</td>
      <td>[2015, 액션, 재난영화, 재밌다, 스토리, 뻔하다, 가족, 스케일, 재난, 지진, 좋다, 주인공, 평점, 재미있다, 박평, 최고, 사람, 평론가, 정도, 장면, 재미, 해운대]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>레버넌트</td>
      <td>[2015, 모험, 디카프리오, 연기, 지루하다, 좋다, 평점, 생존, 스토리, 복수, 시간, 레오, 고생, 재미없다, 아깝다, 노잼, 배우, 잔인하다, 다큐, 재미]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>본 얼티메이텀</td>
      <td>[2007, 액션, 시리즈, 최고, 액션, 첩보, 데이먼, 재밌다, 엔딩, 액션영화, 다시, 마지막, 스토리, 소름, 제이슨, 역시, 평점, 좋다, 첩보물, 정도]</td>
    </tr>
    <tr>
      <th>27</th>
      <td>인비저블맨(invisible)</td>
      <td>[2020, 공포, 무섭다, 공포, 재밌다, 연기, 좋다, 스토리, 포함, 여주, 공포영화, 주인공, 스릴러, 긴장감, 평점, 반전, 사람, 결말, 몰입, 처음, 투명인간, 연출]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>어바웃 타임</td>
      <td>[2013, 멜로, 로맨스, 감동, 평점, 좋다, 인생, 시간, 사랑, 최고, 따뜻하다, 다시, 교훈, 소중하다, 아버지, 지루하다, 마지막, 가족, 사람, 시간여행, 가슴, 재밌다, 행복하다]</td>
    </tr>
    <tr>
      <th>29</th>
      <td>조조 래빗</td>
      <td>[2019, 코미디, 좋다, 전쟁, 연기, 최고, 포함, 슬프다, 연출, 아름답다, 재밌다, 감동, 감독, 인생, 나치, 귀엽다, 스토리, 무겁다, 시선, 웃음, 따뜻하다, 순수하다]</td>
    </tr>
  </tbody>
</table>
</div>



## Import CSV


```python
data.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/data_withtag_dawis.csv', index=False)
result_df.to_csv('/content/gdrive/My Drive/Colab Notebooks/data/tag_dawis.csv', index=False)
```


```python

```
