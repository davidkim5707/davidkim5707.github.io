---
layout: post
title:  "Crawling_ Cine 21 movie review."
description: Crawling.
date:   2020-03-16
use_math: true
---
### Cine 21 movie review crawling.
Crawling

## Cine 21 movie review crawling 


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

### crawling expert review


```python
def get_data(name, genre):
  url_pre= 'http://www.cine21.com/search/?q='+str(name)
  res_pre = requests.get(url_pre)
  soup_pre = bs(res_pre.content, 'html.parser')
  temp = soup_pre.find('ul', {'class': 'mov_list'})
  temp1 = temp.findAll('a')
  code = re.sub(r"[^0-9]+", " ",temp1[0].attrs['href']).strip()

  title_list = []
  genre_list = []
  name_list = []
  review_text_list = []
  score_list = []
  resulted_df = []
  
  url = 'http://www.cine21.com/movie/info/?movie_id=' +str(code)
  res = requests.get(url)
  soup = bs(res.content, 'html.parser')
  score_result = soup.find('div', {'class': 'expert_rating_area'})

  if score_result != None:
    lis = score_result.findAll('li')
  
  
    for li in lis:
        title_list.append(name)
        genre_list.append(genre)
        name_list.append(li.select('span.name')[0].text)
        review_text_list.append(li.select('span.comment')[0].text)
        score_list.append(li.select('span.num')[0].text)


  appended_df = pd.DataFrame({'title':title_list, 'genre':genre_list, 'name':name_list, 'rating':score_list, 'review':review_text_list})

  resulted_df.append(appended_df)
  final_df = pd.concat(resulted_df)
  time.sleep(2)
  return final_df
```


```python
names = ['기생충', '봉오동 전투', '가장 보통의 연애', '극한직업', '곤지암', '도어락', '해적: 바다로 간 산적', '남산의 부장들', '괴물', '밀양']
genres = ['드라마', '액션', '멜로', '코미디', '공포', '스릴러', '모험', '드라마', 'SF', '드라마']
final = []

for name, genre in zip(names, genres):
  final.append(get_data(name, genre))
  print('---{}complete-----'.format(str(name)))

final_expert = pd.concat(final)
final_expert
```

    ---기생충complete-----
    ---봉오동 전투complete-----
    ---가장 보통의 연애complete-----
    ---극한직업complete-----
    ---곤지암complete-----
    ---도어락complete-----
    ---해적: 바다로 간 산적complete-----
    ---남산의 부장들complete-----
    ---괴물complete-----
    ---밀양complete-----
    




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
      <th>genre</th>
      <th>name</th>
      <th>rating</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>김소미</td>
      <td>9</td>
      <td>봉준호가 ‘그 검은 상자’를 열어버렸다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>김현수</td>
      <td>10</td>
      <td>2019 반지하 오디세이</td>
    </tr>
    <tr>
      <th>2</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>김혜리</td>
      <td>9</td>
      <td>정밀한 나머지 비통한 계급의식의 조감도</td>
    </tr>
    <tr>
      <th>3</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>박평식</td>
      <td>8</td>
      <td>‘유쾌한 전율’이 스멀스멀</td>
    </tr>
    <tr>
      <th>4</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>송형국</td>
      <td>9</td>
      <td>만인이 투쟁하는 ‘시대의 서스펜스’</td>
    </tr>
    <tr>
      <th>5</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>이용철</td>
      <td>10</td>
      <td>부르주아의 사려 깊은 매력, 취하거나 찌르거나</td>
    </tr>
    <tr>
      <th>6</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>이주현</td>
      <td>9</td>
      <td>뼈를 때리는 블랙 유머, 오랫동안 얼얼하다</td>
    </tr>
    <tr>
      <th>7</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>장영엽</td>
      <td>9</td>
      <td>시대의 환경을 이식한, 끝내주는 장르영화</td>
    </tr>
    <tr>
      <th>8</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>김성훈</td>
      <td>9</td>
      <td>‘그’ 집과 ‘그’ 집 사이에 놓인 계단 숫자만큼의 자존감 차이에 대해</td>
    </tr>
    <tr>
      <th>9</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>이화정</td>
      <td>9</td>
      <td>더 넓게, 더 깊게. 확장의 시력으로 현 사회의 계급의식을 탐색하는 봉테일적 시각</td>
    </tr>
    <tr>
      <th>10</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>임수연</td>
      <td>9</td>
      <td>예술과 상업, 장르의 구분이 무의미한 드문 영화적 체험</td>
    </tr>
    <tr>
      <th>0</th>
      <td>봉오동 전투</td>
      <td>액션</td>
      <td>박평식</td>
      <td>4</td>
      <td>봉오동 포화 속으로</td>
    </tr>
    <tr>
      <th>1</th>
      <td>봉오동 전투</td>
      <td>액션</td>
      <td>이용철</td>
      <td>4</td>
      <td>&lt;명량&gt;을 바랐을 텐데 &lt;사냥&gt;이 나왔다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>봉오동 전투</td>
      <td>액션</td>
      <td>이화정</td>
      <td>6</td>
      <td>이분법을 강조하니, 수위 오버</td>
    </tr>
    <tr>
      <th>3</th>
      <td>봉오동 전투</td>
      <td>액션</td>
      <td>이주현</td>
      <td>6</td>
      <td>역사를 기억하는 일이 증오를 확인하는 일은 아니어야 할 텐데</td>
    </tr>
    <tr>
      <th>4</th>
      <td>봉오동 전투</td>
      <td>액션</td>
      <td>김현수</td>
      <td>6</td>
      <td>최종병기 류준열</td>
    </tr>
    <tr>
      <th>5</th>
      <td>봉오동 전투</td>
      <td>액션</td>
      <td>허남웅</td>
      <td>5</td>
      <td>시대의 ‘감정’으로 진을 치고, ‘시대정신’을 포위하다</td>
    </tr>
    <tr>
      <th>0</th>
      <td>가장 보통의 연애</td>
      <td>멜로</td>
      <td>임수연</td>
      <td>7</td>
      <td>연애 감정은 아슬아슬 플러팅과 알코올을 타고</td>
    </tr>
    <tr>
      <th>0</th>
      <td>극한직업</td>
      <td>코미디</td>
      <td>김성훈</td>
      <td>7</td>
      <td>설정은 비현실적이되 설득은 리얼리티를 추구하는 웃음</td>
    </tr>
    <tr>
      <th>1</th>
      <td>극한직업</td>
      <td>코미디</td>
      <td>송형국</td>
      <td>7</td>
      <td>웃음을 향한 장인정신, 반갑다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>극한직업</td>
      <td>코미디</td>
      <td>이용철</td>
      <td>5</td>
      <td>반은 웃었다만 그다음은?</td>
    </tr>
    <tr>
      <th>3</th>
      <td>극한직업</td>
      <td>코미디</td>
      <td>임수연</td>
      <td>7</td>
      <td>설 연휴, 친구·애인·가족·친지 누구와 봐도 성공할 코미디+액션</td>
    </tr>
    <tr>
      <th>4</th>
      <td>극한직업</td>
      <td>코미디</td>
      <td>허남웅</td>
      <td>6</td>
      <td>‘닭’치고 웃음</td>
    </tr>
    <tr>
      <th>0</th>
      <td>곤지암</td>
      <td>공포</td>
      <td>김현수</td>
      <td>7</td>
      <td>한 많은 사연 없어도 충분히 무섭다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>곤지암</td>
      <td>공포</td>
      <td>허남웅</td>
      <td>6</td>
      <td>가지 말라면 가지 마 쫌!</td>
    </tr>
    <tr>
      <th>0</th>
      <td>도어락</td>
      <td>스릴러</td>
      <td>임수연</td>
      <td>6</td>
      <td>현실 공포를 포착한 중반까진 눈이 번쩍. 후반의 착취적인 연출이 아쉬워</td>
    </tr>
    <tr>
      <th>1</th>
      <td>도어락</td>
      <td>스릴러</td>
      <td>이용철</td>
      <td>7</td>
      <td>호러 잡는 스릴러가 이런 건가</td>
    </tr>
    <tr>
      <th>2</th>
      <td>도어락</td>
      <td>스릴러</td>
      <td>장영엽</td>
      <td>6</td>
      <td>피해자’의 의미를 다각도로 조명하는 스릴러. 이건 도시 괴담이 아니다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>도어락</td>
      <td>스릴러</td>
      <td>이주현</td>
      <td>6</td>
      <td>공감에서 공포로 향하는 과정이 더 사려 깊었다면</td>
    </tr>
    <tr>
      <th>4</th>
      <td>도어락</td>
      <td>스릴러</td>
      <td>허남웅</td>
      <td>6</td>
      <td>현실반영으로 ‘락’했다가 장르 재미 때문에 손쉽게 열리는 '도어'</td>
    </tr>
    <tr>
      <th>5</th>
      <td>도어락</td>
      <td>스릴러</td>
      <td>김소미</td>
      <td>6</td>
      <td>비정규직 자취 여성의 공포는 현실적, 피해 재현은 판타지</td>
    </tr>
    <tr>
      <th>0</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>박평식</td>
      <td>4</td>
      <td>산으로 간다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>이용철</td>
      <td>5</td>
      <td>&lt;개콘&gt;을 즐겼던 사람이라면</td>
    </tr>
    <tr>
      <th>2</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>이화정</td>
      <td>6</td>
      <td>국새 찾아 팀플레이. &lt;런닝맨&gt; 사극판</td>
    </tr>
    <tr>
      <th>3</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>장영엽</td>
      <td>6</td>
      <td>‘4파전’에서 귀엽고 유쾌한 지점은 선점했다</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>김성훈</td>
      <td>6</td>
      <td>쉴 새 없이 터진다, 유해진</td>
    </tr>
    <tr>
      <th>5</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>송경원</td>
      <td>6</td>
      <td>산적이든 해적이든, 웃기면 장땡</td>
    </tr>
    <tr>
      <th>6</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>김혜리</td>
      <td>6</td>
      <td>대세에 지장 없으면 내처 달리는 서해 파크 후룸라이드</td>
    </tr>
    <tr>
      <th>7</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>황진미</td>
      <td>7</td>
      <td>백성과 국가의 길항적 관계까지 품은 유쾌한 액션 코미디</td>
    </tr>
    <tr>
      <th>0</th>
      <td>남산의 부장들</td>
      <td>드라마</td>
      <td>박평식</td>
      <td>6</td>
      <td>조화롭고 팽팽하나 헛웃음</td>
    </tr>
    <tr>
      <th>1</th>
      <td>남산의 부장들</td>
      <td>드라마</td>
      <td>이용철</td>
      <td>8</td>
      <td>나는 유령과 함께 걸었다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>남산의 부장들</td>
      <td>드라마</td>
      <td>이주현</td>
      <td>7</td>
      <td>실화의 힘과 영화의 힘이 흥미롭게 교차한다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>남산의 부장들</td>
      <td>드라마</td>
      <td>김현수</td>
      <td>7</td>
      <td>쿠데타와 죽음 사이를 장르로 잇다</td>
    </tr>
    <tr>
      <th>4</th>
      <td>남산의 부장들</td>
      <td>드라마</td>
      <td>허남웅</td>
      <td>6</td>
      <td>장르 스타일에 휘발하는 역사</td>
    </tr>
    <tr>
      <th>5</th>
      <td>남산의 부장들</td>
      <td>드라마</td>
      <td>임수연</td>
      <td>8</td>
      <td>그 남자들의 사사로운 감정을 추출해, 장르영화의 재료로 삼다</td>
    </tr>
    <tr>
      <th>0</th>
      <td>괴물</td>
      <td>SF</td>
      <td>김봉석</td>
      <td>7</td>
      <td>괴물이 나오는, 캐릭터 코미디</td>
    </tr>
    <tr>
      <th>1</th>
      <td>괴물</td>
      <td>SF</td>
      <td>김은형</td>
      <td>8</td>
      <td>판타지와 동시대를 유연하게 엮어낸다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>괴물</td>
      <td>SF</td>
      <td>박평식</td>
      <td>8</td>
      <td>풍성하고 날카롭고 영리하다. 괴력!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>괴물</td>
      <td>SF</td>
      <td>이동진</td>
      <td>9</td>
      <td>기념비적인 충무로 오락영화</td>
    </tr>
    <tr>
      <th>4</th>
      <td>괴물</td>
      <td>SF</td>
      <td>황진미</td>
      <td>8</td>
      <td>잡소리는 발로 꺼라. 거대함을 이기는 치열함!</td>
    </tr>
    <tr>
      <th>5</th>
      <td>괴물</td>
      <td>SF</td>
      <td>유지나</td>
      <td>8</td>
      <td>최초로 기록될 합법적인 반미 오락영화의 탄생!</td>
    </tr>
    <tr>
      <th>0</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>김봉석</td>
      <td>8</td>
      <td>인간은 어떻게 구원받을 수 있을까?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>김지미</td>
      <td>9</td>
      <td>응달까지 파고드는 햇살 같은, 미약하지만 끈질긴 구원의 가능성</td>
    </tr>
    <tr>
      <th>2</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>김혜리</td>
      <td>8</td>
      <td>죽고 싶은 명백한 이유, 살아야 하는 은밀한 이유</td>
    </tr>
    <tr>
      <th>3</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>박평식</td>
      <td>8</td>
      <td>“내 울부짖은들, 뉘라 천사의 열에서 들으리오” 밀양 엘레지!</td>
    </tr>
    <tr>
      <th>4</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>이동진</td>
      <td>10</td>
      <td>영화라는 매체가 도달할 수 있는 깊이</td>
    </tr>
    <tr>
      <th>5</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>유지나</td>
      <td>8</td>
      <td>외롭고 상처받은 영혼에게 보내는 선물</td>
    </tr>
    <tr>
      <th>6</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>황진미</td>
      <td>7</td>
      <td>멜로영화-&gt;유괴영화-&gt;기독교영화-&gt;메디컬영화. 전도연 연기 작렬!</td>
    </tr>
    <tr>
      <th>7</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>남다은</td>
      <td>8</td>
      <td>판타지 없이도, 구원의 가능성 없이도, 그래도 살아야 한다는 것</td>
    </tr>
  </tbody>
</table>
</div>



### crawling netizen review


```python
def get_data_netizen(name, genre):
  url_pre= 'http://www.cine21.com/search/?q='+str(name)
  res_pre = requests.get(url_pre)
  soup_pre = bs(res_pre.content, 'html.parser')
  temp = soup_pre.find('ul', {'class': 'mov_list'})
  temp1 = temp.findAll('a')
  code = re.sub(r"[^0-9]+", " ",temp1[0].attrs['href']).strip()

  title_list = []
  genre_list = []
  nickname_list = []
  review_text_list = []
  created_at_list = []
  score_list = []
  resulted_df = []
  i = 1

  url_be = 'http://www.cine21.com/movie/nzreview/list'
  data = {
      'table': 'movie',
      'movie_id': str(code), 
      'selector': '#netizen_review_area',
      'page': '1'
   }
  res_be = requests.post(url=url_be, data=data)
  soup_be = bs(res_be.content, 'html.parser')
  if soup_be != None:
    for i in range(len(soup_be.select('div.page')[-1].text)+1):
      data = {
          'table': 'movie',
          'movie_id': str(code), 
          'selector': '#netizen_review_area',
          'page': i
        }
      res = requests.post(url=url_be, data=data)
      soup = bs(res.content, 'html.parser')
    
      lis = soup.findAll('li')
  
      for li in lis:
          title_list.append(name)
          genre_list.append(genre)
          nickname_list.append(li.select('div.id')[0].text)
          created_at_list.append(datetime.strptime(li.select('div.date')[0].text, "%Y-%m-%d %H:%M:%S"))
          score_list.append(li.select('span.num')[0].text)
          if li.select('div.comment')[0].text != None:
            review_text_list.append(re.sub(r"[^ㄱ-ㅣ가-힣]+", " ",li.select('div.comment')[0].text.strip()))
          else:
            review_text_list.append(re.sub(r"[^ㄱ-ㅣ가-힣]+", " ",li.select('div.comment ellipsis_3')[0].text.strip()))
        
      appended_df = pd.DataFrame({'title':title_list, 'genre':genre_list, 'name':nickname_list, 'score':score_list, 'review':review_text_list})
      i = i+1


  resulted_df.append(appended_df)
  final_df = pd.concat(resulted_df)
  time.sleep(2)
  return final_df
```


```python
names = ['기생충', '봉오동 전투', '가장 보통의 연애', '극한직업', '곤지암', '도어락', '해적: 바다로 간 산적', '남산의 부장들', '괴물', '밀양']
genres = ['드라마', '액션', '멜로', '코미디', '공포', '스릴러', '모험', '드라마', 'SF', '드라마']
final = []

for name, genre in zip(names, genres):
  final.append(get_data_netizen(name, genre))
  print('---{}complete-----'.format(str(name)))
  
final_netizen = pd.concat(final)
final_netizen

```

    ---기생충complete-----
    ---봉오동 전투complete-----
    ---가장 보통의 연애complete-----
    ---극한직업complete-----
    ---곤지암complete-----
    ---도어락complete-----
    ---해적: 바다로 간 산적complete-----
    ---남산의 부장들complete-----
    ---괴물complete-----
    ---밀양complete-----
    




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
      <th>genre</th>
      <th>name</th>
      <th>score</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>nico2985</td>
      <td>10</td>
      <td>이 글은 영화 기생충을 보고나서 제 개인적으로 어떤 평론가의 말도 읽지 않고 생각...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>xswer</td>
      <td>9</td>
      <td>기생하는 배우가 없다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>johnconnor</td>
      <td>7</td>
      <td>블랙 코미디 로써는 훌륭 하지만 사회구성의 필연적인 요소인 계급간의 격차를 애써 부...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>clamcore</td>
      <td>7</td>
      <td>저는 정말 잼있게 보았습니다 우리 사회의 안좋은 점을 영화를 토대로 잘 풀어 나간거...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>기생충</td>
      <td>드라마</td>
      <td>gert</td>
      <td>10</td>
      <td>저도 재미있게 보고 왔습니다 오랜만에 영화인데 다행이었습니다 산수갑산 생각나던 수석...</td>
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
      <th>50</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>2thesea</td>
      <td>10</td>
      <td>누가 누구를 용서할 것인가 누가 누구를 용서할 것인가 영화가 끝날 무렵에 원작이 궁...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>memmentomori</td>
      <td>9</td>
      <td>산다는 건 지독하게 아픈 것 그래도 살아야지 뭐</td>
    </tr>
    <tr>
      <th>52</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>maifs</td>
      <td>8</td>
      <td>그래도 삶은 계속된다</td>
    </tr>
    <tr>
      <th>53</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>audmomma1169</td>
      <td>6</td>
      <td>밀양 영화를 보고나면 그 여운이 사라지기전에 그 느낌을 다시 거슬러 담아놓고 싶을때...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>밀양</td>
      <td>드라마</td>
      <td>koojjh</td>
      <td>10</td>
      <td>상처 아픔 용서 사랑 다 볼수있는 전도연과 송강호가 빛낸</td>
    </tr>
  </tbody>
</table>
<p>215 rows × 5 columns</p>
</div>



### incorporating expert reviews and netizen reviews


```python
final_expert.reset_index(inplace=True)
final_netizen.reset_index(inplace=True)
```


```python
final_expert.drop('index', axis=1, inplace=True)
final_netizen.drop('index', axis=1, inplace=True)
```


```python
from google.colab import files

final_expert.to_csv('final_expert.csv',encoding='euc-kr', index=False)
final_netizen.to_csv('final_netizen.csv',encoding='euc-kr', index=False)

files.download('final_expert.csv')
files.download('final_netizen.csv')
```


```python
a = pd.read_csv('C:/Users/Dawis Kim/Dropbox/자연어처리/크롤링, 감성분석/final_expert.csv', engine='python')
b = pd.read_csv('C:/Users/Dawis Kim/Dropbox/자연어처리/크롤링, 감성분석/final_netizen.csv', engine='python')
```


```python
a["dum"] = 0
b["dum"] = 1
```


```python
c = pd.concat([a,b], ignore_index=True)
c = c.sort_values(by = ['title', 'dum'])
c.drop('level_0', axis= 1, inplace=True)
c.to_csv('C:/Users/Dawis Kim/Dropbox/자연어처리/크롤링, 감성분석/final_movie.csv',encoding='euc-kr', index=False)
c
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
      <th>genre</th>
      <th>name</th>
      <th>rating</th>
      <th>review</th>
      <th>dum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>가장 보통의 연애</td>
      <td>멜로</td>
      <td>임수연</td>
      <td>7</td>
      <td>연애 감정은 아슬아슬 플러팅과 알코올을 타고</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>가장 보통의 연애</td>
      <td>멜로</td>
      <td>hwachul92</td>
      <td>6</td>
      <td>전 여친의 결별로 인하여 상처 받은 남자와 사랑을 아예 믿지 않는 여자와의 만남에서...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>81</th>
      <td>가장 보통의 연애</td>
      <td>멜로</td>
      <td>nirvana1974</td>
      <td>6</td>
      <td>언제나 정답이 없는 연애의 명쾌한 해석</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>가장 보통의 연애</td>
      <td>멜로</td>
      <td>johnconnor</td>
      <td>7</td>
      <td>평범한 이야기를 유쾌하게 풀어 가는군</td>
      <td>1</td>
    </tr>
    <tr>
      <th>83</th>
      <td>가장 보통의 연애</td>
      <td>멜로</td>
      <td>thrill5</td>
      <td>10</td>
      <td>한국 드라마를 싫어한다 다양한 케이블 채널들이 참신하고 과감하고 영화 같은 플롯으로...</td>
      <td>1</td>
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
      <th>153</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>cdj2716</td>
      <td>8</td>
      <td>웃을수 있는 유쾌한 영화</td>
      <td>1</td>
    </tr>
    <tr>
      <th>154</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>taejin0329</td>
      <td>7</td>
      <td>산적은 바다로 갔지만 스토리는 산으로 가버린 아쉬운 코믹영화 아무 생각없이 잠시 웃...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>155</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>shin941228</td>
      <td>2</td>
      <td>아쉬웠다고 말하기도 어려운 영화 모든 요소가 아쉬움 조연으로 영화를 살리기엔 역부족...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>156</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>jisdchm</td>
      <td>9</td>
      <td>영화관에서 내내 웃을만큼 재미있었던 영화</td>
      <td>1</td>
    </tr>
    <tr>
      <th>157</th>
      <td>해적: 바다로 간 산적</td>
      <td>모험</td>
      <td>jinuri80</td>
      <td>8</td>
      <td>손예진 ㅤㄸㅒㅁ에 좀 글리는 영화</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>274 rows × 6 columns</p>
</div>


