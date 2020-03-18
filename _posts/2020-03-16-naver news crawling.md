---
layout: post
title:  "Crawling_ Naver News."
description: Crawling.
date:   2020-03-16
use_math: true
---
### Naver News Crawling.
Crawling

## Naver News Crawling


```python
import requests 
from bs4 import BeautifulSoup as bs
import time
import re
import pandas as pd
from datetime import datetime
```


```python
def get_data(query, page):
  contents_list=[]
  title_list = []
  result = [] 
  time_total = []

  for i in range(page):
    url = 'https://search.naver.com/search.naver'
    params = {
      'where': 'news',
      'query': query,
      'sm': 'tab_pge',
      'sort': '0',
      'photo': '0',
      'field': '0',
      'reporter_article':'', 
      'pd': '0',
      'ds': '',
      'de': '',
      'docid': '',
      'nso': 'so:r,p:all,a:all',
      'mynews': '0',
      'start': str(10*i+1),
      'refresh_start': '0'
    }

    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36'}
    res = requests.get(url=url, params=params, headers=headers)
    soup = bs(res.text,'html.parser')
    temp = soup.select('dl > dd.txt_inline > a')
    temp_title = soup.select('a._sp_each_title')

    output = []
    for link in temp:
        if "href" in link.attrs:
            temp_output = link.attrs['href']
            output.append(temp_output)

    output1 = []
    time_inter = []
    for url in output:
        res1 = requests.get(url=url)
        soup1 = bs(res1.text,'html.parser')
        temp1 = soup1.select('div._article_body_contents')
        time_content = soup1.select('span.t11')
        output1.append(temp1)
        time_inter.append(time_content)
        time.sleep(1)

    output2 = []
    time2=[]
    for i,j in zip(output1,time_inter):
        time2.append(j[0].text)
        for j in i:
          temp2 = j.text
          output2.append(temp2)

    for i, j, h in zip(output2, temp_title, time2):
      i = re.sub('[^0-9a-zA-Zㄱ-힗]', '', i).replace("flash오류를우회하기위한함수추가functionflashremoveCallback","").strip()
      contents_list.append(i)
      j = j.text
      title_list.append(j)
      time_total.append(h)

    appended_output = pd.DataFrame({'날짜':time_total, '제목':title_list, '내용':contents_list})
    time.sleep(3)
 
  result.append(appended_output)
  resulted_output = pd.concat(result)
  return resulted_output

```


```python
query = '코로나'
get_data(query, 2)
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
      <th>날짜</th>
      <th>제목</th>
      <th>내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020.02.09. 오후 5:31</td>
      <td>신종코로나 25번 환자의 아들·며느리도 확진…국내 총 27명(종합)</td>
      <td>4일잔기침며느리가25번에전파추정의심환자888명검사중서울연합뉴스김잔디기자9일국내에서신...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020.02.09. 오전 10:15</td>
      <td>신종코로나 확진자 1명 추가…국내 총 25명·의심환자 960명(종합)</td>
      <td>25번째확진자는73세한국여성중국광둥성방문자의가족중앙방역대책본부9일오전9시기준발표서울...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020.02.09. 오후 5:03</td>
      <td>신종 코로나 확진 2명 추가…25번 환자 아들 부부</td>
      <td>국내에서신종코로나바이러스감염증환자2명이추가로발생했습니다중앙방역대책본부는오늘9일오후4...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020.02.09. 오전 9:30</td>
      <td>신종코로나 중국서 이틀째 사망자 80명 넘어…확진 2천656명↑(종합)</td>
      <td>누적사망811명확진3만7천여명신규확진3천명대2천명대줄어병원서환자의료진80여명집단감염...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020.02.09. 오전 9:54</td>
      <td>韓 수출·GDP 전망치 줄하향…신종 코로나로 ‘몸살’</td>
      <td>신종코로나바이러스감염증신종코로나이확산하면서세계주요투자은행IB과해외경제연구기관들이한국...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020.02.09. 오전 10:37</td>
      <td>"국내 신종코로나 4번 환자 퇴원…세 번째 완치"</td>
      <td>국내신종코로나바이러스감염증환자1명이더완치돼퇴원했습니다국내신종코로나확진자가운데세번째퇴...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020.02.09. 오전 9:02</td>
      <td>삼성, 신종 코로나 피해 협력사에 2조6천억 긴급 지원</td>
      <td>중국정보관리가이드라인배포지원센터도운영지디넷코리아이은정기자삼성이신종코로나바이러스감염증...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020.02.09. 오전 10:03</td>
      <td>[속보]신종 코로나, 확진자 1명 추가...총 25명</td>
      <td>연합뉴스9일국내신종코로나바이러스감염증신종코로나확진자가1명추가로발생했다이로써국내신종코...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020.02.09. 오후 1:11</td>
      <td>신종 코로나 사망자 811명으로 '사스' 넘어...치사율은?</td>
      <td>신규확진자줄고사망자늘어박세열기자중국에서신종코로나바이러스총사망자수가811명으로늘었다2...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020.02.09. 오후 3:08</td>
      <td>"신종코로나 25번 환자, '광둥성 방문' 며느리에게 옮은 듯"(종합)</td>
      <td>정은경중앙방역대책본부장가족내전파추정역학조사중서울연합뉴스서한기신선미기자보건당국은국내2...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020.02.09. 오전 6:03</td>
      <td>"신종코로나 무서워"…편의점 상비약 판매 급증</td>
      <td>서울연합뉴스이신영기자신종코로나바이러스감염증신종코로나확산여파로편의점상비약판매가늘고있다...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020.02.09. 오전 10:43</td>
      <td>신종코로나 누적 사망자 811명…사스 사망자 774명 추월</td>
      <td>신종코로나확진자3만7천500여명치사율은사스10보다약한2미만마스크를쓴중국베이징시민들U...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2020.02.09. 오후 3:08</td>
      <td>"신종코로나 25번 환자, '광둥성 방문' 며느리에게 옮은 듯"(종합)</td>
      <td>정은경중앙방역대책본부장가족내전파추정역학조사중서울연합뉴스서한기신선미기자보건당국은국내2...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2020.02.09. 오후 2:38</td>
      <td>KDI “신종코로나, 경기에 부정적 영향 불가피…경기회복 제약”</td>
      <td>한국개발연구원KDI은신종코로나바이러스감염증사태와관련해향후경기에어느정도의부정적영향은불...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2020.02.09. 오후 2:27</td>
      <td>[신종 코로나]사스 능가하는 경제충격...쏟아지는 '경고음'</td>
      <td>베이징정지우특파원중국후베이성우한에서시작된신종코로나바이러스감염증이중국을넘어세계경제에도...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2020.02.09. 오후 1:12</td>
      <td>법무부 “‘신종 코로나’ 관련 입국 제한으로 현지에서 499명 입국 차단”</td>
      <td>신종코로나바이러스감염증의확산을막기위해일부외국인에대한입국제한조치가시행된가운데법무부가지...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2020.02.09. 오전 10:04</td>
      <td>[속보] 신종코로나 확진자 1명 추가…국내 총 25명</td>
      <td>News1국내에서신종코로나바이러스감염증신종코로나환자1명이추가발생했다9일질병관리본부중...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2020.02.09. 오후 1:00</td>
      <td>자동차부터 기계업종까지…제조업 코로나 쇼크 전방위 확산</td>
      <td>원부자재수급에노심초사차부품업체는일부휴업한곳도화학조선기자재기계업종도수출차질납기지연등손...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020.02.09. 오후 2:31</td>
      <td>민주 공천 심사 본격…광주·전남 정가, 신종코로나 영향 촉각(종합)</td>
      <td>대면선거운동중단대통령마케팅불허따른영향주목지지기반인지도갖춘후보유리정치신인불리분석더불어...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020.02.09. 오후 7:49</td>
      <td>코로나 폭로의사 리원량 모친 "경찰, 내 아들에 한 짓 해명하라"</td>
      <td>눈물을흘리는리원량의어머니SCMP캡처연합뉴스신종코로나바이러스감염증우한폐렴발생사실을최초...</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
