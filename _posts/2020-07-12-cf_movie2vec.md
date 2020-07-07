---
layout: post
title: cf_movie2Vec
categories:
  - Programming
  - Collaborative Filtering
  - movie2Vec
tags:
  - programming
last_modified_at: 2020-07-07
use_math: true
---

### cf_movie2Vec

```python
# mount with google drive
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    


```python
import pandas as pd
pd.set_option('display.max.columns', None)
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
```


```python
# Dataset

df = pd.read_csv('/content/drive/My Drive/Activities/학회_dsLab/텍스트처리스터디/movie2vec/movie_tags_integ.csv', encoding = 'euc-kr') # movie_tags_integ.csv file
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
      <th>title</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>아이캔스피크</td>
      <td>[`좋다`, `감동`, `눈물`, `나문희`, `위안부`, `재미`, `일본`, `...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>덕혜옹주</td>
      <td>[`역사`, `역사왜곡`, `없다`, `왜곡`, `덕혜옹주`, `미화`, `일본`,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>신과함께2</td>
      <td>[`재미`, `없다`, `스토리`, `지루하다`, `평점`, `좋다`, `왜`, `...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1987</td>
      <td>[`좋다`, `없다`, `역사`, `배우`, `가슴`, `최고`, `강동원`, `눈...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>터널</td>
      <td>[`하정우`, `재미`, `없다`, `지루하다`, `터널`, `연기`, `좋다`, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.title.unique()
```




    array(['아이캔스피크', '덕혜옹주', '신과함께2', '1987', '터널', '타짜신의손', '검사외전', '변호인',
           '택시운전사', '럭키', '오만과편견', '맨프럼어스', '프로메테우스', '레미제라블', '맘마미아',
           '캡틴아메리카윈터솔져', '조커', '7광구', '레터스투줄리엣', '500일의썸머', '베놈',
           '님아그강을건너지마오', '덩케르크', '아바타', '너의이름은', '마담프루스트의비밀정원', '더헌트', '왓치맨',
           '겨울왕국2', '히든피겨스', '리틀포레스트', '아가씨', '너의결혼식', '부산행', '신과함께1',
           '살인의추억', '장화홍련', '써니', '초능력자', '마당을나온암탉', '바닷마을다이어리', '캐롤',
           '미녀와야수', '싱스트리트', '베이비드라이버', '날씨의아이', '시간을달리는소녀', '아수라', '불한당',
           '다크나이트라이즈', '시네마천국', '헝거게임판엠의불꽃', '무뢰한', '오늘의연애', '공조', '블랙스완',
           '목격자', '007스카이폴', '간신', '수상한그녀', '82년생김지영', '광해왕이된남자', '국가대표',
           '그녀', '그래비티', '그랜드부다페스트호텔', '나이브스아웃', '도둑들', '러브로지', '레디플레이어원',
           '미드소마', '박쥐', '반창꼬', '베를린', '부당거래', '뷰티인사이드', '성난변호사', '숨바꼭질',
           '신세계', '싸이보그지만괜찮아', '악마를보았다', '양들의침묵', '올드보이', '왕의남자', '위플래쉬',
           '인랑', '최종병기활', '추격자', '코코', '킬빌', '인턴', '라라랜드', '캣츠',
           '원스어폰어타임인할리우드', '이터널선샤인', '위대한개츠비', '인터스텔라', '해피데스데이', '암수살인',
           '비긴어게인', '닥터두리틀', '마션', '벤자민버튼의시간은거꾸로간다', '주토피아', '서치', '사바하',
           '빅쇼트', '그것', '유열의음악앨범', '인사이드아웃', '베테랑', '암살', '완벽한타인', '범죄도시',
           '지금만나러갑니다', '엑시트', '변신', '국가부도의날', '독전', '동주', '내부자들', '가장보통의연애',
           '그것만이내세상', '남한산성', '해적바다로간산적', '돈', '뺑반', '도어락', '봉오동전투', '곤지암',
           '1917', '서치아웃', '덕구', '가버나움', '어벤져스엔드게임', '인셉션', '작은아씨들', '그린북',
           '포드V페라리', '보헤미안랩소디', '위대한쇼맨', '알라딘', '어벤져스인피니티워', '컨저링2', '샌안드레아스',
           '레버넌트', '본얼티메이텀', '인비저블맨', '어바웃타임', '조조래빗'], dtype=object)




```python
# convert a string representation of list to list
import ast

tags = [ast.literal_eval(taglist.replace("`","'")) for taglist in df.tags]
df.tags = tags
```

# Movie2Vec


```python
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine
```


```python
# use pre-trained model for korean : 200 dimensional vector for each word
model = Word2Vec.load('/content/drive/My Drive/Activities/학회_dsLab/텍스트처리스터디/movie2vec/ko.bin') # ko.bin file (korean word2vec model)
```


```python
movie_id = {}
for i in range(df.shape[0]):
  movie_id[df.title[i]]=i
```


```python
tag_vec_size = len(model.wv['강아지'])
movie_vector_matrix = np.empty((df.shape[0],tag_vec_size))

for id in range(df.shape[0]):
  tags_list= df.tags.loc[id]
  num_tags=len(df.tags.loc[id])
  tag_vec_array = np.empty((num_tags, tag_vec_size))

  tag_vec_array
  for i in range(num_tags):
    try:
      tag_vec_array[i]= model.wv[tags_list[i]]
    except KeyError:
      tag_vec_array[i] = np.zeros([1,tag_vec_size])
  movie_vector = np.mean(tag_vec_array, axis=0)
  movie_vector_matrix[id] = movie_vector
```


```python
def compute_cosine(result_vector):
  '''
  Input:
  ------
  result_vector: numpy array to compare with each and every movie vector
  to calculate the cosine similarities between them.
  Output:
  -------
  cosine_similarities: sorted numpy array of movie indices and the
  corresponding cosine similarity; descending (so closest movies should show
  up first)
  '''
  cosine_similarities = np.zeros(df.shape[0])
  for i in range(df.shape[0]):
    cosine_similarities[i] = 1 - cosine(result_vector, movie_vector_matrix[i])
  sorted_indices = np.argsort(cosine_similarities)[::-1] # Get sorted movie indices
  cosine_similarities = cosine_similarities[sorted_indices] # Sort cosine similarities
  return sorted_indices, cosine_similarities
```


```python
def get_result_vector(pos_movies, pos_tags, neg_movies, neg_tags):
  '''
  Takes in lists of indices of positive movies and negative movies, and lists
  of positive tags and negative tags. Gets corresponding movie vectors and tag
  vectors, then adds the positive and subtracts the negative to get the result
  vector to return.
  '''
  # Get movie vectors
  tag_vec_size = len(model.wv['강아지']) # Get size of a tag vector
  pos_movie_vectors = np.zeros((len(pos_movies),tag_vec_size))
  for idx, movie_idx in enumerate(pos_movies):
    pos_movie_vectors[int(idx)] = movie_vector_matrix[movie_id[movie_idx]]
  if len(neg_movies) != 0:
    neg_movie_vectors = np.zeros([len(neg_movies),tag_vec_size])
    for idx, movie_idx in enumerate(neg_movies):
      neg_movie_vectors[int(idx)] = movie_vector_matrix[movie_id[movie_idx]]
  else:
    neg_movie_vectors = np.zeros([1,tag_vec_size])
    
  # Get tag vectors
  if len(pos_tags) != 0:
    pos_tag_vectors = np.zeros([len(pos_tags),tag_vec_size])
    for idx, tag in enumerate(pos_tags):
      pos_tag_vectors[int(idx)] = model.wv[tag]
  else:
    pos_tag_vectors = np.zeros([1,tag_vec_size])
  if len(neg_tags) != 0:
    neg_tag_vectors = np.zeros([len(neg_tags),tag_vec_size])
    for idx, tag in enumerate(neg_tags):
      neg_tag_vectors[int(idx)] = model.wv[tag]
  else:
    neg_tag_vectors = np.zeros([1,tag_vec_size])
    
  # Get result vector
  result_vector = np.sum(pos_movie_vectors, axis=0) \
                        + np.sum(pos_tag_vectors, axis=0) \
                        - np.sum(neg_movie_vectors, axis=0) \
                        - np.sum(neg_tag_vectors, axis=0)
  return result_vector
```


```python
def alt_recommend_movies(pos_movies=[], pos_tags=[], neg_movies=[],neg_tags=[], num_recs=10):
  '''
  Instead of Doc2Vec, takes the average of all the tag vectors created by
  Word2Vec for each movie to get a vector representation of the movie.
  Then, add/subtracts those vectors to get a vector representation of the
  output. Then finds cosine similarity to all other movies, and returns
  the top movies that are closest.
  Input:
  ------
  pos_movies: list of indices of positive tags (tags to add)
  neg_tags: list of indices of negative tags (tags to subtract)
  num_recs: int of num recommendations desired (default is 10)
  Output:
  -------
  alt_similar_movies: list of tuples of similar movies and their cosine
  distances.
  '''
  #self.vectorize_movies()
  result_vector = get_result_vector(pos_movies, pos_tags, neg_movies, neg_tags)
  sorted_indices, cosine_similarities = compute_cosine(result_vector)
  alt_similar_movies = []
  i = 0
  while len(alt_similar_movies) < num_recs:
    # If similar movie is same as movies entered, skip over it
    if df.title[sorted_indices[i]] in pos_movies or sorted_indices[i] in neg_movies:
      i += 1
    else:
      alt_similar_movies.append((df.title.loc[sorted_indices[i]],cosine_similarities[i]))
      i += 1
  recs = [] # Final recommendations to be outputted
  for movie, sim in alt_similar_movies:
    rounded_sim = '{0:.3f}'.format(sim)
    recs.append((movie, rounded_sim))
  return recs
```


```python
def mv_recommend(input=[], movies_cf=[], num = 3):
  '''
  mv_recommend let us get the results only from the movies from the result of collaborative filtering
  ------
  input : list of one movie input
  movies_cf : list of movies from the result of collaborative filtering
  num : number of movies to recommend from the movies_cf
  '''
  i = 1
  for title, similarity in alt_recommend_movies(pos_movies=input, num_recs=len(df)-1):
    if title in movies_cf:
      # if float(similarity) > 0.75: # movie except low similarity
        print('Rank ',i,' : ',title, ' (Similarity : ',similarity,')',sep='')
        if i == num: break
        i = i+1
```

### Example


```python
mv_recommend(input=['라라랜드'],movies_cf=['위대한 쇼맨','비긴 어게인','보헤미안 랩소디','아이 캔 스피크','알라딘'], num = 3)
```

    Rank 1 : 알라딘 (Similarity) : 0.830)
    

## Add CF Result


```python
cf = pd.read_csv('/content/drive/My Drive/Activities/학회_dsLab/텍스트처리스터디/movie2vec/movie_cf.csv')
cf.head()
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
      <th>Unnamed: 0</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>007스카이폴</td>
      <td>1917</td>
      <td>벤자민버튼의시간은거꾸로간다</td>
      <td>부산행</td>
      <td>실미도</td>
      <td>위대한개츠비</td>
      <td>라이언일병구하기</td>
      <td>7광구</td>
      <td>뷰티인사이드</td>
      <td>본얼티메이텀</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1917</td>
      <td>007스카이폴</td>
      <td>포드V페라리</td>
      <td>라이언일병구하기</td>
      <td>위플래쉬</td>
      <td>부산행</td>
      <td>실미도</td>
      <td>아수라</td>
      <td>샌안드레아스</td>
      <td>시네마천국</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1987</td>
      <td>검사외전</td>
      <td>변호인</td>
      <td>태극기휘날리며</td>
      <td>도둑들</td>
      <td>신과함께1</td>
      <td>부산행</td>
      <td>레미제라블</td>
      <td>7번방의선물</td>
      <td>수상한그녀</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500일의썸머</td>
      <td>노트북</td>
      <td>미녀와야수</td>
      <td>라라랜드</td>
      <td>겨울왕국2</td>
      <td>캡틴아메리카윈터솔져</td>
      <td>코코</td>
      <td>레미제라블</td>
      <td>8월의크리스마스</td>
      <td>비긴어게인</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7광구</td>
      <td>신과함께1</td>
      <td>부산행</td>
      <td>7번방의선물</td>
      <td>타짜신의손</td>
      <td>노트북</td>
      <td>검사외전</td>
      <td>007스카이폴</td>
      <td>숨바꼭질</td>
      <td>베놈</td>
    </tr>
  </tbody>
</table>
</div>




```python
def movie_recommend(input, num=3):
  if num > 9:
    print("Number of recommendation should be less than 10")
  else:
    input_ls = []
    input_ls.append(input)
    mv_cf = cf[cf['Unnamed: 0']==input].values.tolist()[0][1:9]
    mv_recommend(input = input_ls, movies_cf = mv_cf ,num=num)
```


```python
movie_recommend('007스카이폴')
```

    Rank 1 : 부산행 (Similarity : 0.601)
    Rank 2 : 1917 (Similarity : 0.585)
    Rank 3 : 7광구 (Similarity : 0.530)
    
