---
layout: post
title:  "Technical Note(preprocessing)_2nd paper"
description: Prediction of industrial accident workers' return to the original work, using Machine Learning.
date:   2020-03-10
use_math: true
---
### Technical note(preprocessing)
Prediction of industrial accident workers' return to the original work, using Machine Learning.

## Data Preprocessing


```python
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Dawis Kim/Dropbox/산재보험패널/1st_cohort/STATA/pswci_5th_03_long_data_V4.csv', engine='python')

df=df.assign(order_id=1)
df["order_id"]=df.groupby(["pid"])["order_id"].cumsum()

df2 = df[["pid", "order_id", "emp06", "jobservice", "medservice", "psyservice", "gender", "age", "a003008", "edu", "workperiod", "con", "disa02", "disa15", 
          "disa06", "c001007", "c001006", "c002004", "c001008", "c002012", "c004012", "ba004014", "ba004001", "c002018", "gb010001", 
          "gb010007", "c002007", "c002008", "ba002001", "ba002002", "ba002003", "bb002002", "bb002004"]]

#Return to work 더미변수#
df["emp06"].unique()
df["rtor_dummy"]=df["emp06"]
for i in range(len(df["emp06"])):
    if df["rtor_dummy"][i]=='원직장복귀자':
        df["rtor_dummy"][i]=1
    else:
        df["rtor_dummy"][i]=0
df["rtor_dummy"].fillna(-1, inplace=True)
df["rtor_dummy"]=df.groupby('pid')['rtor_dummy'].transform(max)
        

#재활서비스 더미변수#
df["jobservice"] = df["jobservice"].astype('category')
df["medservice"] = df["medservice"].astype('category')
df["psyservice"] = df["psyservice"].astype('category')

df["js_dummy"]=df["jobservice"].cat.codes
df["med_dummy"]=df["medservice"].cat.codes
df["psy_dummy"]=df["psyservice"].cat.codes

df["js_dummy"]=df.groupby('pid')['js_dummy'].transform(max)
df["med_dummy"]=df.groupby('pid')['med_dummy'].transform(max)
df["psy_dummy"]=df.groupby('pid')['psy_dummy'].transform(max)

df["js_dummy"].replace({1:0, 0:1}, inplace=True)
df["med_dummy"].replace({1:0, 0:1}, inplace=True)
df["psy_dummy"].replace({1:0, 0:1}, inplace=True)

#성별 변수#
df["gender"] = df["gender"].astype('category')
df["sex_dummy"]=df["gender"].cat.codes
df["sex_dummy"]=df.groupby('pid')['sex_dummy'].transform(max)

#나이 변수#
df["age_dummy"]=df["age"]
df["age_dummy"].replace({"20대 이하":1, "30대":2, "40대":3, "50대":4, "60대 이상":5}, inplace=True)
df["age_dummy"].fillna(0, inplace=True)
df["age_dummy"]=df.groupby('pid')['age_dummy'].transform(max)

#배우자 유무#
df["a003008"].unique()
df["S"]=df["a003008"]
df["S"].replace({"미혼":1, "사별":3, "혼인":4, "이혼":5, "별거":6}, inplace=True)
df["S"].fillna(0, inplace=True)

df["spouse"]=df["S"]
for i in range(len(df["spouse"])):
    if df["spouse"][i]==0:
        if df["order_id"][i]==1:
            df["spouse"][i]=0
        else:
            df["spouse"][i] =df["spouse"][i-1]
    else:
        df["spouse"][i]=df["spouse"][i]  
        
df["spouse_dummy"]=df["spouse"]       
for i in range(len(df["spouse_dummy"])):
    if df["spouse"][i]==4:
        if df["order_id"][i]==5:
            df["spouse_dummy"][i]=1
        else:
            df["spouse_dummy"][i]=0        
    else:
        df["spouse_dummy"][i]=0

df.drop(["S","spouse"], axis=1, inplace=True)
df["spouse_dummy"]=df.groupby('pid')['spouse_dummy'].transform(max)

#교육수준#
df["edu"].unique()
df["edu_dummy"]=df["edu"]
df["edu_dummy"].replace({"고졸":1, "초졸":0, "중졸":0, "무학":0, "대졸이상":2}, inplace=True)
df["edu_dummy"].fillna(-1, inplace=True)
df["edu_dummy"]=df.groupby('pid')['edu_dummy'].transform(max)

#근로기간#
df["workperiod"].unique()
df["work_dummy"]=df["workperiod"]
df["work_dummy"].replace({'1개월 미만':1, '1개월 ~ 2개월 미만':2, '2개월 ~ 3개월 미만':3, '4개월 ~ 5개월 미만':5, '5개월 ~ 6개월 미만':6
  ,'6개월 ~ 1년 미만':7, '1년 ~ 2년 미만':8, '2년 ~ 3년 미만':9, '3년 ~ 4년 미만':10, '4년 ~ 5년 미만':11, '5년 ~ 10년 미만':12, '10년 ~ 20년 미만':13
  ,'20년 이상':14}, inplace=True)
df["work_dummy"].fillna(-1, inplace=True)
df["work_dummy"]=df.groupby('pid')['work_dummy'].transform(max)

#요양기간#
df["con"].unique()
df["con_dummy"]=df["con"]
df["con_dummy"].replace({'3개월이하':1, '3개월초과~6개월이하':2, '6개월초과~9개월이하':3, '9개월초과~1년이하':4, '1년초과~2년이하':5
  ,'2년초과':6}, inplace=True)
df["con_dummy"].fillna(-1, inplace=True)
df["con_dummy"]=df.groupby('pid')['con_dummy'].transform(max)

#장애여부#
df["disa02"].unique()
df["dis_dummy"]=df["disa02"]
df["dis_dummy"].replace({'장해등급 있음(1~14급)':1, '장해등급 없음(무장해)':0}, inplace=True)
df["dis_dummy"].fillna(-1, inplace=True)
df["dis_dummy"]=df.groupby('pid')['dis_dummy'].transform(max)

#장애등급#
df["dis_level"]=df["disa15"]
df["dis_level"].fillna(-1, inplace=True)
df["dis_level"]=df.groupby('pid')['dis_level'].transform(max)
df["dis_level"].replace({0:15}, inplace=True)

#장애등급범주#
df["disa06"].unique()
df["dis_range"]=df["disa06"]
df["dis_range"].replace({'1~3급':0, '4~7급':1, '8~9급':2, '10~12급':3, '13~14급':4, '무장해':5}, inplace=True)
df["dis_range"].fillna(-1, inplace=True)
df["dis_range"]=df.groupby('pid')['dis_range'].transform(max)

#종사상직위#
df["c001007"].unique()
df["emplev_dummy"]=df["c001007"]
df["emplev_dummy"].replace({'상용직':1, '자영업자 또는 고용주':1, '일용직':0, '임시직':0}, inplace=True)
df["emplev_dummy"].fillna(-1, inplace=True)
df["emplev_dummy"]=df.groupby('pid')['emplev_dummy'].transform(max)

#직종구분#
df["c001006"].unique()
df["emplev_cont"]=df["c001006"]
df["emplev_cont"].replace({'단순노무종사자':0, '판매직(영업직, 매장 판매직, 방문 판매, 노점, 통신판매 등)':1, '서비스직':1,
  '기능원 및 관련기능 종사자 (제조업 기능직, 건설업 기능직, 영?':0, '관리자':0,
  '장치조작, 기계조작 및 조립업무 종사자':2, '사무직원':2, '전문가(각종 전문직, 엔지니어 등)':2, '농어업 숙련직':2}, inplace=True)
df["emplev_cont"].fillna(-1, inplace=True)
df["emplev_cont"]=df.groupby('pid')['emplev_cont'].transform(max)

#전체근로자수#
df["c002004"].unique()
df["labor_dummy"]=df["c002004"]
df["labor_dummy"].replace({'5인 미만':1, '5~9인':2, '10~29인':3, '30~99인':4, '100~299인':5, '300~999인':6, '1000인 이상':7}, inplace=True)
df["labor_dummy"].fillna(-1, inplace=True)
df["labor_dummy"]=df.groupby('pid')['labor_dummy'].transform(max)

#근로시간형태#
df["c001008"].unique()
df["labortime_dummy"]=df["c001008"]
df["labortime_dummy"].replace({'전일제':1, '시간제(파트타임, 아르바이트, 다른 직원들보다 1시간이라도 짧?':0}, inplace=True)
df["labortime_dummy"].fillna(-1, inplace=True)
df["labortime_dummy"]=df.groupby('pid')['labortime_dummy'].transform(max)

#교대제여부#
df["c002012"].unique()
df["laborchange_dummy"]=df["c002012"]
df["laborchange_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["laborchange_dummy"].fillna(-1, inplace=True)
df["laborchange_dummy"]=df.groupby('pid')['laborchange_dummy'].transform(max)

#서면 근로 계약서 작성 여부#
df["c004012"].unique()
df["laborcontract_dummy"]=df["c004012"]
df["laborcontract_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["laborcontract_dummy"].fillna(-1, inplace=True)
df["laborcontract_dummy"]=df.groupby('pid')['laborcontract_dummy'].transform(max)

#사업장 지원 만족도#
df["ba004014"].unique()
df["satisfaction_dummy"]=df["ba004014"]
df["satisfaction_dummy"].replace({'매우 불만족':0, '불만족':0, '보통':1, '매우 만족':2, '만족':2}, inplace=True)
df["satisfaction_dummy"].fillna(-1, inplace=True)
df["satisfaction_dummy"]=df.groupby('pid')['satisfaction_dummy'].transform(max)

#원직장과의 관계 유지#
df["ba004001"].unique()
df["ralation_dummy"]=df["ba004001"]
df["ralation_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["ralation_dummy"].fillna(-1, inplace=True)
df["ralation_dummy"]=df.groupby('pid')['ralation_dummy'].transform(max)

#연간소득#
df["income"]=df["c002018"]*12
df["income"].fillna(-1, inplace=True)
df["income"]=df.groupby('pid')['income'].transform(max)

#흡연여부#
df["gb010001"].unique()
df["smoke"]=df["gb010001"]
df["smoke"].replace({'피운다':1, '피우지 않는다':0, np.nan:-1}, inplace=True)
for i in range(len(df["smoke"])):
    if df["smoke"][i]==-1:
        if df["order_id"][i]==1:
            df["smoke"][i]=df.groupby('pid')['smoke'].max()[df["pid"][i]]
        else:
            df["smoke"][i] = df["smoke"][i-1]
    else:
        df["smoke"][i]=df["smoke"][i]
        
df["smoke_dummy"]=df.groupby('pid')['smoke'].transform(sum)
df["smoker"]=df["smoke"]
for i in range(len(df["smoker"])):
    if df["smoke_dummy"][i]==0 and df["order_id"][i]==5:
        df["smoker"][i]=0
    elif df["smoke_dummy"][i]!=0 and df["order_id"][i]==5:
        if df["smoke"][i]==1:
            df["smoker"][i]=1
        else:
            df["smoker"][i]=2
    else:
        df["smoker"][i]=-1

df["smoker"]=df.groupby('pid')['smoker'].transform(max)
df.drop(["smoke","smoke_dummy"], axis=1, inplace=True)
df["smoke_dummy"]=df["smoker"]
df.drop(["smoker"], axis=1, inplace=True)

#금주여부#
df["gb010007"].unique()
df["alcohol"]=df["gb010007"]
df["alcohol"].replace({'마신다':1, '마시지 않는다':0, np.nan:-1, '3':-1}, inplace=True)
for i in range(len(df["alcohol"])):
    if df["alcohol"][i]==-1:
        if df["order_id"][i]==1:
            df["alcohol"][i]=df.groupby('pid')['alcohol'].max()[df["pid"][i]]
        else:
            df["alcohol"][i] = df["alcohol"][i-1]
    else:
        df["alcohol"][i]=df["alcohol"][i]

for i in range(len(df["alcohol"])):
    if df["alcohol"][i]==3:
        if df["order_id"][i]==1:
            df["alcohol"][i]=df.groupby('pid')['alcohol'].max()[df["pid"][i]]
        else:
            df["alcoho"][i] = df["alcoho"][i-1]
    else:
        df["alcohol"][i]=df["alcohol"][i]

df["alcohol_dummy"]=df.groupby('pid')['alcohol'].transform(sum)
df["alcoholr"]=df["alcohol"]
for i in range(len(df["alcoholr"])):
    if df["alcohol_dummy"][i]==0 and df["order_id"][i]==5:
        df["alcoholr"][i]=0
    elif df["alcohol_dummy"][i]!=0 and df["order_id"][i]==5:
        if df["alcohol"][i]==1:
            df["alcoholr"][i]=1
        else:
            df["alcoholr"][i]=2
    else:
        df["alcoholr"][i]=-1

df["alcoholr"]=df.groupby('pid')['alcoholr'].transform(max)
df.drop(["alcohol","alcohol_dummy"], axis=1, inplace=True)
df["alcohol_dummy"]=df["alcoholr"]
df.drop(["alcoholr"], axis=1, inplace=True)

#월평균 근무일수#
df["c002007"].unique()
df["workday_dummy"]=df["c002007"]
df["workday_dummy"].fillna(-1, inplace=True)
df["workday_dummy"]=df.groupby('pid')['workday_dummy'].transform(max)

#하루평균 근무시간#
df["c002008"].unique()
df["workhour_dummy"]=df["c002008"]
df["workhour_dummy"].fillna(-1, inplace=True)
df["workhour_dummy"]=df.groupby('pid')['workhour_dummy'].transform(max)

#의사로부터의 설명#
df["ba002001"].unique()
df["doctorex_dummy"]=df["ba002001"]
df["doctorex_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["doctorex_dummy"].fillna(-1, inplace=True)
df["doctorex_dummy"]=df.groupby('pid')['doctorex_dummy'].transform(max)

#주기적회복정도 설명#
df["ba002002"].unique()
df["recoveryex_dummy"]=df["ba002002"]
df["recoveryex_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["recoveryex_dummy"].fillna(-1, inplace=True)
df["recoveryex_dummy"]=df.groupby('pid')['recoveryex_dummy'].transform(max)

#치료기간적정 여부#
df["ba002003"].unique()
df["cureperiod_dummy"]=df["ba002003"]
df["cureperiod_dummy"].replace({'적정하였음':1, '부족하였음':0}, inplace=True)
df["cureperiod_dummy"].fillna(-1, inplace=True)
df["cureperiod_dummy"]=df.groupby('pid')['cureperiod_dummy'].transform(max)

#주치의직업상담 여부#
df["bb002002"].unique()
df["jobdoctorex_dummy"]=df["bb002002"]
df["jobdoctorex_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["jobdoctorex_dummy"].fillna(-1, inplace=True)
df["jobdoctorex_dummy"]=df.groupby('pid')['jobdoctorex_dummy'].transform(max)

#직업복귀소견서 여부#
df["bb002004"].unique()
df["jobreturnopinion_dummy"]=df["bb002004"]
df["jobreturnopinion_dummy"].replace({'예':1, '아니오':0}, inplace=True)
df["jobreturnopinion_dummy"].fillna(-1, inplace=True)
df["jobreturnopinion_dummy"]=df.groupby('pid')['jobreturnopinion_dummy'].transform(max)




#new table#
df1=df[["pid", "order_id", "rtor_dummy", "js_dummy", "med_dummy", "psy_dummy", "sex_dummy", "age_dummy", "spouse_dummy", 
        "edu_dummy", "work_dummy", "con_dummy", "dis_dummy", "dis_level", "dis_range", "emplev_dummy", "emplev_cont", "labor_dummy", 
        "labortime_dummy", "laborchange_dummy", "laborcontract_dummy", "satisfaction_dummy", "ralation_dummy", "income", "smoke_dummy", 
        "alcohol_dummy", "workday_dummy", "workhour_dummy", "doctorex_dummy", "recoveryex_dummy", "cureperiod_dummy", "jobdoctorex_dummy", 
        "jobreturnopinion_dummy"]]

df1.to_csv("C:/Users/Dawis Kim/Dropbox/산재보험패널/1st_cohort/STATA/1st_cohort_treated_variable.csv")

```
