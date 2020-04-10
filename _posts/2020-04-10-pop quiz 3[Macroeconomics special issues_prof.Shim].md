---
layout: post
title: Macroeconomics special issues_Pop quiz 3[_prof. Shim]
categories:
  - Study_econ&math
tags:
  - economics
  - Stationary
  - Cyclical component
  - Difference
  - HP-filter
last_modified_at: 2020-04-10
use_math: true
---
### pop quiz 3

* [link for pop quiz 3](https://drive.google.com/uc?export=view&id=1yweR7NTCdpkiTDvh1kGsJ0BsbnrYt3bp)  
### Data description 

I used the log-transformed nominal GDP data from BOK(Bank of Korea).
Using HP-filter, I extracted cyclical components from the data. In
addition, I proceeded differencing the data. All process has been done
by Python.

### Check whether the nominal GDP in South Korea shows a specific trend.    

The following figure shows that the nominal GDP in South Korea has been
showing an upward trend.

![png](https://drive.google.com/uc?export=view&id=1_vBXroVeACvCigFR90N4HJToZGMvfsAt)

### Detrending by HP-filter and Difference  

![png](https://drive.google.com/uc?export=view&id=1ZkNe6KtVx2jJGmZ2JA02Tg2qjQVYrmN-)  
![png](https://drive.google.com/uc?export=view&id=1ihdpvn4JBKmN-zbYjrj1PhUqzW1znmfr)  

As we can see in figure 1,2, the first Difference of Nominal GDP is
trend-stationary process likewise the cyclical components of Nominal
GDP. Thus, we can say that Nominal GDP in South Korea follows unit root
process, $I(1)$.

### Python codes for the whole process  

```python
#(0) load required packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import itertools
import pandas_profiling
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
import statsmodels.api as sm
%matplotlib inline

#(1) load data
data = pd.read_csv("nominalgdpkr.csv")

#(2) preprocessing the data
for i in range(len(data)):
    data['gdp'][i] = data['gdp'][i].replace(",","") #erase useless sign in the data
    

data['gdp'] = data['gdp'].astype(float) #type to float
data['gdp'] = data['gdp'].transform(lambda x: np.log(x)) #log-transformation

#(3) extracting cyclical data
names = ['gdp']
columns = [data['gdp']]

cyclical_component_dict = dict()

for name, column in zip(names,columns):
    target = column
    cycle, trend = sm.tsa.filters.hpfilter(target, lamb=1600)
    cyclical_component_dict['{}_cycle'.format(name)] = cycle
    
data_cyclical = pd.DataFrame.from_dict(cyclical_component_dict)
gdp_cycle = data_cyclical['gdp_cycle']
data = data.join(gdp_cycle)

#(4) making difference
data['gdp_diff'] = ""
for i in range(len(data)-1):
    data['gdp_diff'][i+1] = data['gdp'][i+1]-data['gdp'][i]
data = data.replace('', np.nan)
data['gdp_diff'] = data['gdp_diff'].astype(float)

#(5) indexing
data['date'] = pd.to_datetime(data['date'], format='%Y')
data.index = data['date'] #date to index
data.drop('date', axis = 1, inplace = True)

#(6) visualizing
data[['gdp']].plot()
plt.grid(True)
plt.axhline(y=0, color='black', linestyle='--')
plt.legend(loc='center right')
plt.xlabel("")
plt.axvspan('2007-12','2009-06', alpha=0.3, color='grey')
plt.axvspan('2001-03','2001-11', alpha=0.3, color='grey')
plt.axvspan('1990-07','1991-03', alpha=0.3, color='grey')
plt.axvspan('1981-07','1982-11', alpha=0.3, color='grey')
plt.axvspan('1980-02','1980-07', alpha=0.3, color='grey')
plt.axvspan('1973-11','1975-03', alpha=0.3, color='grey')
plt.savefig('1.png')

data.plot()
plt.grid(True)
plt.axhline(y=0, color='black', linestyle='--')
plt.legend(loc='center right')
plt.xlabel("")
plt.axvspan('2007-12','2009-06', alpha=0.3, color='grey')
plt.axvspan('2001-03','2001-11', alpha=0.3, color='grey')
plt.axvspan('1990-07','1991-03', alpha=0.3, color='grey')
plt.axvspan('1981-07','1982-11', alpha=0.3, color='grey')
plt.axvspan('1980-02','1980-07', alpha=0.3, color='grey')
plt.axvspan('1973-11','1975-03', alpha=0.3, color='grey')
plt.savefig('2.png')

colors= ['orange', 'green']
data[['gdp_cycle', 'gdp_diff']].plot(colors= colors)
plt.grid(True)
plt.axhline(y=0, color='black', linestyle='--')
plt.legend(loc='upper right')
plt.ylim((-1,1))
plt.xlabel("")
plt.axvspan('2007-12','2009-06', alpha=0.3, color='grey')
plt.axvspan('2001-03','2001-11', alpha=0.3, color='grey')
plt.axvspan('1990-07','1991-03', alpha=0.3, color='grey')
plt.axvspan('1981-07','1982-11', alpha=0.3, color='grey')
plt.axvspan('1980-02','1980-07', alpha=0.3, color='grey')
plt.axvspan('1973-11','1975-03', alpha=0.3, color='grey')
plt.savefig('3.png')
data
```