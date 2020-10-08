---
layout: post
title: Labor Frisch Elasticity Altongi replicate[course by_prof. Shim]
categories:
  - Study_econ&math
tags:
  - economics
  - Frisch
  - Elasticity
  - Altongi
  - Moon  
last_modified_at: 2020-05-18
use_math: true
---
### homework3

* [link for homework3](https://drive.google.com/uc?export=view&id=1AKvpllSkUOeNq4K5z6HWzIoVPzGSJmDn)  


## Exercise: Frisch Labor Supply Elasticity in Korea 

### Consider the following representative consumer's problem.

$$\begin{aligned}
max\mathbb{E}_{0}\sum^{\infty}_{t=0}\beta^{t}\bigl[lnC_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr]\end{aligned}$$
subject to
$$C_{t} + K_{t+1} = W_{t}H_{t} + (1+r_{t}-\delta)K_{t} + \Pi_{t}$$ where
$W_{t}$ is the hourly wage rate, $r_{t}$ is the rental rate of capital,
and $\Pi_{t}$ is the profit from the firm's problem.\
A representative firm facing production function
$Y_{t} = H^{\alpha}_{t} K^{1-\alpha}_{t}$ maximizes its profit at the
competitive market.

**1. Derive the labor supply equation.**

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{0}\sum_{n=1}^{\infty}\beta^{t}[\ln C_{t}-B\dfrac{H_{t}^{1+\phi}}{1+\phi} + \Lambda_{t}[W_{t}H_{t} + (1+r_{t}-\delta)K_{t} + \Pi_{t}-C_{t} - K_{t+1}]]$$

Then, given the fact that, for the household, the firm's profit is
given, I can derive the following FOCs.

$$\begin{aligned}
&:\dfrac{1}{C_{t}}=\Lambda_{t}
\\ [H_{t}]&:BH_{t}^{\phi}=\Lambda_{t}W_{t}
\\ [K_{t+1}]&:\Lambda_{t}=\mathbb{E}_{t}\beta\Lambda_{t+1}[1+r_{t+1}-\delta] = \mathbb{E}_{t}\beta\Lambda_{t+1}[1+r_{t+1}] \end{aligned}$$

At the last equation, I redefine $r_{t}$ as real interest rate, so that\

*real interest rate$(r_{t})=1+$ rental rate of
capital($r_{t})- \delta$*.

Thus, the labor supply equation is defined as follows:
$$H_{t} = \bigl(\frac{\lambda W_{t}}{B}\bigr)^{\frac{1}{\phi}} = \bigl(\frac{W_{t}}{BC_{t}}\bigr)^{\frac{1}{\phi}}$$

**2. Express the labor supply equation you derived in (i) in logs (eg.
$c_{t} = \log C_{t}$). Estimate the Frisch labor supply elasticity using
the MACRO data available in South Korea (go as far as you can). Describe
your steps and methodology and clearly indicate the data sources.**

The log-transformed labor supply equation is as follows:
$$\begin{aligned}
 h_{t} = \frac{1}{\phi}(w_{t}-\log B - c_{t}) \end{aligned}$$

The first-differenced log-transformed labor supply equation is:
$$\begin{aligned}
 \Delta h_{t} = \frac{1}{\phi}\Delta w_{t} -\frac{1}{\phi}\Delta c_{t} \end{aligned}$$

Thus, $\frac{1}{\phi}$ means Frisch Labor Supply Elasticity. Given the
endogeneity problem between $w_{t}$ and $h_{t}$, I used first-lagged
$w_{t}$ and second-lagged $w_{t}$ as instrument variables for $w_{t}$,
and conducted 2sls regression using Macro variables in South Korea. All
variables are detrended by the H-P filter. In here, I define $H_{t}$ as
(intensive margin of labor supply)\*(extensive margin of labor supply)
since the main requirment of this question is using MACRO data.

As in Table 1, the estimations of Frisch Labor Supply Elasticity derived
by using the Macro data in South Korea are all around 1, which is not
statistically significant.

To see whether the above methodology is valid, I additionally conducted
the following process. Assuming all agents are rational, then, by
equation (4), we can derive the following equation:
$$\beta\Lambda_{t}[1+r_{t}] = r_{t-1} + \epsilon_{\Lambda_{t}}$$

Using taylor expansion, $$\begin{aligned}
(\lambda_{t-1} + \epsilon_{\Lambda_{t}}) &= \lambda_{t-1} + \frac{1}{\lambda_{t}}[\lambda_{t-1} + \epsilon_{\Lambda_{t}} - \lambda_{t-1}]
\\&= \lambda_{t-1} + u_{t} \quad \text{where},\ u_{t} = \frac{\epsilon_{\lambda_{t}}}{\lambda_{t-1}}.\end{aligned}$$

![window](https://drive.google.com/uc?export=view&id=10pbTceV4HP80N9bkHYRAzsFEPyk-pKsy)  

![window](https://drive.google.com/uc?export=view&id=1u0iHqxVGD4ASUrlO7qyVZz1-KJaoeETC)  

![window](https://drive.google.com/uc?export=view&id=1MrfN7gNZr00E8sQLsCxtve-yqQ_UsEqO)  

Then, since $\ln[1+r_{t}] \approx r_{t}$, $$\begin{aligned}
\ln\beta + \lambda_{t} + r_{t} &= \lambda_{t-1} + u_{t}
\\ \Delta \lambda_{t} &= -(\ln\beta + r_{t}) + u_{t}  \quad \text{where},\ u_{t} = \frac{\epsilon_{\lambda_{t}}}{\lambda_{t-1}}\end{aligned}$$

From equation (2), we can derive $\Delta c_{t} = -\Delta\lambda_{t}$.
$$\Delta c_{t} = \ln\beta + r_{t} - u_{t}  \quad \text{where},\ u_{t} = \frac{\epsilon_{\lambda_{t}}}{\lambda_{t-1}}$$

Then, finally, $$\begin{aligned}
 \Delta h_{t} = \frac{1}{\phi}\Delta w_{t} -\frac{1}{\phi} (\ln\beta + r_{t}) + \epsilon_{t} \end{aligned}$$

As above, $\frac{1}{\phi}$ means Frisch Labor Supply Elasticity. Given
the endogeneity problem between $w_{t}$ and $h_{t}$, I used first-lagged
$w_{t}$ and second-lagged $w_{t}$ as instrument variables for $w_{t}$,
and conducted 2sls regression using real interest rate instead of
consumption index in South Korea. All variables are detrended by the H-P
filter.

As in Table 2, the estimations of Frisch Labor Supply Elasticity derived
by using the Macro data in South Korea are all around 2, which is not
statistically significant.

***\* Data description***

\(a\) **Using OECD $\&$ World Bank data**

$\ H_{t}(\text{yearly}) =$ Total hours worked

\- Total hours worked = (average hour worked per employed) \*
(population) \* (Employment to population ratio)

\- average hours worked per employed: Average annual hours worked by
persons engaged from OECD data

\- population : Population ages 15-64 ($\%$ of total population)

\- Employment to population ratio : Employment to population ratio, 15+,
total

\- period: 1991-2017

\(b\) **Using Penn World Table 9.1 data**

$\ H_{t}(\text{yearly}) =$ Total hours worked

\- Total hours worked = (average hour worked per employed) \*
(employment)

\- Employment : Number of persons engaged (in millions) from Penn World
Table 9.1

![window](https://drive.google.com/uc?export=view&id=1eY0KmFT-NW2QWr6jwZvkC7oe32nuisM9)  

![window](https://drive.google.com/uc?export=view&id=1Ff-LdJHwtTpBR1vyMRLcdUOwRZw3_Kwq)  

![window](https://drive.google.com/uc?export=view&id=1llY5S1efwSdfQ0vtgiQv35CHN698XWUb)  

\- average hours worked per employed: Average annual hours worked by
persons engaged from Penn World Table 9.1

\- period: 1991-2017

\(c\) **Using FRED data**

$\ H_{t}(\text{yearly}) =$ Total hours worked

\- Total hours worked = (average hour worked per employed) \*
(employment)

\- Employment : Employment in the Republic of Korea (South Korea)

\- average hours worked per employed: Average Annual Hours Worked per
Employed Person in the Republic of Korea (South Korea)

\- period: 1991-2011

**common variables**

. $r_{t}\text{(yearly)} =$ real interest rate

\- Interest Rates, Discount Rate for Republic of Korea from FRED

\- period: 1991-2017(a,b)/ 1991-2011(c)

. $C_{t}$(yearly) : Consumption

\- domestic consumption expenditure index from BOK data

\- period: 1991-2017(a,b)/ 1991-2011(c)

. $W_{t}$(yearly) : average annual wages

\- data from OECD

\- period: 1991-2017(a,b)/ 1991-2011(c)

**3. Survey the literature that estimates Frisch labor elasticity using
the Korean MICRO data. Discuss the relationship between your estimate
from (ii) and the micro estimates (your answer can exceed two
paragraphs).**

There exist several literatures that estimate Frisch labor elasticity
using the Korean MICRo data. I cited Moon $\&$ Song (2016) paper in
which they conducted estimation using the Korea Labor and Income Panel
Study (KLIPS) from 2000 to 2008. Moon $\&$ Song (2016) found that the
point that estimates at the extensive margin are greater than those at
the intensive margin, but not statistically significant. Table 3 show
the estimation of Frisch labor elasticity using KLIPS by Moon $\&$ Song
(2016). All estimates are aroung 0.2 and statistically significant.

![window](https://drive.google.com/uc?export=view&id=1HDHXgpd-xm1TifPHBOMTjAjHSdpCijpe)  

However, the above-suggested estimates are statistically smaller that
those described in table 1 and 2. This may be due to the existence of
extensive margin of labor supply only considered in MACRO-side
estimation. To check the influence of extensive margin to total hours, I
derived the standard deviations of $H_{t}$ using data from FRED. All
variables are detrended by HP-filter. According to the estimations, the
standard deviations of Total Hours Worked is almost 0.02, but the
standard deviations of intensive margin of labor supply is merely 0.007,
and standard deviations of extensive margin of labor supply is 0.017.
Thus, over the 70$\%$ of entire variations in Total Hours Worked is
explained by not intensive margin but extensive margin. Even though the
intensive margins of both estimations cannot be viewed as same with each
other, it can be stated that the differences between those estimates
using MICRO and MACRO data each may be from the existence of extensive
margin of labor supply considered in MACRO-side estimation.

**4. Suppose that utility function takes the following form, instead of
$lnC_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi},$
$$u(C_{t}, H_{t}) = \frac{\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{1-\gamma}}{1-\gamma}$$
Re-do question 1. Discuss the differences between your solution to
question 1 and solution to this question.**

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{0}\sum_{n=1}^{\infty}\beta^{t}[ \frac{\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{1-\gamma}}{1-\gamma} + \Lambda_{t}[W_{t}H_{t} + (1+r_{t}-\delta)K_{t} + \Pi_{t}-C_{t} - K_{t+1}]]$$

Then, given the fact that, for the household, the firm's profit is
given, I can derive the following FOCs.

$$\begin{aligned}
&:\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{-\gamma}=\Lambda_{t}
\\ [H_{t}]&:\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{-\gamma}\cdot BH_{t}^{\phi}=\Lambda_{t}W_{t}
\\ [K_{t+1}]&:\Lambda_{t}=\mathbb{E}_{t}\beta\Lambda_{t+1}[1+r_{t+1}-\delta] = \mathbb{E}_{t}\beta\Lambda_{t+1}[1+r_{t+1}] \end{aligned}$$

Thus, the labor supply equation is defined as follows:
$$H_{t} = \bigl(\frac{W_{t}}{B}\bigr)^{\frac{1}{\phi}}$$

**5. Re-do question 2 with the labor supply curve you derive in question
4 and explain your findings.**

The log-transformed labor supply equation is as follows:
$$\begin{aligned}
 h_{t} = \frac{1}{\phi}(w_{t}-\log B) \end{aligned}$$

The first-differenced log-transformed labor supply equation is:
$$\begin{aligned}
 \Delta h_{t} = \frac{1}{\phi}\Delta w_{t} \end{aligned}$$

Thus, $\frac{1}{\phi}$ means Frisch Labor Supply Elasticity. Given the
endogeneity problem between $w_{t}$ and $h_{t}$, I used first-lagged
$w_{t}$ and second-lagged $w_{t}$ as instrument variables for $w_{t}$,
and conducted 2sls regression using the above-described variables. All
variables are detrended by the H-P filter. This process is same with
question 2.

As Table 4, the estimations of Frisch Labor Supply Elasticity derived by
using the Macro data in South Korea are all over 2, which is not
statistically significant. These results are a bit bigger than the
results suggested in Table 1 and 2. This seems to be due to the
reduction of explanatory variable compared to the question 1 and 2. In
equation (16), unlikely to the equation (6) and (11), the
first-differeced log-transformed wage is the only explanatory variable.
Moreover, as we can see in Table 1 and 2, the coefficients of other
explanatory variables are positive but not statistically significant.
Thus, the estimates of 2sls regression including one explanatory
variable conducted in this question, may absolve the influences of other
explanatory variables used in the above regressions, which may be linked
to the bigger Frisch labor elasticity.

![window](https://drive.google.com/uc?export=view&id=1fB_9aq9fF7Fm2BPwlth9LGAijwIgNBTr)  

![window](https://drive.google.com/uc?export=view&id=1emMo8ziuSAbFWkJ_S91yYE8hRQt7oS5n)  

![window](https://drive.google.com/uc?export=view&id=1GWGDWZ6fTAQ0ZFHwjGopskEEwX4FDZH8)  

9 Altonji, J. (1986) *Intertemporal Substitution in Labor Supply:
Evidence from Micro Data*. Journal of Political Economy, 94(3),
S176-S215.

Weh-sol Moon, Sungju Song. (2016) *Estimating Labor Supply Elasticity in
Korea*. Korean Journal of Labour Economics, 39(2), 35-51.
