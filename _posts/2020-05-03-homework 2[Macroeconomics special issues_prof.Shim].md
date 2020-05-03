---
layout: post
title: Macroeconomics special issues_Homework2[_prof. Shim]
categories:
  - Study_econ&math
tags:
  - economics
  - Labor Wedge
  - Gali
  - Hall
  - Chang & Kim 
  - Karabarbounis
last_modified_at: 2020-05-03
use_math: true
---
### homework2

* [link for homework2](https://drive.google.com/uc?export=view&id=1K0moAYBE2I0StyqNwj861uyYkWz5Z3Wa)  

## Exercise: Labor Wedge in South Korea 


Consider the following representative consumer's problem.

$$\begin{aligned}
max\mathbb{E}_{0}\sum^{\infty}_{t=0}\beta^{t}\bigl[lnC_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr]\end{aligned}$$
subject to
$$C_{t} + K_{t+1} = W_{t}H_{t} + (1+r_{t}-\delta)K_{t} + \Pi_{t}$$ where
$W_{t}$ is the hourly wage rate, $r_{t}$ is the rental rate of capital,
and $\Pi_{t}$ is the profit from the firm's problem.\
A representative firm facing production function
$Y_{t} = H^{\alpha}_{t} K^{1-\alpha}_{t}$ maximizes its profit at the
competitive market.

**1. Derive the equation for labor wedge**

Following Chang and Kim (2007, AER), define the labor wedge as below:
$$\ln Wedge_{t} = \ln MRS_{t} - \ln \frac{Y_{t}}{H_{t}} +constant.$$

Then, from the Houshold's utility function, labor wedge equation can be
transformed as follows:
$$\ln Wedge_{t} = \ln B\frac{H_{t}^{\phi}}{C_{t}^{-1}} - \ln \frac{Y_{t}}{H_{t}} +constant.$$

**2. Suppose that $\phi = 1$. Determine the value of $B$ and $\alpha$ by
matching steady-state values for (1) total hours worked per year
(normalize total time endowment as one) and (2) average labor income
share in South Korea. This procedure is called as calibration.**

\(1\)

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{0}\sum_{n=1}^{\infty}\beta^{t}[\ln C_{t}+B\dfrac{H_{t}^{1+\psi}}{1+\psi} + \Lambda_{t}[H_{t}^{\alpha}K_{t}^{1-\alpha}-K_{t+1}+(1-\delta)K_{t}-C_{t}]]$$

Then, I can get FOCs. $$[C_{t}]:\dfrac{1}{C_{t}}=\Lambda_{t}$$
$$[N_{t}]:BH_{t}^{\psi}=\alpha\Lambda_{t}H_{t}^{\alpha-1}K_{t}^{1-\alpha}$$
$$[K_{t+1}]:\Lambda_{t}=\mathbb{E}_{t}\beta\Lambda_{t+1}[(1-\alpha)H_{t}^{\alpha}K_{t+1}^{-\alpha}+1-\delta]$$
$$[\Lambda_{t}]:W_{t}H_{t} + (1+r_{t}-\delta)K_{t} + \Pi_{t}-C_{t} - K_{t+1} = 0$$
$$[TVC]:\lim_{t\to\infty}\Lambda_{t}K_{t+1}=0$$

At the steady state, all above equations are transformed as follows:
$$\begin{aligned}
BH^{1+\phi} = \alpha\frac{K}{C}(\frac{H}{K})^{\alpha}\\
(1-\alpha)(\frac{H}{K})^{\alpha}+1-\frac{1}{\beta} = \delta\\
(\frac{H}{K})^{\alpha} - \frac{C}{K} = \delta\end{aligned}$$

Then, we can get
$\alpha(\frac{H}{K})^{\alpha}\frac{K}{C} =\frac{K}{C}(1-\frac{1}{\beta})+1$
by simple calculation using (3),(4) from which we can derive the
following equation: $$BH^{1+\phi} = \frac{K}{C}(1-\frac{1}{\beta})+1$$

Since we assume that $\phi = 1$, we finally acquire the following
equation: $$BH^{2} = \frac{K}{C}(1-\frac{1}{\beta})+1$$

In addtion, using the fact that $r_{t}-\delta = MPN_{t}$, we can get the
following equation from equation (3) as well: $$\beta(r+1) = 1$$

At the steady state, $r = 0.014$ is derived from the average series of
real interest rate, $r_{t}$, wich can be calculated by nominal interest
rate-inflation rate. The period for calculation is 1987Q1 to 2019Q4
since nominal interest rates data is avalilable only from that period.(I
used three-year corporate bonds as nominal interest rate and the growth
rate of CPI as inflation rate). Thus, by simple calculation,
$\beta = 0.988$

I use tha data from BOK to calculate consumption input $C_{t}$ for the
given period. The steady state value for household consumption, $C$, is
derived from the average of the series $C_{t}$. Thus, in my calculation,
$C= 1.36e+07$.

From the BOK data, I get 1.3 times of the real GDP in 1970 according to
the statistic announcement of BOK and use it as $K_{0}$ and construct
the series of capital stock by using the capital accumulation equation
$K_{t+1} = (1-\delta)K_{t}+I_{t}$(quarterly, $\delta = 0.012$). The
steady state value for capital, $K$, is derived from the average of the
series $K_{t}$ for the priod 1987Q1 to 2019Q4. Thus, in my calculation,
$K= 3.28e+08$.

I use the data from OECD to calculate intensive margin of labor using
labor input $H_{t} =$(total worked hours per year)/$(365 \times 24)$ for
the given period. The total worked hours data OECD is yearly data, so
that I transformed it to total hours worked per year data and expanded
it to quarterly data for calculation. The steady state value for total
hours worked per year, $H$, is derived from the average of the series
$H_{t}$. Thus, in my calculation, $H= 0.2741$.

Consequently, we can finally determine the value of $B$ as follows:
$$B = \frac{1}{H^{2}}(\frac{K}{C}(1-\frac{1}{\beta})+1) \approx 9.41$$

\(2\)

I used the labor income share data in BOK(Bank Of Korea) for the given
period. The average value of labor income share is approximately 0.6.

**3. Using the data on $C_{t}, Y_{t}, H_{t}$ (download data from the
Bank of Korea or KOSIS) and the parameter values that you determine in
question 2, obtain the series of log of labor wedge for the sample
period that you have. Notice that you need to detrend the data. Plot the
labor wedge and explain whether it is zero as the RBC model predicts. Is
it comparable to the labor wedge that Chang and Kim (2007, AER) reports
in Figure 2?**

From the BOK data, I download the domestic consumption expenditure
index, and GDP to construct $C_{t},$ and $Y_{t}$ respectively. As
described above, I use the data from OECD to calculate labor input
$H_{t} =$(total worked hours per year)/$(365 \times 24)$. The period for
calculation is 1970-2018. All data are yearly data.
$$\ln Wedge_{t} = \ln B\frac{H_{t}^{\phi}}{C_{t}^{-1}} - \ln \frac{Y_{t}}{H_{t}} +constant.$$

From the above equation, I obtain the series of log of labor wedge for
1970-2018 in South Korea. After calculation, I detrended the series of
log of labor wedge and $H_{t}$, respectively. The plots for the two
detrended series are as follows:

![window](https://drive.google.com/uc?export=view&id=1yEK6QrhqlZs04Bot4zkk4dHCgfIQFhVM)  

From the Figure 1, we can notice that the Labor-Market Wedge in South
Korea have shown some deviations from zero which is not consistent with
the RBC model. That is, unlikely the prediction of RBC model, there
exists some gaps between the firm's marginal product of labor (MPN) and
the household's marginal rate of substitution (MRS).

![window](https://drive.google.com/uc?export=view&id=1yEK6QrhqlZs04Bot4zkk4dHCgfIQFhVM)  

![window](https://drive.google.com/uc?export=view&id=1qMbw4WnYBAPQHKYP3JZh6zyBmZV-dsG9)  


Furthermore, the variation of cyclical components of Labor-Market Wedge
is more fluctuate than the Hours Worked. As suggested in Figure 2, the
higher volatility of wedge is somewhat inconsistent with the result in
Chang $\&$ Kim(2007) who argued that the wedge of the United States is
highly correlated with hours worked, and its volatility is the same
order of magnitude as hours worked.

**4. According to Hall (1997), we can guess that the labor wedge arises
from the preference shock. Can you interpret the labor wedge computed in
question 3 as Hall (1997) did? Check whether the labor wedge is really
exogenous or not by applying the Granger causality test (follow Gal´ı et
al (2007): see Table 3 and the main text and use data on South Korea).**

Hall(1997) showed that, in a model based on standard neoclassical
ingredients, the prime driving force in fluctuations turns out to be
Preference Shock defined as the following equation:
$$x = c - y +(1+\phi)n - log \alpha$$

According to Gali et al(2007), the above Preference shock equation can
be transformed as follows: $$\begin{aligned}
x_{t} &= c_{t} - y_{t} +(1+\phi)n_{t} - log \alpha \\&= (mrs_{t}-mpn_{t})+\xi_{t} \\&= -(\mu_{t}^{p} -\mu_{t}^{w}) +\xi_{t}\end{aligned}$$

Hall(1997)'s assumption of perfect competition in both goods and labor
markets implies $\mu_{t}^{p} = \mu_{t}^{w}$, which allows him to
interpret the variable $x_{t}$, as a preference shock, for under his
assumption $x_{t} = \xi_{t}$. However, if $x_{t}$ indeed reflects
exogenous preference shocks, it should be invariant to any other type of
disturbance. That is, the null hypothesis of preference shocks implies
that $x_{t}$ should be exogenous. To test this, Gali et al(2007)
implemented two tests that reject the null of exogeneity, thus rejecting
the preference shock hypothesis.

One of them is testing the hypothesis of no Granger causality from a
number of variables to the gap measure, $x_{t}$. Following their
setting, I used cyclical component of log GDP, the nominal interest
rate, and the yield spread. I used three-year corporate bonds as nominal
interest rate and the series of 10-year government bonds minus
three-year corporate bonds as yield spread for the period 2001-2018
since the data for 10-year government bond in South Korea is available
from that period. I proceed the test by using STATA.

![window](https://drive.google.com/uc?export=view&id=1DAr0pjsJrrXBbUPYNj83Z5p02G0DewDj)  


![window](https://drive.google.com/uc?export=view&id=1-pj6e443n1B0tLbNm14juwjRc4AKdFyb)  


![window](https://drive.google.com/uc?export=view&id=13F9BLI-1LaL0UPHxvyLNyDnUo9LBc7bk)  


Figure 3 displays the result of testing the hypothesis of no Granger
causality from a number of variables to the gap measure, $x_{t}$. These
statistics in Figure 3 correspond to the bivariate tests using
alternative lag lenths. They indicate that the null of no Granger
causality is rejected for nominal interest rate, and yield spread, at
convensional significance levels. Thus, even though the null hypothesis
of no granger causality from detrended GDP(using both cyclical component
and first differenced) variable cannot be rejected as in Gali et
al(2007), the hypothesis that the labor wedge computed in my setting
mainly reflects variations in preferences seems to be inconsistent with
the evidence of Granger causality. To conclude, I can say that the labor
wedge derived from my setting using data on South Korea is not exogenous
to other types of disturbance, which is not consistent with the ones
Hall(1997) asserted.

![window](https://drive.google.com/uc?export=view&id=1CVF8_Ue0pkW6p8o8zzE-lQbvWItciHeC)  


**5. (Advanced but Interesting Question) According to Karabarbounis
(2014), fluctuations in the labor wedge mostly come from the household
side. In particular, one can decompose the labor wedge as follows:
$$\begin{aligned}
\ln wedge_{t} &= \ln wedge_{t}^{H} + \ln wedge_{t}^{F} \\ \ln wedge_{t}^{H} &= \ln MRS_{t} - \ln W_{t} 
\\ \ln wedge_{t}^{F} &= \ln W_{t} - \ln MPN_{t}\end{aligned}$$ Evaluate
his argument using the Korean data. i.e. is the labor wedge in South
Korea mostly come from the household side? Go far as you can (Hint: How
can we find the hourly wage rate ($W_{t}$) for South Korea? If this is
not that easy, you might rearrange (or adjust) the terms in the above
equations so that you can use the data you can easily obtain.).**

Due to the data limitaion of wages index, I rearranged the above
equations by using the fact that $\frac{W_{t}}{P_{t}} = MRS_{t}$. Thus,
the labor wedge equations can be transformed as follows:
$$\begin{aligned}
\ln wedge_{t} &= \ln wedge_{t}^{H} + \ln wedge_{t}^{F} \\ \ln wedge_{t}^{H} &= \ln MRS_{t} - \ln MPN_{t} -\ln P_{t}
\\ \ln wedge_{t}^{F} &= \ln P_{t}\end{aligned}$$

Thus, I used Consumer Price Index(CPI) as price index from BOK data. The
cyclical components of each labor wedges are shown in Figure 5. As in
Figure 5, After the mid-1980, most of the volatility of labor wedge in
South Korea can be accounted for the labor wedge in Household defined as
Karabarbounis(2014). Until the mid-1980, it seems that the high
volatility of labor wedge related to Firm component offset the
volatility of ones in Household, which might invoke some inconsistencies
between overall labor wedge and household-related labor wage before the
mid-1980.

![window](https://drive.google.com/uc?export=view&id=1EHRywKSND1_Vc3s62CED38Jx9Flbj0Po)  

![window](https://drive.google.com/uc?export=view&id=1tzWTUHue0jJeQyDU-0UcyOBIkZPbr8Qj)  


To check whether above-described transformed-equations are valid or not, 
I used the average annual wages data from OECD for the period of 
1990-2018. While the described high comovements between overall
 and household-related labor wedges in Figure 5 seem to be weaken 
after 2010, overally, the results in Figure 6 derived by using annual 
wages are consistent with the results suggested in Figure 5. 
Thus, in South Korea, while the firm component of the labor 
wedge shows a weak relationship with the overall labor wedge, 
there have existed a tight association between the household 
component of labor wedge and the overall labor wedge.   

![window](https://drive.google.com/uc?export=view&id=18Jk6b-u8Fy6K4_-fWNDavFZ-jf6fnZUG)  

![window](https://drive.google.com/uc?export=view&id=198dfn-cm3jt8LxiZlen1mzAYSt3eAWvK)

Thus, the mechanism behind the labor wedge in South Korea seems to be
highly compatible with the suggetions of Karabarbounis(2014), who showed
that models that generate volatile and countercyclical labor wedges by
modifying the firm side of the neoclassical growth model could be
rejected by the data. To conclude, at least in my setting using data on
South Korea, Karabarbounis(2014) arguements may seem to be substantially
credible.

**Data description and values {#data-description-and-values .unnumbered}**


. $r = i$ - inflation rate

\- i : three-year corporate bonds

\- inflation rate: the growth rate of CPI

\- data from BOK(Bank Of Korea)

\- period: 1987-2019

\- year average: $4.576842\%$ $\&$ quarter average: $1.1442105\%$

. $\alpha=$ labor income share

\- average of labor income share in South Korea

\- data from BOK(Bank Of Korea)

\- period: 1987-2019

\- value: $0.596 \approx 0.6$

. $\delta=$ depreciation rate

\- average of depreciation rates in South Korea

\- data from Penn World Table 9.1

\- period: 1987-2017

\- value: quarter average = $0.012$

. Other remaining variables

\- $C_{t}$(yearly, quarterly) : domestic consumption expenditure index
fromBOK data for

the period 1970-2019.

\- $Y_{t}$(yearly, quarterly) : GDP index from BOK data for the period
1970-2019.

\- $I_{t}$(yearly, quarterly) : total invesment index from BOK data for
the period 1970-2019.

\- $P_{t}$(yearly, quarterly) : Consumer Price Index from BOK data for
the period 1970-2019.

\- $H_{t}$(yearly) : average hours worked per person employed from OECD
data for the

period 1970-2019.

\- $W_{t}$(yearly) : average annual wages from OECD data for the period
1990-2019.

\- 10-year government bonds rate(yearly) from BOK data for the period
2001-2019.

9 Chang $\&$ Kim.(2007) *Heterogeneity and Aggregation: Implications for
Labor-Market Fluctuations*. The American Economic Review, 97(5),
1939-1956.

Hall, R. (1997) *Macroeconomic Fluctuations and the Allocation of Time*.
Journal of Labor Economics, 15(1), 223-250.

Galí, J., Gertler, M., $\&$ López-Salido, J. (2007) *Markups, Gaps, and
the Welfare Costs of Business Fluctuations*. Review of Economics and
Statistics, 89(1), 44-359.

Karabarbounis, L. (2014) *The labor wedge: MRS vs. MPN*. Review of
Economic Dynamics, 17(2), 206-223.
