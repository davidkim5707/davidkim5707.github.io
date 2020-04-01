---
layout: post
title: Adavanced Macro_assignment 3[_prof. Jang]
categories:
  - Study_econ&math
tags:
  - economic
  - log-linearization
  - stochastic growth model
  - shock
  - King, Plosser and Rebelo(1988)
  - Prescott(1986)
last_modified_at: 2020-03-19
use_math: true
---

### assignment 3

* [link for assignment 3](https://drive.google.com/uc?export=view&id=16xDXIuBlv-wNL__aOrldyNYTnZkRQOaV)  


Estimate the persistence and magnitude of productivity shock
------------------------------------------------------------

I first have to construct quarterly series of capital stock $(K_{t})$
and labor input $(N_{t})$. From the BOK data, I get 1.3 times of the
real GDP in 1970 according to the statistic announcement of BOK and use
it as $K_{0}$ and construct the series of capital stock by using the
capital accumulation equation
$K_{t+1}=(1-\delta)K_{t}+I_{t}(quarterly, \delta=0.0125)$. I use the
data from OECD to calculate labor input $N_{t}=$(total worked hours per
year)$/(365*24)$. The period for calculation is 1970Q1 to 2017Q4 since
the data for total worked hours is available only from that period. I
assumed a cobb-Douglas production function, and labor-augmenting
technology progress.
$$Y_{t}=A_{t}K_{t}^{1-\alpha}(N_{t}X_{t})^{\alpha}$$
$$X_{t}/X_{t-1} = \gamma_{X}$$ I calculate the solow residual with my
data set ($\alpha=0.61$, I derive it using the labor income share date
from BOK).
$$S_{t}=\dfrac{Y_{t}}{K_{t}^{1-\alpha}N_{t}^{\alpha}}=A_{t}X_{t}^{\alpha}$$
Take a log to Solow residual. $$\log S_{t}=\log A_{t}+\alpha logX_{t}$$
Since the problem assumed the temporary variation in productivity
follows an AR(1) process in logs $A_{t}=\bar{A}e^{a_{t}}$ and using the
above assumptions we get $$\log A_{t}=\log \bar{A}+\alpha_{t}$$
$$\log X_{t}-\log X_{t-1}=\log \gamma_{t}$$ Thus,
$$\log S_{t}=\log \bar{A}+a_{t}+\alpha(\log X_{0}+t\log \gamma_{t})$$ To
get rid of the trend part in the $\log S_{t}$ and get the cyclical part
$a_{t}$, I use the HP filter. Then an AR(1) regression is run to get the
persistence and magnitude. $$a_{t}=\rho_{a}a_{t-1}+\epsilon_{a,t}$$
$$\epsilon_{a,t}\sim\mathit{N}(0,\sigma_{a}^{2})$$
$$where, \rho_{a}=0.7888, \sigma_{a}=0.0573$$

![image](table_1.png){width="1.0\\linewidth"}

[\[fig:onecol\]]{#fig:onecol label="fig:onecol"}

Estimate the persistence and magnitude of government shock
----------------------------------------------------------

I proceed in a similar direction with the above problem. From the
government spending data from BOK, I take a log to the government
spending and use the HP filter to calculate the $g_{t}$. Then run a
regression to get the AR(1) parameter and the standard deviation.
$$g_{t}=\rho_{g}g_{t-1}+\epsilon_{g,t}$$
$$\epsilon_{g,t}\sim\mathit{N}(0,\sigma_{g}^{2})$$
$$where, \rho_{g}=0.7077, \sigma_{g}=0.0198$$

![image](table_2.png){width="1.0\\linewidth"}

[\[fig:onecol\]]{#fig:onecol label="fig:onecol"}

Casting the system of equations
-------------------------------

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{t}\sum_{n=1}^{\infty}\beta_{t}[\log C_{t}+\dfrac{B(1-N_{t})^{1-\psi}}{1-\psi}]+\mathbb{E}_{t}\sum_{n=1}^{\infty}\Lambda_{t}[A_{t}K_{t}^{1-\alpha}(N_{t}X_{t})^{\alpha}-K_{t+1}+(1-\delta)K_{t}-C_{t}-G_{t}]$$
Then, I can get FOCs. $$[C_{t}]:\beta^{t}\dfrac{1}{C_{t}}=\Lambda_{t}$$
$$[N_{t}]:\beta^{t}B(1-N_{t})^{-\psi}=\alpha\Lambda_{t}A_{t}K_{t}^{1-\alpha}X_{t}^{\alpha}N_{t}^{\alpha-1}$$
$$[K_{t+1}]:\Lambda_{t}=\mathbb{E}_{t}\Lambda_{t+1}[(1-\alpha)A_{t+1}K_{t+1}^{-\alpha}N_{t+1}X_{t+1}^{\alpha}+1-\delta]$$
$$[TVC]:\lim_{t\to\infty}\Lambda_{t}K_{t+1}=0$$ Also, I have the
equation for the capital accumulation
$$K_{t+1}=(1-\delta)K_{t}+A_{t}K_{t}^{1-\alpha}N_{t}X_{t}^{\alpha}-C_{t}-G_{t}$$
Now I detrend the above equations using
$c_{t}=\dfrac{C_{t}}{X_{t}},k_{t}=\dfrac{K_{t}}{X_{t}},g_{t}=\dfrac{G_{t}}{X_{t}},\gamma_{t}=\dfrac{X_{t+1}}{X_{t}}$,
and $\lambda_{t}=\dfrac{\Lambda_{t}X_{t}}{\beta^{t}}$.
$$\dfrac{1}{c_{t}}=\lambda_{t}$$
$$B(1-N_{t})^{-\psi}=\alpha\lambda_{t}A_{t}k_{t}^{1-\alpha}N_{t}^{\alpha-1}$$
$$\gamma_{x}\lambda_{t}=\mathbb{E}_{t}\lambda_{t+1}[(1-\alpha)A_{t+1}k_{t+1}^{1-\alpha}N_{t+1}+1-\delta]$$
$$\gamma_{x}k_{t+1}=(1-\delta)k_{t}+A_{t}k_{t}^{1-\alpha}N_{t}^{\alpha}-c_{t}-g_{t}$$
Then, we log linearize the above equations
($\hat{c_{t}}=\dfrac{c_{t}-c}{c}$ where c is a steady state value for
$c_{t}$). $$-\hat{c_{t}}=\hat{\lambda}$$
$$\frac{\psi N}{1-N}\hat{N_{t}}=\hat{\lambda_{t}}+\hat{A_{t}}+(1-\alpha)(\hat{k_{t}}-\hat{N_{t}})$$
$$\hat{\lambda}_{t}=\mathbb{E}_{t}[\hat{\lambda}_{t+1}+\eta\hat{A}_{t+1}-\alpha\eta\hat{k}_{t+1}+\alpha\eta\hat{N}_{t+1}]$$
$$\hat{A_{t}}+(1-\alpha)\hat{k_{t}}+\alpha\hat{N_{t}}=\mathbb{E}_{t}[s_{i}\phi\hat{k}_{t+1}+s_{c}\hat{c_{t}}+s_{g}\hat{g_{t}}-s_{i}(\phi-1)\hat{k_{t}}]$$
$$where, s_{c}=\dfrac{c}{y},s_{g}=\dfrac{g}{y},s_{i}=\dfrac{(\gamma_{x}-1+\delta)k}{y},\eta=\dfrac{(1-\alpha)A_{t}k_{t}^{-\alpha}N_{t}^{\alpha}}{(1-\alpha)A_{t}k_{t}^{-\alpha}N_{t}^{\alpha}+1-\delta}=\dfrac{r+\delta}{1+r}$$
$$, and \hspace{0.3cm} \phi=\dfrac{\gamma_{x}}{\gamma_{x}-(1-\delta)}$$
Therefore, we can construct matrices to express above relationship. From
(1) and (2),
$$M_{cc}\left[\begin{array}{c} \hat{c_{t}} \\ \hat{N_{t}} \end{array} \right] = M_{cs}\left[\begin{array}{c} \hat{k_{t}} \\ \hat{\lambda_{t}} \end{array} \right]+ M_{ce}\left[ \begin{array}{c} \hat{A_{t}} \\ \hat{g_{t}} \end{array} \right]$$
Where,
$$M_{cc}=\begin{bmatrix} \ -1&0 \\ 0&\frac{\psi N}{1-N}-(\alpha-1) \end{bmatrix}$$
$$M_{cs}=\begin{bmatrix} \ 0&1 \\ (1-\alpha)&1 \end{bmatrix}$$
$$M_{ce}=\begin{bmatrix} \ 0&0 \\ 1&0 \end{bmatrix}$$ From (3) and (4),
$$M_{ss}(B)\left[\begin{array}{c} \hat{k}_{t+1} \\ \hat{\lambda}_{t+1} \end{array} \right] = M_{sc}(B)\left[\begin{array}{c} \hat{c}_{t+1} \\ \hat{N}_{t+1} \end{array} \right]+ M_{ce}(B)\left[ \begin{array}{c} \hat{A}_{t+1} \\ \hat{g}_{t+1} \end{array} \right]$$
Where,
$$M_{ss,0}=\begin{bmatrix} \ -\eta\alpha&1 \\ -s_{i}\frac{\gamma_{x}}{\gamma_{x}-(1-\delta)}&0 \end{bmatrix}$$
$$M_{ss,1}=\begin{bmatrix} \ 0&-1 \\ (1-\alpha)-s_{i}\frac{1-\delta}{\gamma_{x}-(1-\delta)}&0 \end{bmatrix}$$
$$M_{sc,0}=\begin{bmatrix} \ 0&-\eta\alpha \\ 0&0 \end{bmatrix}$$
$$M_{sc,1}=\begin{bmatrix} \ 0&0 \\ s_{c}&-\alpha \end{bmatrix}$$
$$M_{se,0}=\begin{bmatrix} \ -\eta&0 \\ 0&0 \end{bmatrix}$$
$$M_{se,1}=\begin{bmatrix} \ 0&0 \\ -1&s_{g} \end{bmatrix}$$

Express the output, investment, labor productivity, and wage
------------------------------------------------------------

$$\hat{y_{t}}=\hat{A_{t}}+(1-\alpha)\hat{k_{t}}+\alpha\hat{N_{t}}$$
$$\hat{y_{t}}=s_{c}\hat{c_{t}}+s_{i}\hat{i_{t}}+s_{g}\hat{g_{t}}$$
$$\hat{i_{t}}=\dfrac{1}{s_{i}}\hat{y_{t}}-\dfrac{s_{c}}{s_{i}}\hat{c_{t}}-\dfrac{s_{g}}{s_{i}}\hat{g_{t}}$$
$$\hat{y_{t}}-\hat{N_{t}}=\hat{A_{t}}+(1-\alpha)\hat{k_{t}}+(\alpha-1)\hat{N_{t}}$$
$$\hat{w_{t}}=\hat{A_{t}}+(1-\alpha)\hat{k_{t}}+(\alpha-1)\hat{N_{t}}$$

Calibrate the structural parameters
-----------------------------------

1.  I drived $\alpha=0.61$ using the labor income share date from BOK.

2.  I took quarterly depreciation rate, $\delta=0.0125$ from Pen World
    Table 9.0

3.  $\gamma_{X}=1.014$ since the average growth rate of GDP for period
    from 1987 to 2018 is 1.014

4.  $s_{g}=0.182$ using the average $\dfrac{G}{Y}$ for the given period

5.  $r=0.017$ from the equation that (nominal interest rate-inflation
    rate) for the given period(I use three-year corporate bonds as
    nominal interest rate and CPI as inflation rate).

6.  $N=0.2924$ from the average of the series $N_{t}$

7.  $\beta=0.996$ from $\beta(1+r)=\gamma_{X}$

8.  $\psi=0.4$ from Park, Choonsung(2017)

Print elements of Matrix
------------------------

I plugged in the above parameters and system of equations to calculate
the $M_{cc}$, $M_{cs}$, $M_{ce}$, $M_{ss}(B)$, $M_{sc}(B)$, and
$M_{se}(B)$. $$M_{cc}=\begin{bmatrix} \ -1&0 \\ 0&0.8032 \end{bmatrix}$$
$$M_{cs}=\begin{bmatrix} \ 0&1 \\ 0.3900&1 \end{bmatrix}$$
$$M_{ce}=\begin{bmatrix} \ 0&0 \\ 1&0 \end{bmatrix}$$
$$M_{ss}(B)=M_{ss,o}+M_{ss,1}(B)$$ $$M_{sc}(B)=M_{sc,0}+M_{sc,1}(B)$$
$$M_{se}(B)=M_{se,0}+M_{se,1}(B)$$
$$M_{ss,0}=\begin{bmatrix} \ -0.0177&1 \\ 13.4054&0 \end{bmatrix}$$
$$M_{ss,1}=\begin{bmatrix} \ 0&-1 \\ -13.4451&0 \end{bmatrix}$$
$$M_{sc,0}=\begin{bmatrix} \ 0&-0.0177 \\ 0&0 \end{bmatrix}$$
$$M_{sc,1}=\begin{bmatrix} \ 0&0 \\ -0.4677&0.6100 \end{bmatrix}$$
$$M_{se,0}=\begin{bmatrix} \ -0.029&0 \\ 0&0 \end{bmatrix}$$
$$M_{se,1}=\begin{bmatrix} \ 0&0 \\ 1&-0.182 \end{bmatrix}$$

Print values of other variables
-------------------------------

$$W=\begin{bmatrix} \ 1.0125&0.0915 \\ 0.0091&0.9793 \end{bmatrix}$$
$$R=\begin{bmatrix} \ 0&0 \\ -0.0499&0 \end{bmatrix}$$
$$Q=\begin{bmatrix} \ 0.1312&-0.0136 \\ 0.0012&-0.0001 \end{bmatrix}$$
$$\mu=\begin{bmatrix} \ 0.9653&0 \\ 0&1.0390 \end{bmatrix}$$
$$M_{ke}=\begin{bmatrix} \ 0.9653&0.1139&-0.0111 \\ 0&0.7888&0 \\ 0&0&0.7077 \end{bmatrix}$$
$M_{ke}$ shows how decision rules are decided and updated by state
variables.
$$H=\begin{bmatrix} \ -0.6530&-0.1898&0.0271 \\ 0.6530&0.1898&-0.0271 \\ -0.3274&1.0087&0.0338 \\ 0.1903&1.6153&0.0206 \\ 0.5177&0.6066&-0.0132 \\ -0.3285&4.3575&-0.4245 \\ -0.0227&0.0343&0.0007 \end{bmatrix}$$
$H$ shows, under certain state variables, how optimal decision updates
control vector and flow vector.
$$\left[\begin{array}{c} \lambda_{t} \\ C_{t} \\ N_{t} \\ Y_{t} \\ W_{t} \\ I_{t} \\ r_{t} \end{array}\right]=(H)\left[\begin{array}{c} K_{t} \\ A_{t} \\ G_{t} \end{array}\right]$$

Plot the impulse response to a one percent increase of $\epsilon_{a,t}$
-----------------------------------------------------------------------

![[]{label="fig:long"}](figure_1.png){#fig:long width="1.0\\linewidth"}

[\[fig:onecol\]]{#fig:onecol label="fig:onecol"}

The IRFs show that 1 percent positive TFP shock would increase capital
for first 6 periods and then the capital would start going back to its
steady state. This happens because the TFP shock increase the marginal
productivity of the capital which leads to a jump in investment.
Moreover, the consumption is increased due to the increase in output.
However the increase is not that dramatic because of the motive for
smoothing its consumption(at first, the interest rate becomes
substantial enough to raise consumption, which is not long-lasting). The
jump in the output is caused by the increased amount of labor due to the
increased wage(and the marginal productivity of the labor). Interest
rate moves identically to the MPK.

Plot the impulse response to a one percent decrease of $\epsilon_{a,t}$
-----------------------------------------------------------------------

![[]{label="fig:long"}](figure_2.png){#fig:long width="1.0\\linewidth"}

[\[fig:onecol\]]{#fig:onecol label="fig:onecol"}

The oil shock would decrease the marginal productivity of capital, which
is shown in the change in the interest rate. Therefore, the investment
shows a stiff drop and consequently the capital goes below the steady
state level. After the 5 periods of decrease, the capital slowly
recovers its steady state level. Due to the decline in marginal
productivity of lobor shown in the decrease in the wage, the demand for
labor decreases and the labor goes through a sudden drop. After 5
periods, the labor recovers its steady state level and actually even
exceeds the original level. It is because of the wealth effect coming
from the decrease in consumption. The consumption shows a little decline
since there is a motive for smoothing out the consumption. Overall, the
output drops but returns to its initial level but it does not goes above
the steady state level because the labor increase from the wealth effect
is not that great compared to the decrease in the capital.

Plot the impulse response to a one percent increase of government spending shock
--------------------------------------------------------------------------------

![[]{label="fig:long"}](figure_3.png){#fig:long width="1.0\\linewidth"}

[\[fig:onecol\]]{#fig:onecol label="fig:onecol"}

The government spending shock decreased the consumption(substitution
effect) but the decline in the consumption is small due to the smoothing
behavior. The increase in the labor happens because the wealth effect
from the decreased consumption. Because the supply of labor increases,
the wage decreases(MPL decreases). The investment is decreased because
the government spending crowded out the investment. However, it does not
affect the capital too much and the capital shows a very stable
movement. Therefore, the stable capital and increased labor boosts up
the output a little bit. The increased labor input increases the MPK as
shown in the interest rate.

Compute the government expenditure multiplier
---------------------------------------------

$$\dfrac{dY}{dG}=\dfrac{dY/Y}{dG/G}\times\dfrac{1}{s_{g}}$$ From the
matrix H,
$$\dfrac{dY/Y}{dG/G}=0.0206\hspace{0.2cm} / \hspace{0.2cm} s_{g}=0.182$$
$$\therefore \dfrac{dY}{dG}=0.1131$$ Therefore, a ten dollars increase
in the government spending only increases the output about two dollars.
As a result, it is hard to say the government spending is an effective
way to boost the output in the given economy.

999
