---
layout: post
title: Macroeconomics special issues_Homework4[_prof. Shim]
categories:
  - Study_econ&math
tags:
  - economics
  - wealth effect
  - News shock
last_modified_at: 2020-06-04
use_math: true
---
### homework4

* [link for homework4](https://drive.google.com/uc?export=view&id=171RQVCBFT3NvhirtS-efHE59ACxppY7w)  


Exercise: Household Problem with News Shock 

Consider the following representative consumer's problem.

$$\begin{aligned}
max\mathbb{E}_{0}\sum^{\infty}_{t=0}\beta^{t}u(c_{t}, H_{t})\end{aligned}$$
subject to $$c_{t} + p_{t}a_{t+1} = w_{t}H_{t} + (p_{t}+d_{t})a_{t}$$
where $w_{t}$ is the hourly wage rate, $p_{t}$ is the after-dividend
stock price, and $d_{t}$ is dividend payout.\
Assume $u_{c}>0, u_{H}<0, u_{cc}<0, u_{HH}<0,$ and $u_{cH}>0$.

**1. Discuss economic meaning of $u_{cH}>0$ in less than three
sentences.**

It indicates that marginal utility of labor increases as consumption
increases, which means that, if consumption increased, then hours would
decrease. That is, $u_{cH}>0$ indicates the negative correlations
between consumption and labor.

**2. Derive the labor supply equation.**

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{0}\sum_{n=1}^{\infty}\beta^{t}[u(c_{t}, H_{t}) + \Lambda_{t}[w_{t}H_{t} + (p_{t}+d_{t})a_{t}-c_{t} - p_{t}a_{t+1}]$$

Then, I can derive the following FOCs.

$$\begin{aligned}
&:u_{C}=\Lambda_{t}
\\ [H_{t}]&:-u_{H}=\Lambda_{t}w_{t}
\\ [a_{t+1}]&:p_{t}\Lambda_{t}=\mathbb{E}_{t}\beta\Lambda_{t+1}[p_{t+1}+d_{t+1}]\end{aligned}$$

Thus, the labor supply equation is defined as follows:
$$-\frac{u_{H}}{u_{C}}= w_{t}$$

**3. Suppose that the consumer receives news that future TFP will hike.
Note that news do not have any direct impact on $w_{t}$, the wage rate.
Show that consumption and hours worked are negatively correlated under
this type of news shock on TFP (Hint: Use the labor supply equation you
derive in question 2 and go as far as you can). Discuss economic meaning
of your solution.**

Let's suppose a positivie news shock on TFP. people who consume normal
goods are likely to increase their consumptions since they expect an
increment of future income. Thus, the increases in consumption means
that the marginal utility of consumption, $u_{C}$, goes down. Then, by
the follwing labor supply equation, $$-\frac{u_{H}}{u_{C}}= w_{t}$$
given that the current wage is invariant to the news shock on TFP,
$-u_{H}$ should go down as well. That is, $u_{H}$ should increase, which
follows the decrease in $H_{t}$.

This can be interpretted as follows. The positive news shock on TFP
induces a positive expectations on future income for economic agents,
which is \"Wealth Effect\". Therefore, given that the substitute effect
on labor supply is shut down, people have incentives to increase current
consumption and decrease current labor supply by wealth effect.

**4. Suppose that utility function takes the following form.
$$u(C_{t}, H_{t}) = \frac{\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{1-\gamma}}{1-\gamma}$$
Re-do question 3. Discuss the differences between your solution to
question 3 and solution to this question.**

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{0}\sum_{n=1}^{\infty}\beta^{t}[ \frac{\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{1-\gamma}}{1-\gamma} + \Lambda_{t}[w_{t}H_{t} + (p_{t}+d_{t})a_{t}-c_{t} - p_{t}a_{t+1}]$$

Then, I can derive the following FOCs.

$$\begin{aligned}
&:\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{-\gamma}=\Lambda_{t}
\\ [H_{t}]&:\bigl(C_{t} - B\frac{H_{t}^{1+\phi}}{1+\phi}\bigr)^{-\gamma}\cdot BH_{t}^{\phi}=\Lambda_{t}W_{t}
\\ [a_{t+1}]&:p_{t}\Lambda_{t}=\mathbb{E}_{t}\beta\Lambda_{t+1}[p_{t+1}+d_{t+1}]\end{aligned}$$

Thus, the labor supply equation is defined as follows:
$$H_{t} = \bigl(\frac{W_{t}}{B}\bigr)^{\frac{1}{\phi}}$$

With the above labor supply equation, the positive news shock on TFP
does not influence the labor supply directly. This is because the only
incentive to change labor supply is wage in this setting and wage is
invariant to the news shock. The utility function given in this question
represents GHH preferences in which the wealth effect on labor supply is
mute. Thus, contrary to the case in question 3, there exists no wealth
effect on the labor supply in this setting as well as the substitution
effect. Therefore, there is no decrease in labor supply corresponding to
a positive new shock under this setting.

**5. Beaudry and Portier (2006)'s idea to identify the news shock is
that stock price contains information on future TFP. Assume that
$d_{t} = \phi (Z_{t})$ where $f' > 0$ and $Z_{t}$ denotes TFP level of
this economy to capture this idea. Using the consumption Euler equation,
formulate this idea mathematically and provide your economic reasoning
behind the formula you derive. For simplicity of the analysis, assume
that
$u(c_{t}, H_{t}) = \frac{c_{t}^{1-\sigma}}{1-\sigma}-\mathcal{V}(H_{t})$
and $c_{t}= d_{t}$ (market clearing condition; consider a simple Lucas
tree model). If necessary, provide additional assumption(s).**

I first set up a Lagrangian.
$$\mathcal{L}=\mathbb{E}_{0}\sum_{n=1}^{\infty}\beta^{t}[\frac{c_{t}^{1-\sigma}}{1-\sigma}-\mathcal{V}(H_{t}) + \Lambda_{t}[w_{t}H_{t} + (p_{t}+d_{t})a_{t}-c_{t} - p_{t}a_{t+1}]$$

Then, I can derive the following FOCs.

$$\begin{aligned}
&c_{t}^{-\sigma}=\Lambda_{t}
\\ [a_{t+1}]&:p_{t}\Lambda_{t}=\mathbb{E}_{t}\beta\Lambda_{t+1}[p_{t+1}+d_{t+1}]\end{aligned}$$

From equation (8),(9), I can derive the following consumption Euler
equation:
$$p_{t}c_{t}^{-\sigma} =\mathbb{E}_{t}\beta c_{t+1}^{\sigma}[p_{t+1}+d_{t+1}]$$
which can be transformed as follows, $$\begin{aligned}
p_{t}=\beta\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+1}}\bigr)^{\sigma}[p_{t+1}+d_{t+1}]\end{aligned}$$

Then, we can get
$$p_{t+1}=\beta\mathbb{E}_{t}\bigl(\frac{c_{t+1}}{c_{t+2}}\bigr)^{\sigma}[p_{t+2}+d_{t+2}]$$
and, insert it to the equation (10), then
$$p_{t}=\beta\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+1}}\bigr)^{\sigma}[d_{t+1}] + \beta^{2}\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+1}}\bigr)^{\sigma}\bigl(\frac{c_{t}}{c_{t+1}}\bigr)^{\sigma}[p_{t+2}]$$
repeat it to infinity, then $$\begin{aligned}
p_{t}&=\beta\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+1}}\bigr)^{\sigma}[d_{t+1}] + \beta^{2}\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+2}}\bigr)^{\sigma}[d_{t+2}]+ \beta^{3}\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+3}}\bigr)^{\sigma}[d_{t+3}] + \cdots
\\ &= \sum_{i=1}^{\infty}\beta^{i}\mathbb{E}_{t}\bigl(\frac{c_{t}}{c_{t+i}}\bigr)^{\sigma}[d_{t+i}]
\\ &= \sum_{i=1}^{\infty}\beta^{i}\mathbb{E}_{t}\bigl(\frac{d_{t}}{d_{t+i}}\bigr)^{\sigma}[d_{t+i}]
\\ &= \sum_{i=1}^{\infty}\beta^{i}\mathbb{E}_{t}\bigl(\frac{\phi(z_{t})}{\phi(z_{t+i})}\bigr)^{\sigma}[\phi(z_{t+i})]\end{aligned}$$
the equation (13) is from the market clearing condition.

Thus, the stock price is the present discounted values of stream of
future dividened payout which contains information on future TFP. That
is, the above-derived equation does well describe the Beaudry and
Portier (2006)'s idea.
