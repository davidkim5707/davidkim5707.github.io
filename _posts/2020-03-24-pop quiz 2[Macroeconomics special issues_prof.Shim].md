---
layout: post
title: Macroeconomics special issues_Pop quiz 2[_prof. Shim]
categories:
  - Study_econ&math
tags:
  - economics
  - AR(1) process
  - MA process
  - RBC
last_modified_at: 2020-04-02
use_math: true
---

### pop quiz 2

* [link for pop quiz 2](https://drive.google.com/uc?export=view&id=1IR697e9sqZUmVyqXvunNW1T2IvPWbBxI)  

Prove Lemma1 

(Key Moments of AR(1) process). For an AR(1) process with $|\phi|<1,$
the followings holds:
$$\mu_{t} = \frac{c}{1-\phi},\ \gamma_{0t}=\frac{\sigma^{2}}{1-\phi^{2}},\ \gamma_{jt}=\frac{\phi^{j}\sigma^{2}}{1-\phi^{2}}$$
Hence, moments (upto the second order) do not depend on t.

[(proof)]{.nodecor} 


We can transform the AR(1) process as follows by using lag term, L.

$$\begin{aligned}
 y_{t} &= c + \phi y_{t-1} + \epsilon_{t} \\&= c + \phi Ly_{t} + \epsilon_{t}  \end{aligned}$$

So, $$\begin{aligned}
(1-\phi L)y_{t} &= c +\epsilon_{t} \end{aligned}$$

and, by assuming that $1-\phi L \neq 0$, $$\begin{aligned}
y_{t} &= \frac{(c +\epsilon_{t})}{1-\phi L} 
\\& = (c +\epsilon_{t})(1 + \phi L + \phi^{2} L^{2} + \cdots)
\\& = \frac{c}{1-\phi} + \epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots)\end{aligned}$$

Thus, since $\epsilon_{t}$ follows **w.n.** process,
$\mu_{t} = E[X_{t}] = \frac{c}{1-\phi}$.

As we obtained above, we can rewrite AR(1) process as follows:
$$y_{t} = \frac{c}{1-\phi} + \epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots)$$

Then,
$$VAR(y_{t}) =VAR(\epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots))$$

then, the following holds: $$\begin{aligned}
VAR(y_{t}) &=VAR(\epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots))
\\&= VAR(\epsilon_{t} + \phi \epsilon_{t-1} + \phi^{2}\epsilon_{t-2} + \cdots))
\\&= \sigma^{2}(1 + \phi  + \phi^{2} + \cdots)
\\&= \frac{\sigma^{2}}{1-\phi^{2}}\end{aligned}$$

the equation (3) comes from the fact that
$E(\epsilon_{t}\epsilon_{t-1}) = 0.$ Thus,
$\gamma_{0t}=\frac{\sigma^{2}}{1-\phi^{2}}.$

We know that,
$$y_{t} = \frac{c}{1-\phi} + \epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots)$$
$$y_{t-j} = \frac{c}{1-\phi} + \epsilon_{t-j}(1 + \phi L + \phi^{2} L^{2} + \cdots)$$

Then, $$\begin{aligned}
y_{t}y_{t-1} = \bigl(\frac{c}{1-\phi}\bigr)^{2} + \frac{c}{1-\phi}\bigl(\epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots) + \epsilon_{t-j}(1 + \phi L + \phi^{2} L^{2} + \cdots)\bigr)\\ + \epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots)\cdot\epsilon_{t-j}(1 + \phi L + \phi^{2} L^{2} + \cdots)\end{aligned}$$

Hence, $$\begin{aligned}
E(y_{t}y_{t-j}) = \bigl(\frac{c}{1-\phi}\bigr)^{2}+ E\bigl(\epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots)\cdot\epsilon_{t-j}(1 + \phi L + \phi^{2} L^{2} + \cdots)\bigr)\end{aligned}$$

and, since, $$\begin{aligned}
E\bigl(\epsilon_{t}(1 + \phi L + \phi^{2} L^{2} + \cdots)\cdot\epsilon_{t-j}(1 + \phi L + \phi^{2} L^{2} + \cdots)\bigr) = \sigma^{2}(\phi^{j}+\phi^{j+2} + \cdots)\end{aligned}$$

the covariance of $y_{t},\ y_{t-j}$ is as follows: $$\begin{aligned}
COV(y_{t},y_{t-j}) &= E(y_{t}y_{t-j}) - E(y_{t})E(y_{t-j}) \\&= \bigl(\frac{c}{1-\phi}\bigr)^{2}+ \sigma^{2}(\phi^{j}+\phi^{j+2} + \cdots) - \bigl(\frac{c}{1-\phi}\bigr)^{2} \\& = \sigma^{2}(\phi^{j}+\phi^{j+2} + \cdots) \\& = \frac{\phi^{j}\sigma^{2}}{1-\phi^{2}}\end{aligned}$$
  
