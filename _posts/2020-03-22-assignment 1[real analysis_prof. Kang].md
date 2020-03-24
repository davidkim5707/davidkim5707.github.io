---
layout: post
title: assignment 1[Real Analysis_prof. Kang]
categories:
  - Study_econ&math
tags:
  - Mathmatics
  - real analysis
  - measure
  - Rudin
  - Folland
last_modified_at: 2020-03-22
use_math: true
---

* [link for pop quiz 1](https://drive.google.com/uc?export=view&id=1vKRFKhy4KXQDDGWqZFR0CueiP5cg2cAD)
---
author:
- Kim Dawis(2019311156)
date: '2020-03-28'
title: '**Real Analysis assignment 1**'
---

Assingment 1 {#assingment-1 .unnumbered}
============

rudin.11.15 {#rudin.11.15 .unnumbered}
-----------

(i). prove that $\phi$ is additive.

First, let's define any elementary set A be a finite disjoint union of
intervals. Then, $\phi(A)$ would be a sum of the lengths of those
disjoint intervals if any intervals includes 0 as an endpoint, and 1
larger than the sum of the lengths if one interval has 0 as an its
endpoint.

Let's consider 2 disjoint elementary sets A and B. Then, $\phi(A\cup B)$
will be the sum of the lengths of the intervals in $A\cup B$ if no
elementary set has intervals which contain 0 as its endpoint, And 1
larger than the sum if one of them contains an interval with 0 as
endpoint.

Thus, in either case, $\phi(A\cup B) = \phi(A) + \phi(B)$ when
$A\cap B = \emptyset$.

(ii). prove that $\phi$ is not regular.

Let's consider a subset A of (0,a\] where $0<a\leq 1$. Then, there's no
closed subset, B, of A in which $\phi(A) < \phi(B) + \epsilon$, since it
is always that $\phi(A) = a+1$ and $\phi(B) < 1$.

(ii). prove that $\phi$ is not countable additive set function on a
$\sigma-$ring.

Let's consider the following case.

$$(0,\frac{1}{k}] = \bigcup_{n=1}^{\infty} ({\frac{1}{k^{n+1}}, \frac{1}{k^{n}}}],\ \text{where}\ k>1$$

Then,

$$\phi((0,\frac{1}{k}]) = 1+\frac{1}{k},\ \text{which is not equal to}\ \phi(\bigcup_{n=1}^{\infty} ({\frac{1}{k^{n+1}}, \frac{1}{k^{n}}}]),\ \text{since}\ \phi(\bigcup_{n=1}^{\infty} ({\frac{1}{k^{n+1}}, \frac{1}{k^{n}}}]) = \frac{1}{k}.$$

rudin.11.16 {#rudin.11.16 .unnumbered}
-----------

If $\int_{A}fdu = 0$ for every measurable subset A of a measurable set
E, then $f(x)=0,\ \text{almost everywhere on E.}$

If $f\geq 0$ and $\int_{A}fdu = 0$, then $f(x)=0$ almost everywhere on
A.

By the dominated convergence theorem,
$$\int_{A}[(f(x))^2-\frac{1}{2}]dx = 0,\ \text{for all A}.$$ Thus, by
theorem 1,
$$f(x)=\pm\frac{1}{\sqrt{2}}\ \text{almost everywhere on E.}$$ Let A be
the set of points of E where $$f(x)=\frac{1}{2}.$$ Then, by theorem 2,
$f(x)=0$ almost everywhere on A. Since the function $f(x)\neq 0$, it
follows that A has measure 0. Similarly, the set where
$f(x)=-\frac{1}{2}$ has also measure 0.

So, we showed that $$m(A)=0.$$

rudin.11.17 {#rudin.11.17 .unnumbered}
-----------

Assume that n satisfies $$sin\ nx \geq \delta, x \in, \delta >0.$$ It
follows that
$$\int_{E}sin\ nxdx \geq \int_{E}\delta dx = \delta\int_{E}dx,$$ which
leads to $$\int_{E}sin\ nxdx \geq \delta \mu(E).$$ Then, by the Bassel
inequality, the inequality, $$\int_{E}sin\ nxdx \geq \delta \mu(E).$$
stands only for finite number of integers $n$ because the Fourier
coefficient of the function $\mathcal{X}_{E}$ contains the integral
$\int_{E}sin\ nxdx.$

Thus, it is proved that there are at most finitely many integers $n$
such that $$sin nx \geq \delta,\ \text{for all} x \in E.$$
