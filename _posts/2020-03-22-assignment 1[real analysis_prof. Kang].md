---
layout: post
title: Real Analysis_assignment 1[_prof. Kang]
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
### assignment 1
* [link for assignment 1](https://drive.google.com/uc?export=view&id=1oDyZRwxD_wA9PrL7YVSp8EOIPFSb4lXp)


Assingment 1 {#assingment-1 .unnumbered}

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

folland.1.1 {#folland.1.1 .unnumbered}
-----------

(a).

Let $E_{1}, E_{2} \in \mathcal{R},$ where $\mathcal{R}$ is a ring. Since
$\mathcal{R}$ is closed under differences,
$$E_{1}\backslash(E_{1}\backslash E_{2})=E_{1} \cap E_{2} \in \mathcal{R}$$

Thus, we can inductively conclude that $\mathcal{R}$ is closed under
finite intersections.

Let's consider $\sigma-$ring, $\mathcal{R}$. And, suppose that
${E_{i}} \in \mathcal{R},\ \text{for}\ i=1,...,\infty.$ Moreover, Let

$A=\bigcup E_{i}$ and $\widetilde{E}_{i} = A\backslash E_{i}$. Then, by
property of $\sigma-$ring,
$\widetilde{E}_{i} \in \mathcal{R},\ \text{forall}\ i,$ which

means that
$F = E_{1}\backslash(\bigcup_{i=2}^{\infty}\widetilde{E}_{i}) \in \mathcal{R}.$
$$F=E_{1} \cap (\bigcap_{i=2}^{\infty}\widetilde{E}_{i}^{c})=E_{1} \cap(\bigcap_{i=2}^{\infty}(A^{c} \cup E{i})=\bigcap_{i=2}^{\infty}((E_{1} \cap A^{c})\cup(E_{1} \cap E{i}))=\bigcap_{i=2}^{\infty}(E_{1} \cap E{i}) = \bigcap_{i=2}^{\infty}E_{i}.$$

Thus, $\mathcal{R}$ is closed under countable intersections.

(b).

If $\mathcal{R}$ is a ring(resp. $\sigma-$ring) and $X \in \mathcal{R},$
then, for all $E \in \mathcal{R}$, $X-E \in \mathcal{R},$ which means

that $\mathcal{R}$ is an algebra(resp. $\sigma-$algebra). On the other
hand, if $\mathcal{R}$ is an algebra(resp.

$\sigma-$algebra), then, by nonemptyness, there exists
$E \in \mathcal{R}$, hence $X = E \cup E^{c} \in \mathcal{R}.$

(c).

Suppose that
$\mathcal{M}=\{E\subset X : E \in \mathcal{R}\ or\ E^{c} \in \mathcal{R}\}$,
which means that $\mathcal{M}$ is closed under

complements.Suppose that $\{E_{n}\}\in \mathcal{M}.$ if $\mathcal{R}$ is
a $\sigma-$ring, then it is closed under

countable intersections and countable unions, so that
$A=\bigcap_{j\geq1,E_{j}^{c}\in \mathcal{R} }E_{j}^{c} \in \mathcal{R}$
and

$B=\bigcup_{j\geq1,E_{j} \in \mathcal{R}}E_{j} \in \mathcal{R}.$ Then,

$$\begin{aligned}
\bigcup_{j\geq1}E_{j} &= \biggl[\bigcup_{j\geq1,E_{j}^{c}E_{j}\in \mathcal{R}}\biggr]\cup\biggl[\bigcup_{j\geq1,E_{j}E_{j} \in \mathcal{R}}\biggr] \\&= \biggl[\biggl[\bigcap_{j\geq1,E_{j}^{c}\in \mathcal{R}}E_{j}^{c}\biggr]\cap\biggl[\bigcup_{j\geq1,E_{j} \in \mathcal{R}}E_{j}\biggr]^{c}\biggr]^{c} \\&= (A-B)^{c}.\end{aligned}$$

Since $A,B \in \mathcal{R}$ and $A-B \in \mathcal{R}$, by definition of
$\mathcal{M}$, $\bigcup_{j\geq1}E_{j} = (A-B)^{c} \in \mathcal{M},$
which

means that $\mathcal{M}$ is also closed under countable unions, hence it
is a $\sigma-$algebra.

(d).

Suppose that
$\mathcal{M}=\{E\subset X : E \cap F\in \mathcal{R}\ \text{for all}\ F \in \mathcal{R}\}$.
Let's suppose $E \in \mathcal{M}$ and

$F \in \mathcal{R}$. Then, $E\cap F \in \mathcal{R}$ and since
$\mathcal{R}$ is closed under differences, $E^{c}\cap F=F\backslash E=$

$F\backslash(E\cap F) \in \mathcal{R}$, thus, $\mathcal{M}$ is closed
under complements. Let $\{E_{n}\} \in \mathcal{M}.$ Then,
$E_{n}\cap F \in$

$\mathcal{R}$, which results in
$\bigcup(E_{n} \cup F) = (\bigcup E_{n})\cup F \in \mathcal{R}$. Thus,
since $(\bigcup E_{n}) \in \mathcal{M}$, $\mathcal{M}$ is also

closed under countable union, hence it is a $\sigma-$algebra.

folland.1.4 {#folland.1.4 .unnumbered}
-----------

$(\Rightarrow)$

It is proved by the definition of $\sigma-$algebra.

$(\Leftarrow)$

Let $\mathcal{R}$ be an algebra and closed under countable increasing
unions. We also know that

any countable unions can be represented as an countable increasing
unions
$$\bigcup_{i=1}^{\infty}E_{i} = \bigcup_{i=1}^{\infty}[\bigcup_{k=1}^{i}E_{k}]$$

Thus, $\mathcal{R}$ is closed under countable unions, hence it is a
$\sigma-$algebra.

folland.1.7 {#folland.1.7 .unnumbered}
-----------

Since $\mu_{1},...,\mu_{n}$ are measures on $(\mathcal{X},\mathcal{M})$,
$\mu_{i}(\emptyset)=0\ \forall \ i = 1,...n,$ and therefore,
$\sum_{j=1}^{n}a_{j}\mu_{j}(\emptyset)=0.$ Next, suppose that
$\{E_{j}\}_{1}^{\infty} \in \mathcal{M}$ and $\{E_{j}\}_{1}^{\infty}$
are pariwise disjoint, then, $$\begin{aligned}
\biggl(\sum_{i=1}^{n}a_{i}\mu_{i}\bigr)\bigl(\bigcup_{i=1}^{\infty}E_{i}\bigr) = \sum_{i=1}^{n}a_{i}\mu_{i}\bigl(\bigcup_{i=1}^{\infty}E_{i}\bigr) = \sum_{i=1}^{n}a_{i}\bigl(\sum_{i=1}^{\infty}\mu_{i}(E_{i})\bigr) \\
= \sum_{i=1}^{n}\sum_{i=1}^{\infty}a_{i}\mu_{i}(E_{i}) = \sum_{i=1}^{\infty}\sum_{i=1}^{n}a_{i}\mu_{i}(E_{i})=\sum_{i=1}^{\infty}\bigl(\sum_{i=1}^{n}a_{i}\mu_{i}\bigr)(E_{i})\end{aligned}$$
Thus, $\sum_{i=1}^{n}a_{i}\mu_{i}$ is countably additive, which means
that $\sum_{i=1}^{n}a_{i}\mu_{i}$ is a measure on
$(\mathcal{X},\mathcal{M})$ as well.

folland.1.8 {#folland.1.8 .unnumbered}
-----------

(i).

By definition,
$\liminf E_{j} = \bigcup_{i=1}^{\infty}\bigcap_{j=i}^{\infty}E_{j}$.
Let's define an increasing sequence of sets in

$\mathcal{M}$, $A_{i}=\bigcap_{j=i}^{\infty}E_{j}$. Then,
$$\mu(\liminf E_{j}) = \mu(\bigcup_{i=1}^{\infty}\bigcap_{j=i}^{\infty}E_{j}) = \mu(\bigcup_{i=1}^{\infty}A_{i}) = \lim_{i\rightarrow\infty}\mu(A_{i}) \ (\text{Continuity from below})$$

By definition of $A_{i}$, $A_{i} \subset E_{j},\ \text{for}\ j \geq i,$
so $A_{i} \subset \liminf E_{j}.$ It is as follows:
$$\mu(\liminf E_{j}) = \lim_{i\rightarrow\infty}\mu(A_{i}) \leq \liminf\mu(E_{j})$$

(ii).

By definition,
$\limsup E_{j} = \bigcap_{i=1}^{\infty}\bigcup_{j=i}^{\infty}E_{j}$.
Suppose that $\mu\bigl(\bigcup_{j=i}^{\infty}E_{j}\bigr)<\infty.$ Let's
define

an decreasing sequence of sets in $\mathcal{M}$,
$A_{i}=\bigcup_{j=i}^{\infty}E_{j}$. Then,
$$\mu(\limsup E_{j}) = \mu(\bigcap_{i=1}^{\infty}\bigcup_{j=i}^{\infty}E_{j}) = \mu(\bigcap_{i=1}^{\infty}A_{i}) = \lim_{i\rightarrow\infty}\mu(A_{i}) \ (\text{Continuity from above})$$

By definition of $A_{i}$, $E_{j} \subset A_{i},\ \text{for}\ j \geq i,$
so $\limsup E_{j} \leq \mu(A_{i}),\ \text{for all}\ i.$ It is as
follows:
$$\limsup E_{j} \leq \lim_{i\rightarrow\infty}\mu(A_{i}) = \mu(\liminf\mu(E_{j}))$$

folland.1.10 {#folland.1.10 .unnumbered}
------------

(i).

$\mu_{E}(\emptyset) = \mu(\emptyset \cap E) = \mu(\emptyset) = 0.$

(ii).

Let, $\{A_{i}\}_{1}^{\infty} \subset \mathcal{M}$ and
$\{A_{i}\}_{1}^{\infty}$ disjoint. Then:
$$\mu_{E}\bigl(\bigcup_{i=1}^{\infty}A_{i}\bigr) = \mu\bigl(E \cap \bigl(\bigcup_{i=1}^{\infty}A_{i}\bigr)\bigr) = \mu\bigl(\bigl(\bigcup_{i=1}^{\infty}E \cap A_{i}\bigr)\bigr)  \stackrel{!}{=} \sum_{i=1}^{\infty}\mu(E \cap A_{i}) = \sum_{i=1}^{\infty}\mu_{E}(A_{i})$$

The reason why $\stackrel{!}{=}$ is established is that, if
$\{A_{i}\}_{1}^{\infty}$ is a disjoint family of sets, then

$\{A_{i} \cap E \}_{1}^{\infty}$ will be disjoint as well. Thus, we have
shown $\mu_{E}$ is a meaure.

folland.1.12 {#folland.1.12 .unnumbered}
------------

(a).

If $E,F \in \mathcal{M}$ and $\mu(E \bigtriangleup F)$, then, since
$(E\backslash F)$ and $(F \backslash E)$ are disjoint,
$$0 = mu(E \bigtriangleup F) = \mu((E\backslash F) \cup (F \backslash E)) = \mu(E\backslash F) + \mu(F \backslash E)$$

Thus, since $\mu \geq 0$,
$\mu(E\backslash F) = \mu(F \backslash E) = 0.$

We also know that $E = ((E \cap F)\cup(E\backslash F))$ and
$F = ((F \cap E)\cup(F\backslash E)).$ Then, $$\begin{aligned}
\mu(E) = \mu(E \cap F) + \mu(E\backslash F) = \mu(E \cap F) \\
\mu(F) = \mu(F \cap E) + \mu(F\backslash E) = \mu(F \cap E)\end{aligned}$$

Thus, $\mu(E \cap F) = \mu(E) = \mu(F).$

(b).

Reflexivity: $E \sim E$ since $\mu(E \bigtriangleup E) = 0$.

Symmetricity: $E \sim F \Leftrightarrow F \sim E$ since
$\mu(E \bigtriangleup F) = \mu(F \bigtriangleup E)$.

Transitivity: note that
$(E \bigtriangleup F) \bigtriangleup (F \bigtriangleup G) = E \bigtriangleup G$.
So if $\mu(E\bigtriangleup F) = 0 = mu(F \bigtriangleup G)$,

then, since $A \bigtriangleup B \subset A\cup B$, $$\begin{aligned}
\mu(E\bigtriangleup G) = \mu((E \bigtriangleup F) \bigtriangleup (F \bigtriangleup G)) &\leq \mu((E \bigtriangleup F) \cup (F \bigtriangleup G)) \\&
\leq \mu(E\bigtriangleup F) + \mu(F \bigtriangleup G) = 0\end{aligned}$$

(c).

Let $\rho(E,F) = \mu(E \bigtriangleup F).$ Then, $$\begin{aligned}
\rho(E,G) = \mu(E\bigtriangleup G) &= \mu((E \bigtriangleup F) \bigtriangleup (F \bigtriangleup G)) \\&\leq \mu((E \bigtriangleup F) \cup (F \bigtriangleup G)) \\&
\leq \mu(E\bigtriangleup F) + \mu(F \bigtriangleup G) = \rho(E,F)+\rho(F,G)\end{aligned}$$

folland.1.14 {#folland.1.14 .unnumbered}
------------

Let $\mu$ be a semifinite measure on a $\sigma-$algebra $\mathcal{M}$,
and $\mu(E)=\infty$. Let
$M = \sup\{\mu(F): F \subset E, \mu(F) \leq \infty \}$, and suppose
$M \leq \infty$. Then, for each $\epsilon \geq 0$, there exists a set
$F_{\epsilon} \subset E$ such that
$M-\epsilon < \mu(F_{\epsilon}) \leq M.$ Consider the sequence
$\{F_{1/n}\}_{n=1}^{\infty}$, and let
$G_{n} = \bigcup_{m=1}^{n}F_{1/m} \in \mathcal{M}$, which forms a
sequence of increasing sets in $E$. Thus,
$M-\frac{1}{n} \leq \mu(F_{1/n}) \leq \mu(G_{n}) \leq M.$ Let
$G = \bigcup_{n=1}^{\infty}G_{n},$ then $G_{n } \in \mathcal{M}$ and
$\mu(G) =M.$

Now consider the set $E\backslash G$, which has infinite measure since
$E$ has infinite measure and $G$ has finite measure. Then since $\mu$ is
semifinite, there exists a set $H\subset E\backslash G$, which have
positive finite measure, $0 < \mu(H) < \infty$. So
$\mu(G\cup H) = \mu(G) + \mu(H) > M$, which contradicts the definition
of M. Thus, $\sup\{\mu(F): F \subset E, \mu(F) \leq \infty \} = \infty$,
which means that for any $C \geq 0$ there exists an $F \subset E$ such
that $C<\mu(F)<\infty$.

folland.1.15 {#folland.1.15 .unnumbered}
------------

(a).

Since, for any
$E \in \mathcal{M},\ \emptyset \subset E\ \text{and} \ \mu(\emptyset) =0,$
we have that $0 \in \{\mu(F) : F \subset E\ \text{and}\ $

$\mu(F) < \infty\}.$ Thus,
$$\mu_{0}(E)  = \sup\{\mu(F) : F \subset E\ \text{and}\ \mu(F) < \infty\} \geq 0$$

And,
$$\mu_{0}(\emptyset)  = \sup\{\mu(F) : F \subset \emptyset\ \text{and}\ \mu(F) < \infty\} =\sup\{0\} =  0$$

Let's Assume that $E_{1}, E_{2},\cdot \cdot \cdot$ is a sequence of
pariwise disjoint sets in $\mathcal{M}$ and $E =$

$\bigcup_{i=\mathcal{N}}E_{i}.$ For any set $F \in \mathcal{M}$ such
that $F \subset E$ and $\mu(F) < +\infty$, we have that

$\{F \cap E_{i}\}_{i\in \mathcal{N}}$ is a family of disjoint sets in
$\mathcal{M}$ and
$$F = F \cap E = \bigcup_{i\in \mathcal{N}}(F \cap E_{i})$$

Then, $$\begin{aligned}
\mu_{0}\bigl(\bigcup_{n=1}E_{n}\bigr) &= \sup\{\mu(E) : E \subset \bigcup_{n=1}E_{n},\mu(E) < \infty\}
\\& =\sup\{\sum_{n=1}^{\infty}\mu(E\cap E_{n}) : E \subset \bigcup_{n=1}E_{n},\sum_{n=1}^{\infty}\mu(E\cap E_{n}) < \infty\} \\& = \sup\{\sum_{n=1}^{\infty}\mu(F_{n}) : F_{n} \subset E_{n},\sum_{n=1}^{\infty}\mu(F_{n}) < \infty\} \\&=  \sum_{n=1}^{\infty}\sup\{\mu(F_{n}) : F_{n} \subset E_{n},\mu(F_{n}) < \infty\} = \sum_{n=1}^{\infty}\mu_{0}(E_{n})\end{aligned}$$

Therefore, $\mu_{0}$ is a measure on $(X, \mathcal{M})$. $\mu_{0}$ is
clearly semifinite since if $\mu_{0} = \infty$, then

$\sup\{\mu(F) : F \subset E,\mu(F) < \infty\}$, in particular there
exists $F\subset E$ such that $0<\mu(F)<$

$\infty,$ and since $\mu(F)$ is finite, we have $\mu(F)=\mu_{0}(F)$.

(b).

Suppose $\mu$ is semidefinite. if $\mu(E) = \infty$, then by (14), for
any $C>0$ there is a set

$F\subset E$ with $C<\mu(F)<\infty$, so $\mu_{0}(E) = \infty = \mu(E).$
Conversely, If $\mu_{0}(E)= \infty$, then

clearly $\mu(E) = \infty$. This is because, for any $C>0$, there is an
$F$ such that $F\subset E$ with

$C<\mu(F)\leq\mu(E).$

(c).

I have no idea.

folland.1.17 {#folland.1.17 .unnumbered}
------------

$(\Rightarrow)$

Since $\mu^{*}$ is subadditive, it follows that,
$$\mu^{*}\bigl(E\cap\bigl(\bigcup_{j=1}^{\infty}A_{j}\bigr)\bigr) = \mu^{*}\bigl(\bigcup_{j=1}^{\infty}E\cap A_{j}\bigr) \leq \sum_{j=1}^{\infty}\mu^{*}(E\cap A_{j})$$

$(\Leftarrow)$

Let $B_{n}=\bigcup_{j=1}^{\infty}A^{j}$. Then the $B_{n}$ form an
increasing sequence of $\mu^{*}-$measurable sets.

Let $B=\bigcup_{n=1}^{\infty}B^{n}$. Then $$\begin{aligned}
\mu^{*}(E \cap B_{n}) &= \mu^{*}(E \cap B_{n} \cap A_{n}) +\mu^{*}(E \cap B_{n}\cap A_{n}^{c})
\\& = \mu^{*}(E \cap A_{n}) +\mu^{*}(E \cap B_{n-1})\end{aligned}$$

So, by induction
$\mu^{*}(E \cap B_{n}) = \sum_{j=1}^{n}\mu^{*}(E \cap A_{j}).$ It
follows $$\begin{aligned}
\mu^{*}(E) &= \mu^{*}(E \cap B_{n}) +\mu^{*}(E \cap B_{n}^{c})
\\& \geq  \sum_{j=1}^{n}\mu^{*}(E \cap A_{j}) +\mu^{*}(E \cap B_{n}^{c})\end{aligned}$$

Then, taking the limit as $n\longrightarrow\infty$, we have
$$\mu^{*}(E)\geq  \sum_{j=1}^{\infty}\mu^{*}(E \cap A_{j}) +\mu^{*}(E \cap B_{n}^{c})$$

So,
$$\mu^{*}(E)-\mu^{*}(E \cap B_{n}^{c})\geq  \sum_{j=1}^{\infty}\mu^{*}(E \cap A_{j})$$

But $\mu^{*}(E)-\mu^{*}(E \cap B_{n}^{c})=\mu^{*}(E\cap B)$ since $B$ is
$\mu^{*}-$measurable set. Hence, it follows
$$\mu^{*}\bigl(E\cap\bigl(\bigcup_{j=1}^{\infty}A_{j}\bigr)\bigr) = \mu^{*}(E \cap B)\geq  \sum_{j=1}^{\infty}\mu^{*}(E \cap A_{j})$$

Conclusively, we can get
$\mu^{*}\bigl(E\cap\bigl(\bigcup_{j=1}^{\infty}A_{j}\bigr)\bigr)=\sum_{j=1}^{\infty}\mu^{*}(E \cap A_{j}).$

folland.1.18 {#folland.1.18 .unnumbered}
------------

(a).

Let $E \subset X$. Then,
$$\mu^{*}(E) = \inf\{\sum_{j=1}^{\infty}\mu_{0}(A_{j}):A_{j} \in \mathbb{A}, E \subset \bigcup_{j}^{\infty}A_{j}\}$$

So, for any $\epsilon > 0$, There exists
$E\subset \sum_{j=1}^{\infty}A_{j}$ for sets $A_{j} \in \mathbb{A},$ and
$$\sum_{j=1}^{\infty}\mu_{0}(A_{j}) \leq \mu^{*}(E) + \epsilon$$

Then letting $A = \sum_{j=1}^{\infty}A_{j} \in \mathbb{A}_{\rho}$, we
have
$$\mu^{*}(A) \leq \sum_{j=1}^{\infty}\mu^{*}(A_{j}) \leq \sum_{j=1}^{\infty}\mu_{0}(A_{j}) \leq \mu^{*}(E) + \epsilon$$

(b).

Suppose $\mu^{*}(E) < \infty$, and if $E$ is $\mu^{*}-$measurable, we
have, for each $n$, $E \subset A_{n},\ A_{n} \in \mathbb{A}_{\rho}$

with $$\mu^{*}(E) \leq \mu^{*}(A_{n}) \leq \mu^{*}(E) + \frac{1}{n}$$

Letting $B=\bigcap_{n=1}^{\infty}A_{n}$, we have
$B\in \mathbb{A}_{\rho\delta},\ E \subset B$ and
$$\mu^{*}(E) \leq \mu^{*}(B) \leq \mu^{*}(E)$$

Thus, using the measurability of E, we have
$$\mu^{*}(B) = \mu^{*}(E) = \mu^{*}(B\cap E) +\mu^{*}(B \cap E^{c}) = \mu^{*}(E) + \mu^{*}(B\backslash E)$$

Hence, we conclude that $\mu^{*}(B\backslash E)=0.$

To show the converse, assume we have an set
$B \in \mathbb{A}_{\rho\delta}$ with $E \subset B$, and
$\mu^{*}(B\backslash E)=0.$

If $F \subset X$, then $$\begin{aligned}
\mu^{*}(F\cap E) +\mu^{*}(F \cap E^{c}) &\leq \mu^{*}(F\cap B) +\mu^{*}(F \cap B^{c})+\mu^{*}(B\backslash E)=0
\\& \leq \mu^{*}(F\cap B) +\mu^{*}(F \cap B^{c}) = \mu^{*}(F)\end{aligned}$$

since $B$ is $\mu^{*}-$measurable. Clearly
$\mu^{*}(F) \leq \mu^{*}(F\cap E) +\mu^{*}(F \cap E^{c})$, so $E$ is
$\mu^{*}-$measurable.

(c).

Suppose $\mu_{0}$ is $\sigma-$finite. Then there exist
$B_{j} \in \mathbb{A}$ with $\mu_{0}(B_{j})<\infty$ and
$\bigcup_{j=1}^{\infty}B_{j}=X.$

By part (a), we have that for each $E\cap B_{j}$, there exist an set
$A_{jn} \in \mathbb{A}_{\rho}$ with $E\cap B_{j} \subset A_{jn}$

and $$\mu^{*}(A_{jn})\leq \mu^{*}(E\cap B_{j}) +\frac{2^{-j}}{n}$$

Let $A_{n} = \bigcup_{j=1}^{\infty}A_{jn}$. Then
$A_{n} \in \mathbb{A}_{\rho},\ E\subset A_{n}$ and
$$\mu^{*}(A_{j})\leq \mu^{*}(E) +\frac{1}{n}$$

Then letting $B=\bigcup_{n=1}^{\infty}A_{n}$, we have that
$B \in \mathbb{A}_{\rho\delta},\ E\subset B$ and
$\mu^{*}(E)\leq \mu^{*}(B).$

folland.1.19 {#folland.1.19 .unnumbered}
------------

Suppose $E$ is $\mu^{*}-$measurable, then
$$\mu^{*}(X) = \mu^{*}(X\cap E) + \mu^{*}(X\cap E^{*})$$

But $\mu^{*}(X) = \mu_{0}(X)$ and $\mu^{*}(X\cap E) = \mu^{*}(E)$.
Hence, we have $$\mu^{*}(E) = \mu_{0}(X)-\mu^{*}(E^{*}) = \mu_{*}(E)$$

Conversly, suppose $\mu^{*}(E) = \mu_{*}(E).$ By problem 18.a, there
exists $A_{n} \in\mathbb{A}_{\rho}$ with $E^{c} \subset A$

and $\mu^{*}(A) \leq \mu^{*}(E^{c}) + \epsilon$ for any $\epsilon > 0$.
Consider $A^{c}$. We have $$\begin{aligned}
\mu^{*}(A^{c}) &\geq \mu^{*}(X) -\mu^{*}(A) = \mu_{0}(X) - \mu^{*}(A) 
\\& \geq \mu_{0}(X) - (\mu^{*}(E^{c})+\epsilon) = \mu_{0}(X)-(\mu_{0}(X) - \mu^{*}(E)+\epsilon) = \mu^{*}(E)-\epsilon\end{aligned}$$

Taking $B$ as the collection of all intersections of such $A$, we have
$B \subset E,\ B \in \mathbb{A}_{\rho\delta}$

and, since
$\mu^{*}(E)-\mu^{*}(B) \leq \mu^{*}(E)-\mu^{*}(A^{c}) \leq \epsilon$,
$\mu^{*}(E\backslash B).$ Thus, we have that $E$ is

$\mu^{*}-$measurable.
