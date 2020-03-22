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

rudin.1.1 {#rudin.1.1 .unnumbered}
---------

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