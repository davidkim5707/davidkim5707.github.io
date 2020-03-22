---
layout: post
title: assignment 1[Real Analysis_prof. Kang]
categories:
  - Study_econ
tags:
  - Mathmatics
  - real analysis
  - measure
  - Rudin
  - Folland
last_modified_at: 2020-03-22
use_math: true
---

<p>\documentclass[11pt]{article}<br />
\usepackage{fullpage} \usepackage{amsmath, amssymb, amsthm,graphicx} \usepackage{geometry}<br />
\geometry{letterpaper}</p>
<p>%Lines starting with a % are comments and won't show up in your document.</p>
<p>\title{\textbf{Real Analysis assignment 1}} %Your title goes here! \author{Kim Dawis(2019311156)} %You put your name here. \date{2020-03-28} %to get rid of the date just delete the % in front of date{}. You can also fill it in with a different date if you like. For example, if you want to make it seem like you did the assignment ages ago you could write in a date from last week instead of the due date. Not that I've ever done such a thing myself ...</p>
<p>\begin{document} \maketitle</p>
<p>\section*{Assingment 1}</p>
<p>\subsection*{rudin.1.1}</p>
<p>(i). prove that $\phi$ is additive.</p>
<p>\vspace{.1in} First, let's define any elementary set A be a finite disjoint union of intervals. Then, $\phi(A)$ would be a sum of the lengths of those disjoint intervals if any intervals includes 0 as an endpoint, and 1 larger than the sum of the lengths if one interval has 0 as an its endpoint.</p>
<p>Let's consider 2 disjoint elementary sets A and B. Then, $\phi(A\cup B)$ will be the sum of the lengths of the intervals in $A\cup B$ if no elementary set has intervals which contain 0 as its endpoint, And 1 larger than the sum if one of them contains an interval with 0 as endpoint.</p>
<p>Thus, in either case, $\phi(A\cup B) = \phi(A) + \phi(B)$ when $A\cap B = \emptyset$.</p>
<p>\vspace{.1in} \noindent (ii). prove that $\phi$ is not regular.</p>
<p>\vspace{.1in} Let's consider a subset A of (0,a] where $0&lt;a\leq 1$. Then, there's no closed subset, B, of A in which $\phi(A) &lt; \phi(B) + \epsilon$, since it is always that $\phi(A) = a+1$ and $\phi(B) &lt; 1$.</p>
<p>\vspace{.1in} \noindent (ii). prove that $\phi$ is not countable additive set function on a $\sigma-$ring.</p>
<p>\vspace{.1in}</p>
<p>\begin{thebibliography}{999} \bibitem{ParkChoonsung:2017}{Park, Choonsung(2017). Consumption, Reservation Wages, and Aggregate Labor Supply. Working Paper.} \end{thebibliography}</p>
<p>\end{document}</p>