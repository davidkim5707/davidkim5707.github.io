---
layout: post
title: assignment 4[Adavanced Macro_prof. Jang]
categories:
  - Study_econ
tags:
  - economic
  - Indivisible labor
  - employment lottery
  - Aiyagari, Christiano and Eichenbaum(1992)
last_modified_at: 2020-03-19
use_math: true
---

### assignment 4

* [link for assignment 4](https://drive.google.com/uc?export=view&id=1mAecmq-vcY0J6DD4V92Qm-mjDE8y1A9z)  

# Advanced Macroeconomics 4

### Kim Dawis(2019-92073)

## 1 Indivisible labor with employment lottery

### 1.1 Set up the dynamic programming problem of the household.

```
max
ct,lt
```
#### ∑T

```
t=
```
```
βt[u(ct) +v(lt)]
```
```
s.t. kt+1= (1−δ)kt+rtkt+wtht
```
I denote the consumption, investment, and the next period capital of working individuals
asc 1 ,i 1 ,k′ 1. Additionally, I denote the same for non-working individuals asc 2 ,i 2 ,k′ 2

V(k,x) = max
α,b,k′ 1 ,k′ 2 ,c 1 ,c 2

```
(α[u(c 1 )+v(1− ̄h)+βE[V(k 1 ′,z′)]]+(1−α)[u(c 2 )+v(1)+βE[V(k 2 ′,z′)]])
```
s.t.
c 1 +i 1 +p(α)b=w(z,k) ̄h+r(z,k)k 1
c 2 +i 2 +p(α)b=b+r(z,k)k 2
k′ 1 = (1−δ)k+i 1
k′ 2 = (1−δ)k+i 2

for insurance company,
Π =p(α)b−(1−α)b

### 1.2 Write down the first order necessary conditions.

The above value function can be rewritten using the household constraints and the zero
profit condition for the insurance company.

V(k,x) = max
α,b,k′ 1 ,k′ 2 ,c 1 ,c 2

```
(α[u(w(z,k) ̄h+r(z,k)k 1 −k 1 ′+ (1−δ)k−(1−α)b) +v(1−h ̄) +βE[V(k′ 1 ,z′)]]
```
```
+(1−α)[u(b+r(z,k)k 2 −k′ 2 −(1−δ)k−(1−α)b) +v(1) +βE[V(k 2 ′,z′)]])
```
The FOCs are
[b] :αu′(c 1 )(1−α) = (1−α)u′(c 2 )α


```
[k′ 1 ] :u′(c 1 ) =βE[Vk(k′ 1 ,z′)]
[k′ 2 ] :u′(c 2 ) =βE[Vk(k′ 2 ,z′)]
```
### 1.3 Household consumtion

From the FOC of [b] and the strict concavity of the utility function,c 1 =c 2

### 1.4 Household investment

From the FOC of [k′ 1 ] and [k′ 2 ] and the result of above,k′ 1 =k′ 2. This, in turn, implies that
i′ 1 =i′ 2 from the household constraints.

### 1.5 Elasticity of aggregate labor supply

```
u(ct,lt) =α(u(ct)) +v(1− ̄h) + (1−α)(u(ct) +v(1))
=u(ct) +αv(1− ̄h) + (1−α)v(1)
```
Let’s normalize thatv(1) = 0 and defineB=

```
−v(1− ̄h)
h
```
#### .

```
u(ct,lt) =u(ct)−Bα ̄h
```
```
u(ct)−BNt
```
This linear utility function in regards to the total labor hoursNtimplies that the infinitely
elastic aggregate labor supply curve.


## 2 Model in Aiyagari, Christiano and Eichenbaum

(prep)

```
maxE 0
```
#### ∑∞

```
t=
```
```
βtu(ct,nt)
```
s.t
ct≤yt−(gt+kt+1)
0 ≤nt≤N,ct≥ 0 ,kt+1≥ 0 ,k 0 ≥ 0
yt=f(kt,nt)
gt=G(gTt +gPt)
Ψ(gp′|gp) =prob(gpt+1≤gp|gtp=gp) and Ψ is decreasing ingp.

Then the Bellman equation is

```
V(k,gP,gT) = max
c,k′,n∈A(k,g)
```
```
[u(c,n) +E[βV(k′,gP′,gT′|gP)]]
```
```
A(k,g) =c,k′,n;c≥ 0 , 0 ≤n≤N,k′≥k,c+k′+g≤f(k,n)
```
We can rearrange the Bellman equation.

```
V(k,gP,gT) = max
k≤k′≤f(k,n)−g
```
#### [

```
maxc,k′,n∈A(k,g)
n,cp∈B(k,k′+g)
```
#### ]

```
+E[βV(k′,gP′,gT′)|gP]
```
Where,
B(k,k′+t) =cp,n: 0≤n≤N, 0 ≤cP≤f(k,n)−(g+k′)

Then, the static part of this problem is choosingcandngivenk′.

```
W(k,k′+g) = max
n,cp∈B(k,k′+g)
```
```
u(cp,n)
```
Here, we can get the solution for the labor supply and consumption.

```
n=h(k,k′+g)
```
```
c=q(k,k′+g)
```
The dynamic part of the problem is choosingk′.

```
V(k,kP,kT) = max
k≤k′≤f(k,n)−g
```
```
W(k,k′+g) +E[βV(k′,gP′,gT′)|gP]
```
Here, we can getk′.


### 2.1 Respondences to an increase in government consumption

- Investment

```
Differentiate the both sides of the dynamic problems byk′
```
```
−Wk′(k,k′+g) =E[βVk′(k′,gP′,gT′)|gP]
```
The LHS (=uc(c,n)) represents the marginal cost of the investment while RHS
(βuc(c′,n′)fk′(k′,n′)) represents the marginal benefit of it. Marginal cost function is up-
ward sloping because higher investment lowers the current consumption, increasing the
marginal utility of the current consumption (which is again the marginal cost of the in-
vestment). Marginal benefit function is downward sloping because more capital lowers the
marginal utility of current consumption and the next period productivity of capital(which
are again the marginal benefit of the investment when multiplied by each other).

Since the marginal cost is a function of g, both of the temporary and persistent gov-
erment spending shock moves the marginal cost function upward. The upward movement


is due to the increased marginal utility of current consumption(caused by lower current
consumption followed by higher government spending).

Since the marginal benfit is a function ofgP, only the persistent government spending
shock moves the marginal benefit function upward. The upward movement is due to the
higher marginal utility of consumption for the next period. The lower consumption next
period is caused because the household is aware of the persistent government spending for
the next period.

```
According to the graph above, we can see the result;k′ 2 < k′ 3. Thus, We can conclude
```
that 1 +

```
dk′
dgP
```
#### >1 +

```
dk′
dgT
>0. Therefore, the investment is larger for the economy with the
```
persistent government spending shock compared to that with the transitory government
spending shock. Also, compared to the economy without any government spending shock,
the economy with the transitory government spending shock experience a drop in the in-
vestment. However, the response in the case of the persistent government shcok is unclear
because it depends on the relative movement of investment MC and MB.

- Employment

```
Differentiate both sides ofn=h(k′,k′+g) byg. Then we get
dn
dg
```
#### =

```
dh
dg
```
#### +

```
dj
dk′
```
```
dk′
dg
```
#### =

```
dh
dg
```
#### (1 +

```
dk′
dg
```
#### )

THe second equality comes from the symmetry

```
dh
dg
```
#### =

```
dh
dk′
```
. Moreover,

```
dh
dg
>0(positive value
```
from assuming the positive imcome effect of the leisure) does not depend on whether the
government spending shock is transitory or persistent. From the investment part above, it

is known that 1 +

```
dk′
dgP
```
#### >1 +

```
dk′
dgT
>0. Therefore,
```
```
dn
dgP
```
#### >

```
dn
dgT
```
#### >0.

- Output

ConsiderY=f(k,n). For the persistent government shock, higher level of employment
achieved as shown above. Therefore, the output is larger for the persistent government
shock compared to the temporary government shock.

- Consumption

```
Differentiate both side ofc=q(k,k′+g) byg. Then we get
dc
dg
```
#### =

```
dq
dg
```
#### +

```
dq
dk′
```
```
dk′
dg
```
#### =

```
dq
dg
```
#### (1 +

```
dk′
dg
```
#### )


Since
dq
dg
<0, we can confirm that
dc
dgP

#### <

```
dc
dgT
```
#### <0.

- wage

We have shown that the employment is higher for the persistent shock than the tem-
porary shock. Sincew=fn(k,n) andwis a decreasing function of n, the persistent shock

achieves lower wage than the transitory shock.

```
dw
dgP
```
#### <

```
dw
dgT
```
#### <0.

- Interest rate

We have shown that the employment is higher for the persistent shock than the tem-
porary shock. Sincer=fk(k,n) and r is an increasing function of n, the persistent shock

achieves higher interest rate than the transitory shock.

```
dr
dgP
```
#### >

```
dr
dgT
```
#### >0.

### 2.2 Plots for the impulse response

As shown in the above, the response of the investment is larger for the persistent govern-
ment spending shock that for the temporary shock. Also, the change for the employment,
output, and interest rate is larger for the persistent shock than for the transitory shock,
both showing positive jumps. Moreover, the change for the consumption and wage is
smaller for the persistent shock than for the transitory shock while both show negative
jumps. Therefore, we can conclude that the above IRFs are consitent with the theoretical
analysis in the above.

```
Figure 1: Persistent Government Spending Shock
```

```
Figure 2: Transitory Government Spending Shock
```
### 2.3 Plots for the impulse response given a composite consumption func-

### tion

Given the change of consumer’s utility function fromu(c,n) tou(c+αg,n) where 0<
α <1 which implies that government spending is an imperfect substitution for private
consumption, the static problem would change to

```
W(k,k′+g) = max
n,cp∈B(k,k′+g)
```
```
u(c+αg,n)
```

Therefore, when a government spending shock hits the economy, consumption would
fall by less than the baseline model. As a result, the marginal utility of consumption would
also increase by less than the baseline. Consequentely the marginal benefit and cost curves
would shift by less than the baseline.
I suggest the adjusted graph below.

As you can see,k′ 4 > k 2 ′. The transitory shock would decrease investment and wage
to a lesser degree and increase employment, output, and interest rate to a lesser degree
than the baseline. One noticing point is that private consumption would decrease by more
than the baseline since output would increase by less and investment would decrease by
less than the baseline. Intuitively, private consumption can be partially substitued with
government spending.
Though we cannot conclude for sure whetherk′ 5 is bigger or less thank 3 ′, we can surely
know thatk′ 5 > k 4 ′. In other words, the relative size of impact between persistent and
transitory shocks is the same as the baseline model.

```
With the new utility function, the model in baseline changes as below.
```
```
(ct+αgt)−^1 =λt
```
Thus the log linear approximation changes to

```
sc
sc+αsg
(−cˆt) +
```
```
αsg
sc+αsg
(−ˆgt) =λˆt
```
Then,

```
Mcc=
```
#### [

```
xicc∗(sc/(sc+nalpha∗sg)) −xicl∗nabar/(1−nbar)
xilc (−xinn−omegan)−xill∗nbar/(1−nbar)
```
#### ]

```
Mce=
```
#### [

```
0 nalpha∗sg/(sc+nalpha∗sg)
(1 +omegaa) omegag
```
#### ]

where nalpha represents theαin the new utility function.


Figure 3: Persistent Government Spending Shock


Figure 4: Transitory Government Spending Shock


As you can see, asαincreases, the effect of the shock decreases. As before, persistence
shocks show stronger effects than transitory shocks. Whenα= 1 or complete substitutabil-
ity, only private consumption decreases because it is substituted by government spending.
Below is a table summarizing the government spending multiplier for various macroe-
conomic variables. As you can see in the table, the multipliers tend to decrease for output,

```
Table 1: Government Spending Multiplier
```
employment, consumption, and interest rate asαincreases and increase for investmet and
wage, and interest rate for the transitory shock. This is also applies to the persistent shock
except in the case of investment which tends to increase. This is because that persistent
shock has impact on both the marginal benefit and cost curves together, so depending on
the magnitude of the shift, the government multiplier can become positive and negative.
We can also detect that the effect of the persistent shock tents to be larger than the
transitory shock for the overall values ofα.
One more interesting point is that, whenα= 1, only consumption tends to decrease(as
it is substituted by government spending) and all other variables tend to remain the same.
This is because government spending plays a role as a perfect substitute whenα= 1.


