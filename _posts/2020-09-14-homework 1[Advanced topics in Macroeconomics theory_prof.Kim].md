---
layout: post
title: Advanced topics in Macroeconomics theory_Homework1[_prof. Kim]
categories:
  - Study_econ&math
tags:
  - economics
  - grid search
  - value function iteration
last_modified_at: 2020-09-14
use_math: true
---
### homework1

* [link for homework1](https://drive.google.com/file/d/1NwIns2RpvsaFT8t7OPsIhRT7dhed_d8P/view?usp=sharing)

Consider the following problem.

$$\max_{\{c_{t},k_{t+1}\}^{\infty}_{t=0}}\sum^{\infty}_{t=0}\beta^{t}u(c_{t})$$
subject to an initial $k_{0}$ and a law of motion for capitla
$$c_{t} + k_{t+1} = F(k_{t})+(1-\delta)k_{t}$$

**1. Write the Bellman equation.**

$$\begin{aligned}
V(k) = \max_{k'}[u(c)+\beta V(k')]\\ \text{s.t.} \ \; \; c + k' = F(k)+(1-\delta)k\end{aligned}$$

**Now assume that $u(c) = log(c),\ F(k)=Ak^{\alpha}$ where
$A>0,\ \alpha \in (0,1),\ \beta \in (0,1)$, and $\delta=1$.**

**2. Derive $V(k)$. Note that $V(k)$ is a function of $k$ as well as
parameters $(A,\alpha,\beta)$.**

Let's guess $V(k)= B+M\ln k$. Then, $V'(k)=M/k$, and $V'(k')=M/k'$.

We can rewrite the Bellman equation as follows:
$$V(k) = \max_{k'}[ln(Ak^{\alpha}-k')+\beta V(k')]$$

Differentiation with respect to $k'$, yields
$$-\frac{1}{Ak^{\alpha}-k'}+\beta V'(k')=0$$

Substitute for $V'(k')$, then $$-\frac{1}{Ak^{\alpha}-k'}+\beta M/k'=0$$

Rearrange the above equation $$k'=\frac{A\beta M}{1+\beta M}k^{\alpha}$$

Now, let's substitute guess-and-verify equation into Bellman equation.
$$B+M\ln k=lnc +\beta(B+M\ln k')$$

Next, substitute the value of $k'$,
$$B+M\ln k=ln(Ak^{\alpha}-\frac{A\beta M}{1+\beta M}k^{\alpha}) +\beta B+\beta M \ln (\frac{A\beta M}{1+\beta M}k^{\alpha})$$

This can be reduced as follows: $$\begin{aligned}
B+M\ln k= &[ln(A)+\beta B + \beta M ln(A\beta M) - \beta M ln(1+\beta  M) -ln(1+\beta M)]\\& + \alpha(1+\beta M) ln(k)\end{aligned}$$

Thus, we get $$M=\frac{\alpha}{1-\alpha\beta}$$

and
$$B = (1-\beta)^{-1}[\frac{\alpha\beta}{1-\alpha\beta}ln(A\alpha\beta)+ln(A-A\alpha\beta)]$$

**3. Derive the saving function $g(k)$.**

Since $k'=\frac{A\beta M}{1+\beta M}k^{\alpha}$,

$$k'=\frac{\frac{A\alpha\beta}{1-\alpha\beta}}{1+\frac{\alpha\beta}{1-\alpha\beta}}k^{\alpha}$$

Then, $$k'=g(k)=A\alpha\beta k^{\alpha}$$

**4. Write a code for a value function iteration through the grid
search. Report the total number of iteration. Let
$K=[\underline{\sbox\tw@{$k$}\dp\tw@\z@\box\tw@},\bar{k}]$ be the domain
for $k$ such that $g(\bar{k})<\bar{k}$ and
$g(\underline{\sbox\tw@{$k$}\dp\tw@\z@\box\tw@})>\underline{\sbox\tw@{$k$}\dp\tw@\z@\box\tw@}$.
Set $N=30$ equally spaced grid points on $K$.**

Total number of iteration = 349.

I used Matlab for value function iteration. Codes are as follows:

```Matlab
clc;
clear all;

%% Store the parameters in a structure
beta = 0.96;
alpha = 0.36;
delta = 1;
A = 5;


%% Solve for the steady state
ks = ((1/beta - 1 + delta)/(alpha * A))^(1/(alpha-1));

%% Create a grid for K
nbk = 30; % number of data points in the grid
crit = 1; % convergence criterion
epsi = 1e-6; % convergence parameter
iter = 0; % number of iteration

dev = 0.9; % maximal deviation from steady state
kmin = (1-dev)*ks; % lower bound on the grid
kmax = (1+dev)*ks; % upper bound on the grid

dk = (kmax-kmin)/(nbk); % implied increment
kgrid = linspace(kmin,kmax,nbk)'; % builds the grid

v0 = zeros(nbk,1);
v = zeros(nbk,1); % value function
dr = zeros(nbk,1); % decision rule (will contain indices)
tv = zeros(nbk,1);

%% Grid Search


while crit>epsi;
    
iter = iter+1;

for i=1:nbk
%
% compute indexes for which consumption is positive.
% if consumption is negative, then the process is implausible. 
% c = (A*kgrid(i)^alpha+(1-delta)*kgrid(i)-k)
%
tmp = (A*kgrid(i)^alpha+(1-delta)*kgrid(i)-kmin);
imax = min(floor(tmp/dk)+1,nbk);
%
% consumption and utility
%
c = A*kgrid(i)^alpha+(1-delta)*kgrid(i)-kgrid(1:imax);
util = log(c);
%
% find value function
%
[value, index] = max((util+beta*v(1:imax)));
tv(i)=squeeze(value);
dr(i)=index;
end;

crit = max(abs(tv-v)); % Compute convergence criterion
v = tv; % Update the value function
end

%% Save result
save results.mat

%% Final solution
% approximation values from grid search
k_star = kgrid(dr);
c_star = A*kgrid.^alpha+(1-delta)*kgrid-k_star;
util_star= log(c_star);
v_star = util_star/(1-beta);

v_true = (1-beta)^(-1)*(alpha*beta/(1-alpha*beta)*log(A*alpha*beta)+log(A-A*alpha*beta))+alpha/(1-alpha*beta)*log(kgrid);
g_true = A*alpha*beta*kgrid.^(alpha);

%% plot
figure(1);
plot(kgrid, v_true, 'r', kgrid, v_star, 'b','LineWidth',2);
xlabel('kgrid'); 
ylabel('Value'); 
legend({'True value','Approximation'},'Location','east','FontSize',16);
title('Transition of Value function');
ax = gca;
ax.FontSize = 13;

figure(2);
plot(kgrid, g_true, 'r', kgrid, k_star, 'b', 'LineWidth',2);
xlabel('kgrid'); 
ylabel('Saving'); 
legend({'True value','Approximation'},'Location','east','FontSize',16);
title('Transition of Policy function');
ax = gca;
ax.FontSize = 13;
```

**5. Plot the true value function from 2 and your approximation from 4
in one graph.**

![Transition of Value function](figure1.eps){width="\\textwidth"}

**6. Plot the approximation error of policy functions. That is, present
$\hat{g}(k_{i})-g(k_{i})$ for each grid point $k_{i}$ where
$\hat{g}(k_{i})$ is the approximated policy function from 4 and
$g(k_{i})$ is the true saving function from 3.**

![Transition of Policy function](figure2.eps){width="\\textwidth"}

9 Joe Haslag. Guess and Verify notes for Dyn prog. Retrieved from
*http://faculty.missouri.edu/ haslagj/files/guesandvernotes.pdf*.

Fabrice Collard. Value Iteration. Retrieved from
*http://fabcol.free.fr/pdf/lectnotes7.pdf*.
