---
layout: post
title: Technical Note for 3rd paper proposal.
categories:
  - Technical Notes
tags:
  - Technical Notes
last_modified_at: 2020-06-24
use_math: true
---

### Technical note
Measuring the impact of COVID-19_tentative title  

We used fmincon in Matlab.  

### figure 1  

#### objective function for figure  

```Matlab
function F = opt_dw2(x, beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T)

cs1 = x(1);
cr1 = x(2);
h1  = x(3);
cs2 = x(4);
cr2 = x(5);
h2 = x(6);
cs3 = x(7);
cr3 = x(8);
h3 = x(9);
s1 = x(10);
k2 = x(11);
m1 = x(12);
m2 = x(13);
m3 = x(14);

F= log(cs1)+log(cr1)-B*(h1^(1+psi))/(1+psi)+beta*(1-ec*(cs1)-eh*h1)*(log(cs2)+log(cr2)-B*(h2^(1+psi))/(1+psi))+beta*(ec*(cs1)+eh*h1)*(log(cs3)+log(cr3)-B*(h3^(1+psi))/(1+psi)-uh);
F= -F;
end

```
#### constraints  

```Matlab
function [con,coneq]=coneq(x,beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T)

cs1 = x(1);
cr1 = x(2);
h1  = x(3);
cs2 = x(4);
cr2 = x(5);
h2 = x(6);
cs3 = x(7);
cr3 = x(8);
h3 = x(9);
s1 = x(10);
k2 = x(11);
m1 = x(12);
m2 = x(13);
m3 = x(14);
coneq=[w1*h1-cs1-cr1-s1-k2+T; w2*h2-cs2-cr2+(1+r)*s1+k2; (1+r)*s1+w3*h3-cs3-cr3+k2];
con=[cr1+s1+m1-w1*h1; cs1+k2-m1-T; cr2+m2-w2*h2-(1+r)*s1; cs2-m2-k2; cr3+m3-w3*h3-(1+r)*s1; cs3-m3-k2];
end
```

#### fmincon  

```Matlab
clear all;


beta = 0.9;
B    = 20;
w1   = 1;
w2   = 1;
w3   = 0.8;
ec_set   = [0:0.1:0.5];
eh_set   = [0:0.1:0.5];
r    = 0.02;
z    = 1;
psi  = 1;
gamma= 0.99;
T    = 0.2;
uh   = 10;

T_set= [0:0.05:0.3];
W_set= [0:0.1:0.8];


%% both

for l = 1:6
   for k = 1:5    
            eh = eh_set(l);
            ec = ec_set(k);
            T = T_set(1);
            w3 = W_set(1);
            
        x0  = [0.5,0.5,0.3,0.5,0.5,0.3,0.5,0.5,0.3,1,0,0,0,0];
        
        lb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        ub = [];
        A = [];
        b = [];
        Aeq = [];
        beq = [];

        options=optimoptions('fmincon','Algorithm','sqp',...
        'TolCon',10^(-6),'TolX',10^(-6),'TolFun',10^(-12),...
        'MaxIter', 5*10^2,'MaxFunEvals',2*10^5,...
        'MaxSQPIter',10^3);
   
        x = fmincon('opt_dw2',x0,A,b,Aeq,beq,lb,ub,'coneq2',options,beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T); %
        %y = fmincon('opt_dw1',x0,A,b,Aeq,beq,lb,ub,'coneq3',options,beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T); % baselinemodel with restriction
        
        cs(k,l) = x(1);
        h1(k,l)= x(3);
        s(k,l) = x(10);
        cr(k,l)= x(2);
        p(k,l) = ec*(x(1))+eh*x(3);        
        csratio(k,l) = x(1)/(x(1)+x(2)+x(10));
        crratio(k,l) = x(2)/(x(1)+x(2)+x(10));
        sratio(k,l) = x(10)/(x(1)+x(2)+x(10));
        k2(k,l) = x(11);        
        %cs1(k,l) = y(1);
        %h11(k,l)= y(3);
        %s1(k,l) = y(10);
        %cr1(k,l)= y(2);
        %p1(k,l) = ec*(y(1))+eh*y(3);
        %csratio1(k,l) = y(1)/(y(1)+y(2)+y(10));
        %crratio1(k,l) = y(2)/(y(1)+y(2)+y(10));
        %sratio1(k,l) = y(10)/(y(1)+y(2)+y(10));

   end

end
  figure(1)
  surf(csratio)
  figure(2)
  surf(crratio)
  figure(3)
  surf(sratio)
  figure(4)
  surf(h1)
  figure(5)
  surf(p)
  
    
  %figure(1)
  %surf(W_set,T_set,csratio1)
  %figure(2)
  %surf(W_set,T_set,crratio1)
  %figure(3)
  %surf(W_set,T_set,sratio1)
  %figure(4)
  %surf(W_set,T_set,h11)
  %figure(5)
  %surf(W_set,T_set,p1)
  %plot(T_set, a, T_set, b, T_set, c)
  %legend('1', '2', '3')

```

### figure 2-4  

#### objective function for figure  

```Matlab
function F = opt_dw2(x, beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T)

cs1 = x(1);
cr1 = x(2);
h1  = x(3);
cs2 = x(4);
cr2 = x(5);
h2 = x(6);
cs3 = x(7);
cr3 = x(8);
h3 = x(9);
s1 = x(10);
k2 = x(11);
m1 = x(12);
m2 = x(13);
m3 = x(14);

F= log(cs1)+log(cr1)-B*(h1^(1+psi))/(1+psi)+beta*(1-ec*(cs1)-eh*h1)*(log(cs2)+log(cr2)-B*(h2^(1+psi))/(1+psi))+beta*(ec*(cs1)+eh*h1)*(log(cs3)+log(cr3)-B*(h3^(1+psi))/(1+psi)-uh);
F= -F;
end

```
#### constraints  

```Matlab
function [con,coneq]=coneq(x,beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T)

cs1 = x(1);
cr1 = x(2);
h1  = x(3);
cs2 = x(4);
cr2 = x(5);
h2 = x(6);
cs3 = x(7);
cr3 = x(8);
h3 = x(9);
s1 = x(10);
k2 = x(11);
m1 = x(12);
m2 = x(13);
m3 = x(14);
coneq=[w1*h1-cs1-cr1-s1-k2+T; w2*h2-cs2-cr2+(1+r)*s1+k2; (1+r)*s1+w3*h3-cs3-cr3+k2];
con=[cr1+s1+m1-w1*h1; cs1+k2-m1-T; cr2+m2-w2*h2-(1+r)*s1; cs2-m2-k2; cr3+m3-w3*h3-(1+r)*s1; cs3-m3-k2];
end
```

#### fmincon  

```Matlab
clear all;

beta = 0.9;
B    = 20;
w1   = 1;
w2   = 1;
w3   = 0.8;
ec_set   = [0:0.1:0.5];
eh_set   = [0:0.1:0.5];
r    = 0.02;
z    = 1;
psi  = 1;
gamma= 0.99;
T    = 0.2;
uh   = 10;

T_set= [0:0.05:0.3];
W_set= [0:0.1:0.8];


%% both

for l = 1:7
   for k = 1:9    
            eh = eh_set(4);
            ec = ec_set(4);
            T = T_set(l);
            w3 = W_set(k);
            
        x0  = [0.5,0.5,0.3,0.5,0.5,0.3,0.5,0.5,0.3,1,0,0,0,0];
        
        lb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        ub = [];
        A = [];
        b = [];
        Aeq = [];
        beq = [];

        options=optimoptions('fmincon','Algorithm','sqp',...
        'TolCon',10^(-6),'TolX',10^(-6),'TolFun',10^(-12),...
        'MaxIter', 5*10^2,'MaxFunEvals',2*10^5,...
        'MaxSQPIter',10^3);
   
        x = fmincon('opt_dw2',x0,A,b,Aeq,beq,lb,ub,'coneq2',options,beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T); %
        %y = fmincon('opt_dw1',x0,A,b,Aeq,beq,lb,ub,'coneq3',options,beta, B, ec, eh, psi, gamma, uh,w1,w2,w3,r,T); % baselinemodel with restriction
        
        cs(k,l) = x(1);
        h1(k,l)= x(3);
        s(k,l) = x(10);
        cr(k,l)= x(2);
        p(k,l) = ec*(x(1))+eh*x(3);        
        csratio(k,l) = x(1)/(x(3)+T);
        crratio(k,l) = x(2)/(x(3)+T);
        sratio(k,l) = x(10)/(x(3)+T);
        m(k,l) = x(12);   
        k2(k,l) = x(11);
        cs2(k,l) = x(4);
        cr2(k,l) = x(5);
        crratio1(k,l) = x(2)/(x(1)+x(2));
        csratio1(k,l) = x(1)/(x(1)+x(2));  
        %cs1(k,l) = y(1);
        %h11(k,l)= y(3);
        %s1(k,l) = y(10);
        %cr1(k,l)= y(2);
        %p1(k,l) = ec*(y(1))+eh*y(3);
        %csratio1(k,l) = y(1)/(y(1)+y(2)+y(10));
        %crratio1(k,l) = y(2)/(y(1)+y(2)+y(10));
        %sratio1(k,l) = y(10)/(y(1)+y(2)+y(10));

   end

end
  figure(1)
  surf(cs)
  figure(2)
  surf(cr)
  figure(3)
  surf(s)
  figure(4)
  surf(h1)
  figure(5)
  surf(p)
  figure(6)
  surf(k2)
  figure(7)
  surf(m)
  figure(8)
  surf(csratio1)
  figure(9)
  surf(crratio1)
    
  %figure(1)
  %surf(W_set,T_set,csratio1)
  %figure(2)
  %surf(W_set,T_set,crratio1)
  %figure(3)
  %surf(W_set,T_set,sratio1)
  %figure(4)
  %surf(W_set,T_set,h11)
  %figure(5)
  %surf(W_set,T_set,p1)
  %plot(T_set, a, T_set, b, T_set, c)
  %legend('1', '2', '3')

```
