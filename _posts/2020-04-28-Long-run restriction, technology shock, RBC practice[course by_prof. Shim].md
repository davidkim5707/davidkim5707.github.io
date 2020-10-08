---
layout: post
title: Long-run restriction, technology shock, RBC practice[course by_prof. Shim]
categories:
  - Study_econ&math
tags:
  - economics
  - Long-run restriction
  - VAR
  - technology shock
  - NK 
  - RBC
last_modified_at: 2020-04-28
use_math: true
---
### homework1

* [link for homework1](https://drive.google.com/uc?export=view&id=1D08ZyOe7J15jrmlAUp4bohVfmgVLM9AI)  

### Using the Long-run restriction that non-technology shock cannot affect the growth rate of employment in the long-run (following the specification of Gal´ı (1999)), show the impulse response function of hours worked (employment) to the identified technology shock.

As follows Gali(1999), I estimated the impulse response function of
housrs worked (employment) to the identified technology shock using
South Korea's yearly data covering period 1963-2011 from Federal Reserve
Economic Data. Gali(1999) obtained results by using the
first-differenced log of the employed civilian labor
force(\"employment\") as a labor-input measure and the baseline series
for labor productivity was constructed by subtracting the labor-input
measure from the log of GDP. Thus, to check whether the South Korea
employment data should be differenced or not, I conducted ADF(Augmented
Dickey-Fuller) unit root test by STATA.

![window](https://drive.google.com/uc?export=view&id=1opc7CKq_sIKrsQafLty0JB9Mgl-t8XYi)  

The test reveals that the null hypothesis that the assigned variable
follows a unit root cannot be rejected, which means that the log of the
employment variable in South Korea needs to be first-differenced. Using
the Long-run restriction that non-technology shock cannot affect the
growth rate of employment in the long-run, Figure 2 displays the impluse
response of hours worked (employment) to the identified technology
shock. All data are yearily, and not seasonally adjusted.


![window](https://drive.google.com/uc?export=view&id=1WcsPgpW2oqw-yRhl2S0iT6C6GH7vGlGV)  

 
### Compare your results with Gal´ı (1999).


As you can see in the Figure 3, the impulse response of hours worked
(employment) in South Korea annual data appears totally different from
the ones in Gali(1999). According to Gali(1999), in response to a
positive technology shock of size equal to one-standard deviation, hours
worked experiences an immediate decrease of about 0.4 percent,
eventually stabilizing at a level somewhat lower, which is interpreted
as the reason for the gap between the initial increase in labor
productivity and the (smaller) increase in output.

![window](https://drive.google.com/uc?export=view&id=1WcsPgpW2oqw-yRhl2S0iT6C6GH7vGlGV)
![window](https://drive.google.com/uc?export=view&id=1Oc4Ea7xm9nS7lgNYi7ArrHhIaVb5oxTJ)  


However, contrary to the result of Gali(1999), the impulse response of
hours worked (employment) in South Korea shows an immediate increase,
but the initial rise appears to be vanished in a short period.

As following the plot technicians in Blanchard $\&$ Quah(1989), Figure 4
displays the cumulative summation of estimated impulse response of hours
worked and productivity to the identified technology shock in both of
South Korea and Japan(from Gali(1999)).

![window](https://drive.google.com/uc?export=view&id=1ttHi7DYSn3FzjGEHteBeKep3ScH12_y0)
![window](https://drive.google.com/uc?export=view&id=1QI274ucWdxf1Zss5aR9tBVZuQmtnzx2H)  

The result of Figure 4 is quite interesting in that hours worked in most
countries suggested in Gali(1999) but Japan experience a decrease in
response to the positive technology shock. However, hourse worked in
South Korea from new VAR and Japan show an increase. There may exist
numerous reasons to elicit the eccentric response in South Korea. One
may assert that the sticky price model in NK suggested by Gali(1999)
would not be valid to explicate the relation between employment and
technology shock in South Korea, which can be interpreted that South
Korea employment is more paralled with RBC models. Others may argue
that, citing the conseqeunces in Figure 4, there would be specific
un-observed characterisitcs around the East Asis leading the peculiar
responses of employment. Unfortunately, in this assignment, it is
insufficient to deduce which arguments may be more vaild among numerous
assertions.

However, as we can see in Figure 5, Kim et.al(2008) suggested a impulse
response of hours worked to the technology shock defined as labor
productivity, which is consistent to the Gali(1999) and totally
different from mine. Kim et.al(2008) revealed that they used the data
covering period 1985-2002 for labor productivity, which is defined as
the ratio of gross domestic product (GDP) to total hours worked, is
compiled by the Bank of Korea. in addition, their measure of employment
is hours worked, which we derived as the product of total number of
employed workers and average hours worked per week ($h_{}i$) and all
variables are converted into constant 2000 prices. They also used
bivariate long-run restriction VAR model to estimate the impulse
response of hours worked to the identified technology shock.

![window](https://drive.google.com/uc?export=view&id=1ttHi7DYSn3FzjGEHteBeKep3ScH12_y0)
![window](https://drive.google.com/uc?export=view&id=17KbB7GpNikae2H3EADNsH75Dulrh3rqA)  

Thus, if the VAR specification using in both are same exactly, one
inference is that the eccentric response of hours worked described in
Figure 2-(a) may be due to the wrong data using. Thus, it may be needed
to re-estimate the impulse response of hours worked using the data used
in KIm et.al(2008) before concluding that South Korea has peculiar
impuls response of hours worked to the identified technology shock.

9 Gali, J.(1999) *Technology, Employment, and the Business Cycle: Do
Technology Shocks Explain Aggregate Fluctuations?*. The American
Economic Review, 89(1), 249-271.

Blanchard, O., $\&$ Quah, D. (1989) *The Dynamic Effects of Aggregate
Demand and Supply Disturbances*. Economic Review, 79(4), 655-673.

Kim, S.(2008) *Productivity and employment in a developing country:
Evidence from Republic of Korea*. ERD Working Paper Series, 0(116),
1-30.
