# Stability of human preference

https://www.frontiersin.org/articles/10.3389/fnhum.2017.00289/full

Place 8 items in order of preference. Call the ranking of image N `1 <= R(N) <= 8`
Repeat at a later point. Call this `S(N)` 

Variation `V = sum( abs(R(N)-S(N)) ) [N=1..8]` lies in `(0,32)`. 

The paper defines a consistency as `Con = (32-V)/8` which lies in `(0,4)`

Could calculate AB consistency as `Cab = sum( 0.5*(1+sign( (R(N)-R(M))*(S(N)-S(M)) )) )/28   [N=1..7, M=N+1..8]`. The term `sign( (R(N)-R(M))*(S(N)-S(M)) )` has value 1 if the two were ranked in the same order, -1 if the order was reversed; 0.5*1+() is therefore 1 or 0, and as there are 28 pairs `Cab` lies in `(0,1)` (analogous to the AB score in my code).

WLOG, we can put `R(N)=N`, and then for each possible set of `S(N)` (of which there are 8! = 40320, a very manageable number) calculate `Con` and `Cab` to establish a rough relationship. A given value of `Con` can arise from a number of permutations with different `Cab`, so average over all possible permutations of `S`:

|Con|mean Cab|
|-|-|
|0.00|0.214|
|0.25|0.265|
|0.50|0.319|
|0.75|0.371|
|1.00|0.422|
|1.25|0.475|
|1.50|0.527|
|1.75|0.578|
|2.00|0.630|
|2.25|0.680|
|2.50|0.731|
|2.75|0.781|
|3.00|0.830|
|3.25|0.877|
|3.50|0.922|
|3.75|0.964|
|4.00|1.000|

In the paper they found that a large random sample scored `Con` of 1.37, consistent with a `Cab` of 0.5. The average `Con` was 2.56 in tests 14 days apart, corresponding to a `Cab` of 0.743, or 74.3%.