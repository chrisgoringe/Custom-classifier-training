# On AB scoring

How much do you need to do?

Let's define the concept of 'depends on'; Image A's score _depends on_ Image B's score if they have been compared or if, _when_ Image A was compared with Image C, Image C's score _depended on_ Image B. Note that this is not reflexive (A _depends on_ B does not imply B _depends on_ A).

Take a set of `N` images (`N^2` possible dependencies), and say that after `x` comparisons, the average image _depends on_ a fraction `D(x)` of the images (or there exist `DN^2` dependencies). Since at the start there are no dependencies, `D(0)=0`.

If we now compare two images, each will, as a result of that comparison, depend on all the images the other depended on, and on the other. So that's `2 + 2.D.N` possibly new dependencies. But some of these dependecies may have already existed - assuming everything is random, a fraction `D` of them. So the number of new dependencies is `(1-D).(2+2DN)`. Since there are `N^2` possible dependencies, the change in `D` is given by `dD = (1-D).(2+2DN)/N^2 = 2/(N^2) . (1-D)(1+ND) = 2/(N^2) . [-ND^2 + (N-1)D + 1]`.

To get to D = 0.5 
```
 500 images requires  1554 comparisons
1000 images requires  3455 comparisons
1500 images requires  5486 comparisons
2000 images requires  7602 comparisons
```

to get to D = 1/SQRT(N)
```
 500 images requires   800 comparisons
1000 images requires  1759 comparisons
1500 images requires  2782 comparisons
2000 images requires  3846 comparisons
```

(interesting - about half as many)

Rule of thumb... 2-4 comparisons per image