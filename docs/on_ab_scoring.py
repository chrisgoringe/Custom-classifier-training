import math
def x_needed(N, threshold=0.5):
    x = 0
    D = 0

    while D<threshold:
        D += (1-D)*(2+2*D*N)/(N*N)
        x += 1

    return x

threshold = 0.5
for a in range(5,21,5):
    n = 100*a
    print("{:>4} images requires {:>5} comparisons".format(n, x_needed(n, 1/math.sqrt(n))))