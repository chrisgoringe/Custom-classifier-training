import math, random
from scipy.stats import spearmanr

def weight(lcw, count):
    w = math.pow(1-lcw,count)

def spear(data):
    a = sorted(data, key=lambda a:a[1], reverse=True)
    a = list(x[0] for x in a)
    print(spearmanr( a, list(range(len(data))) ))

setsize = 600
comparisons = 0
k = 0.7
data = [ [i,0.001*random.random(),0] for i in range(setsize) ]   # actual rank, score, contests

while comparisons<1500:
    i = random.randrange(setsize)
    j = random.randrange(setsize)
    while j==i: j = random.randrange(setsize)
    delta = data[i][1] - data[j][1]
    p_i = 1.0/(1.0+math.pow(10,-delta))
    if i<j:
        data[i][1] += k * (1-p_i)
        data[j][1] -= k * (1-p_i)
    else:
        data[i][1] -= k * (1-p_i)
        data[j][1] += k * (1-p_i)        
    data[i][2] += 1
    data[j][2] += 1
    comparisons += 1
    if comparisons%100 == 0: spear(data)

