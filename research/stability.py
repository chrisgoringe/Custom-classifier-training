import statistics

def sign(x):
    return 1 if x>0 else -1 if x<0 else 0

def con(S):
    V = sum( abs(n-S[n]) for n in range(8) )
    return (32-V)/8

def cab(S):
    sum = 0
    for n in range(7):
        for m in range(n+1,8):
            sum += 0.5*(1+sign( (n-m)*(S[n]-S[m]) ))
    return sum/28

def gen_s():
    S = [0]*8
    for S[0] in range(8):
        for S[1] in range(8):
            if S[1] in S[0:1]: continue
            for S[2] in range(8):
                if S[2] in S[0:2]: continue
                for S[3] in range(8):
                    if S[3] in S[0:3]: continue
                    for S[4] in range(8):
                        if S[4] in S[0:4]: continue
                        for S[5] in range(8):
                            if S[5] in S[0:5]: continue
                            for S[6] in range(8):
                                if S[6] in S[0:6]: continue
                                for S[7] in range(8):   
                                    if S[7] in S[0:7]: continue
                                    yield(S)

mapping = {}
for S in gen_s():
    cn = con(S)
    cb = cab(S)
    p = mapping.get(cn,[])
    p.append(cb)
    mapping[cn] = p

mapping = list( (cn, statistics.mean(mapping[cn])) for cn in mapping )
mapping.sort()
for cn, cb in mapping:
    print("|{:>4.2f}|{:>5.3f}|".format(cn,cb))
    