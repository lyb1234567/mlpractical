import math
import operator as op
from functools import reduce
from math import factorial
def nPr(n, r):
    return int(factorial(n)/factorial(n-r))
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom
def root(a,b,c):
    try:
        delta=b**2-4*a*c
        x1=(-b+math.sqrt(delta))/(2*a)
        x2=(-b-math.sqrt(delta))/(2*a)
        if x1>0:
            return x1
        else:
            return x2
    except:
        raise ValueError
p=nPr(2048,12)
print(nPr(3,2))
p2=pow(2048,12)
solution=root(1,-1,-(math.log(2))*2*p)
solution2=root(1,-1,-(math.log(2))*2*p2)
print("All the potential options:"+"{:e}".format(p))
print("{:e}".format(solution))
print("\n\n")
print("All the potential options:"+"{:e}".format(p2))
print("{:e}".format(solution2))

