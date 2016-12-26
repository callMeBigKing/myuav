from scipy import integrate
import numpy as np



r=2
r_=r*0.9

def func(x,y):
    re=0
    if   (0.9*r)**2<x**2+y**2<r**2:
        re=1
    return re
q,err=integrate.dblquad(func,-r,r,lambda x:0,lambda x:r)
print(q)