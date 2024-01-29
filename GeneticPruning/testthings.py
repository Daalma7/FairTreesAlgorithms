# File to test things 

import numpy as np

a = np.array([0,1,1,0])
b = np.array([0,1,0,1])

newa = a[:min(len(a), len(b))]
newb = b[:min(len(a), len(b))]

try:
    idx = np.where( (newa>newb) != (newa<newb) )[0][0]

    if a[idx] < b[idx]: print("a < b")
    if a[idx] > b[idx]: print("a > b")
except IndexError:
    
    if len(a) < len(b): print("a < b")
    if len(a) > len(b): print("a > b")
    if len(a) == len(b): print("a == b")

a = [0,0,0,0,0]
b = 1

a.insert(2, 1)
print(a)