import numpy as np

import matplotlib.pyplot as plt
from filtre import __Filtre__

M =np.array([[[3, 1, 5], [9, 8, -1], [10, 12, 2]],[[15,3,8],[0,6,11],[4,9,-2]]])
#FF=[60,61,3,4,5,6,7,8,9,10,11,12,13,51,50,51,17,18,19,20,21,-20,-40,24,25,26,27,28,69,70]
print(np.shape(M))
l = len(M)
plt.subplot(2,2,1)
plt.plot(M[:,:,1])
#plt.show(R)
print("****",M[1,2])
print("**___**",M[1][2])
for i in range(1):
    for j in range(2):
        for k in range (2):
            if ((M[i,j,k] > (M[i - 1,j,k] + 0.2)) & (M[i,j,k] > (M[i + 1,j,k] + 0.2))):
                M[i,j,k] = (M[i - 1,j,k] + M[i + 1,j,k]) / 2
plt.subplot(2, 2, 2)
plt.plot(M[:,:,1])
            #plt.show(RR)

if (M[0,0,0] > M[1,0,0] + 2):
    M[0,0,0] = M[1,0,0] - 1
plt.subplot(2, 2, 3)
plt.plot(M[:,:,1])
            #plt.show(RR)

if (M[1 ,0,0] > M[1 - 1,0,0] + 2):
    M[1 ,0,0] = M[1 -1,0,0] + 1
plt.subplot(2, 2, 4)
plt.plot(M[:,:,1])

plt.show()

""""
#for i in range(samples - 1):
for i in range(30):
    FF=__Filtre__(FF)
RR = plt.plot(FF)
plt.show(RR)
"""
"""       
for i in range(l - 1):
    if ((FF[i] > (FF[i-1] + 2)) & (FF[i] > (FF[i+1] + 2))):
        FF[i] = (FF[i - 1]+FF[i + 1])/2
        RR = plt.plot(FF)
        plt.show(RR)

if (FF[0]>FF[1]+2):
    FF[0]=FF[1]-1
    RR = plt.plot(FF)
    plt.show(RR)

if (FF[l-1]>FF[l-2]+2):
    FF[l-1]=FF[l-2]+1
    RR = plt.plot(FF)
    plt.show(RR)


"""""