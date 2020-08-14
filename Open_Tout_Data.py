##
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from detect_vegetation import __detect_vegetation__
from spectrum import __Wavelength__
##

g = open('REFLECTANCE_Tomatoas_Monday.dat', 'rb')
# im1= g.readline(samples*bands*lines)
im = np.fromfile(g, np.uint16,(1024 * 448 * 1020))
g.close()
im2 = np.reshape(im, [1024, 448, 1020], order="F")
im2=np.transpose(im2,[0,2,1])
X=np.array(im2[739:820,629:690,:])
b=np.shape(X)
avg_X = np.squeeze(np.mean(np.mean(X, axis=1), axis=0))
W1=__Wavelength__()
morph=9
nclass = 4
(avg2,IM,im_class_pos,im_class_pos_op)=__detect_vegetation__(im2,W1,morph,nclass)
print("Finish___1")
###
g = open('REFLECTANCE_Tomatoes_Tuesday.dat', 'rb')
# im1= g.readline(samples*bands*lines)
imm1 = np.fromfile(g, np.uint16,(1024 * 448 * 1021))
g.close()
im3 = np.reshape(imm1, [1024, 448, 1021], order="F")
im3=np.transpose(im3,[0,2,1])
X1=np.array(im3[739:820,629:690,:])
b1=np.shape(X1)
avg_X1 = np.squeeze(np.mean(np.mean(X1, axis=1), axis=0))
(avg3,IM1,im_class_pos1,im_class_pos_op1)=__detect_vegetation__(im3,W1,morph,nclass)
print("Finish___2")
######
g = open('REFLECTANCE_tomatoes_tuesday_outside_2_2019-05-14_05-48-45.dat', 'rb')
imm2 = np.fromfile(g, np.uint16,(1024 * 448 * 981))
g.close()
im4 = np.reshape(imm2, [1024, 448, 981], order="F")
im4=np.transpose(im4,[0,2,1])
X2=np.array(im4[:,:,:])
b=np.shape(X2)
avg_X2 = np.squeeze(np.mean(np.mean(X2, axis=1), axis=0))
(avg4,IM2,im_class_pos2,im_class_pos_op2)=__detect_vegetation__(im4,W1,morph,nclass)
print("Finish___3")
####
g = open('refl_no_430.dat', 'rb')
imm3 = np.fromfile(g, np.uint16,(1024 * 422 * 1021))
g.close()
im5 = np.reshape(imm3, [1024, 422, 1021], order="F")
im5=np.transpose(im5,[0,2,1])
#X=np.array(im2[739:820,629:690,:])
X3=np.array(im5[280:770,380:850,:])
b=np.shape(X3)
avg_X3 = np.squeeze(np.mean(np.mean(X3, axis=1), axis=0))
W2=W1[26:448]
(avg5,IM3,im_class_pos3,im_class_pos_op3)=__detect_vegetation__(X3,W2,morph,nclass)
print("Finish___4")
######
g = open('refl_no_4300.dat', 'rb')
imm4 = np.fromfile(g, np.uint16,(1024 * 422 * 1021))
g.close()
im6 = np.reshape(imm4, [1024, 422, 1021], order="F")
im6=np.transpose(im6,[0,2,1])
#X=np.array(im6[739:820,629:690,:])
X4=im6[249:600,449:950,:]
b=np.shape(X4)
avg_X4 = np.squeeze(np.mean(np.mean(X4, axis=1), axis=0))
(avg6,IM4,im_class_pos4,im_class_pos_op4)=__detect_vegetation__(X4,W2,morph,nclass)
print("Finish___5")
#####
samples = 1024
bands = 448
lines = 1020
g = open('REFLECTANCE_Tomatoes_Friday.dat', 'rb')
# im1= g.readline(samples*bands*lines)
imm5 = np.fromfile(g, np.uint16,(1024 * 448 * 1020))
g.close()
im7 = np.reshape(imm5, [1024, 448 , 1020], order="F")
im7=np.transpose(im7,[0,2,1])
X5=np.array(im7[250:620,450:920,:])
b=np.shape(X5)
avg_X5 = np.squeeze(np.mean(np.mean(X5, axis=1), axis=0))
(avg7,IM5,im_class_pos5,im_class_pos_op5)=__detect_vegetation__(im7[250:620,450:920,:],W1,morph,nclass)
print("Finish___6")

plt.plot(W1,avg2/10000,label="Monday")
plt.plot(W1,avg3/10000,label="Thuesday")
plt.plot(W2,avg5/10000,label="Wednesday")
plt.plot(W2,avg6/10000,label="Thursday")
plt.plot(W1,avg7/10000,label="Friday")
plt.plot(W1,avg4/10000,label="Thuesday_outside")
plt.legend()
plt.show()