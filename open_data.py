import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from detect_vegetation import __detect_vegetation__
from spectrum import __Wavelength__
from Xml_file import _Tree_
"""
samples = 1024
bands = 448
lines = 1020
"""
A="/home/talel/Desktop/hyper/Tomatoes_Monday/metadata/Tomatoas_Monday.xml"
(sam,li)=_Tree_(A)
samples = int(sam)
bands = 448
lines = int(li)
g = open('/home/talel/Desktop/hyper/Tomatoes_Monday/capture/REFLECTANCE_Tomatoas_Monday.dat', 'rb')
# im1= g.readline(samples*bands*lines)
im = np.fromfile(g, np.uint16,(samples * bands * lines))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2,[0,2,1])
X=np.array(im2[739:820,629:690,:])
b=np.shape(X)
#A=plt.imshow(X[:,:,234],interpolation='mitchell',cmap='viridis',norm=LogNorm())
#A=plt.imshow(X[:,:,234],interpolation='gaussian',cmap='viridis',norm=LogNorm())
#plt.show(A)
avg_X = np.squeeze(np.mean(np.mean(X, axis=1), axis=0))
Wavelength=__Wavelength__()
#C=plt.plot(Wavelength,avg_X /10000,'b')
#plt.show(C)
morph=9
nclass = 4
(avg2,avg22,IM,im_class_pos,im_class_pos_op)=__detect_vegetation__(im2,Wavelength,morph,nclass)
print("Class",np.shape(avg2))
data_01 = np.asarray((avg2/10000))
np.savetxt("Ref_Monday_Classification.csv", data_01, delimiter=",")
plt.subplot(2,3,1)
plt.imshow(X[:,:,234],interpolation='mitchell',cmap='viridis',norm=LogNorm())
plt.subplot(2,3,2)
plt.plot(Wavelength,avg_X /10000,'b')
plt.subplot(2,3,3)
plt.imshow(IM, interpolation='mitchell', cmap='viridis', norm=LogNorm())
plt.subplot(2,3,4)
plt.imshow(im_class_pos)
plt.subplot(2,3,5)
plt.plot(Wavelength,avg2/10000,'b')
#plt.subplot(2,3,6)
#plt.imshow(im_class_pos_op, cmap=plt.cm.gray)
plt.subplot(2,3,6)
plt.plot(Wavelength,avg2 /10000,'b')
plt.show()
