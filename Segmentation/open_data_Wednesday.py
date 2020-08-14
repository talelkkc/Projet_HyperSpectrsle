import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from detect_vegetation import __detect_vegetation__
from spectrum import __Wavelength__
from Xml_file import _Tree_
from Nombre_ligne import _Nmb_ligne
"""
samples = 1024
bands = 422
lines = 1021
"""
A='/home/talel/Desktop/Trav_IMP/metadata/Tomatoes_Wednesday.xml'
B='spec_day34.txt'

(sam,li)=_Tree_(A)
samples = int(sam)
bands = int(_Nmb_ligne(B))
lines = int(li)
g = open('/home/talel/Desktop/hyper/Tomatoes_Wednesday/capture/refl_no_430.dat', 'rb')
im = np.fromfile(g, np.uint16,(samples * bands * lines))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2,[0,2,1])
#X=np.array(im2[739:820,629:690,:])
X=np.array(im2[280:770,380:850,:])
b=np.shape(X)
avg_X = np.squeeze(np.mean(np.mean(X, axis=1), axis=0))
Wavelength=__Wavelength__()
Wavelength2=Wavelength[26:448]
morph=9
nclass = 4
(avg4,avg4,IM,im_class_pos,im_class_pos_op)=__detect_vegetation__(X,Wavelength2,morph,nclass)
print("Class",np.shape(avg4))
avg4 = ((avg4/4000)*4100)/10000
data_01 = np.asarray((avg4))
np.savetxt("Ref_Wednesday_Classification.csv", data_01, delimiter=",")
plt.subplot(2,3,1)
plt.imshow(im2[279:770,379:850,234],interpolation='mitchell',cmap='viridis',norm=LogNorm())
plt.subplot(2,3,2)
plt.plot(Wavelength2,avg_X /10000,'b')
plt.subplot(2,3,3)
plt.imshow(IM, interpolation='mitchell', cmap='viridis', norm=LogNorm())
plt.subplot(2,3,4)
plt.imshow(im_class_pos)
plt.subplot(2,3,5)
plt.plot(Wavelength2,avg4,'b')
plt.subplot(2,3,6)
plt.imshow(im_class_pos_op, cmap=plt.cm.gray)
plt.show()
