import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from detect_vegetation import __detect_vegetation__
from spectrum import __Wavelength__
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
samples = 1024
bands = 448
lines = 1020

#/home/talel/Desktop/hyper/Tomatoes_Friday/capture/REFLECTANCE_Tomatoes_Friday.dat'
g = open('/home/talel/Desktop/hyper/Tomatoes_Friday/capture/REFLECTANCE_Tomatoes_Friday.dat', 'rb')
# im1= g.readline(samples*bands*lines)
im = np.fromfile(g, np.uint16,(samples * bands * lines))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2,[0,2,1])

X=np.array(im2[250:620,450:920,:])
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
(avg6,avg66,IM,im_class_pos,im_class_pos_op)=__detect_vegetation__(im2[250:620,450:920,:],Wavelength,morph,nclass)
print("Class",np.shape(avg6))
data_01 = np.asarray((avg6/10000))
np.savetxt("Ref_Friday_Classification.csv", data_01, delimiter=",")
plt.subplot(2,3,1)
plt.imshow(im2[250:620,450:920,234],interpolation='mitchell',cmap='viridis',norm=LogNorm())
#plt.imshow(im2[:,:,234],interpolation='mitchell',cmap='viridis',norm=LogNorm())
plt.subplot(2,3,2)
plt.plot(Wavelength,avg_X /10000,'b')
plt.subplot(2,3,3)
plt.imshow(IM, interpolation='mitchell', cmap='viridis', norm=LogNorm())
plt.subplot(2,3,4)
plt.imshow(im_class_pos)
plt.subplot(2,3,5)
plt.plot(Wavelength,avg6/10000,'b')
plt.subplot(2,3,6)
plt.imshow(im_class_pos_op, cmap=plt.cm.gray)

plt.show()
