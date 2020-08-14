import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from detect_vegetation import __detect_vegetation__
from spectrum import __Wavelength__
from filtre import __Filtre__
samples = 1024
bands = 422
lines = 1021
#g = open('REFLECTANCE_Tomatoes_Thursday.dat', 'rb')
g = open('refl_no_4300.dat', 'rb')
im = np.fromfile(g, np.uint16,(samples * bands * lines))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2,[0,2,1])
X=np.array(im2[739:820,629:690,:])
XX=im2[249:600,449:950,:]
b=np.shape(X)
avg_X = np.squeeze(np.mean(np.mean(XX, axis=1), axis=0))
Wavelength=__Wavelength__()
Wavelength2=Wavelength[26:448]
morph=9
nclass = 4
(avg5,avg55,IM,im_class_pos,im_class_pos_op)=__detect_vegetation__(XX,Wavelength2,morph,nclass)
print("Class",np.shape(avg5))
data_01 = np.asarray((avg5/10000))
np.savetxt("Ref_Thursday_Classification.csv", data_01, delimiter=",")
plt.subplot(2,3,1)
plt.imshow(im2[249:600,449:950,234],interpolation='mitchell',cmap='viridis',norm=LogNorm())
plt.subplot(2,3,2)
plt.plot(Wavelength2,avg_X /10000,'b')
plt.subplot(2,3,3)
plt.imshow(IM, interpolation='mitchell', cmap='viridis', norm=LogNorm())
plt.subplot(2,3,4)
plt.imshow(im_class_pos)
plt.subplot(2,3,5)
plt.plot(Wavelength2,avg5/10000,'b')
plt.subplot(2,3,6)
plt.imshow(im_class_pos_op, cmap=plt.cm.gray)
plt.show()
