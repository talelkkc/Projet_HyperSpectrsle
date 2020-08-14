import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from skimage.morphology import disk
from skimage.morphology import opening , erosion,dilation, closing, white_tophat

from sklearn import mixture
from spectrum import __Wavelength__
from filtre import __Filtre__
import skimage

samples = 1024
bands = 422
lines = 1021
g = open('refl_no_430.dat', 'rb')
im = np.fromfile(g, np.uint16,(samples * bands * lines))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2, [0, 2, 1])
print("im2", np.shape(im2))
im4=np.array(im2[320:650,380:440, :])
#im3=im2[350:500,550:850,:]
#AAA=plt.imshow(im2[320:650,380:440, 234], interpolation='mitchell', cmap='viridis', norm=LogNorm())
#plt.show(AAA)
A=im4.shape[0]
B=im4.shape[1]
C=im4.shape[2]
M=im4
M_ftr=M
F=0

for i in range(0,A):
    print("___",i)
    for j in range(0,B):
        x=max(M[i,j,:])
        if (x > 10000)&(F<50000000):
            F=F+1
            if (M_ftr[i,j,0]>(M_ftr[i,j,1]+2000))|(M_ftr[i,j,0]>10000):
                M_ftr[i,j,0] = min(M_ftr[i,j,1],1000)
            if (M_ftr[i, j, C-1] > M_ftr[i, j, C - 2] + 2000) | (M_ftr[i, j, C-1] > 10000):
                M_ftr[i, j, C-1] = min(M_ftr[i, j, C - 2], 1000)
            for k in range(0,C-1):
                if (M_ftr[i,j,k]>(M_ftr[i,j,k-1]+2000))|(M_ftr[i,j,k]>(M_ftr[i,j,k+1]+2000)):
                    M_ftr[i, j, k]=min(M_ftr[i,j,k-1],M_ftr[i,j,k+1])

            if max(M_ftr[i,j,:])>10000:
                
                if F>0:
                    #plt.subplot(3,3,1)
                    plt.subplot(2, 1, 1)
                    plt.title(" Avant** ")
                    plt.plot(M[i, j, :])
                
                if F>0:
                    plt.subplot(3,3,2)
                    #plt.subplot(2, 1, 2)
                    plt.title(" Après ")
                    plt.plot(M_ftr[i, j, :])
print("FF FF",F)



XX=np.array(im2[320:650,380:440, :])
b = np.shape(XX)
print("____---__", b)
plt.subplot(3, 3, 3)
plt.imshow(im2[320:650,380:440, 234], interpolation='mitchell', cmap='viridis', norm=LogNorm())
avg_X = np.squeeze(np.mean(np.mean(XX, axis=1), axis=0))
Wavelength = __Wavelength__()
Wavelength2=Wavelength[26:448]
print("*_*_*_*_*", np.shape(Wavelength2))

plt.subplot(3, 3, 4)
plt.plot(Wavelength2, avg_X / 10000, 'b')
#######
nclass = 3
#im3=np.array(im2[249:600,449:950,:])
im3= M_ftr
bands = im3.shape[2]
samples = im3.shape[0]
lines = im3.shape[1]
pas = 5
Wv_pas = np.array(Wavelength2[0:bands:pas])
spl_im2 = np.array(im3[0:samples:10, 0:lines:10, 0:bands:pas])
#plt.subplot(2,3,5)
#plt.imshow(spl_im2[:,:,50],interpolation='mitchell',cmap='viridis',norm=LogNorm())
avg_X1 = np.squeeze(np.mean(np.mean(spl_im2, axis=1), axis=0))
#plt.subplot(2,3,6)
#plt.plot(avg_X1/10000,'b')
Yc = np.array(np.reshape(spl_im2, [spl_im2.shape[0]*spl_im2.shape[1], spl_im2.shape[2]], order="F"))
#plt.subplot(2,3,4)
#plt.plot(Yc[500,:])
print("____YYY____", np.shape(Yc))
GMModel = mixture.GaussianMixture(n_components=nclass, covariance_type='full', max_iter=100, n_init=14).fit(Yc)
print("______GG____", GMModel)
im22 = im3[:, :, 0:bands:pas]
Q = np.array(np.reshape(im3[:, :, 0:bands:pas], [im3.shape[0]*im3.shape[1], im22.shape[2]], order="F"))
prediction_gmm = np.array(GMModel.predict(Q))
IM = np.reshape(prediction_gmm, [samples, lines], order="F")
print("****IM****", np.shape(IM))
plt.subplot(3, 3, 5)
plt.imshow(IM, interpolation='mitchell', cmap='viridis', norm=LogNorm())

Wl3=Wavelength2[0:bands:pas]
EE1=GMModel.means_[0, :]
EE2=GMModel.means_[1, :]
EE3=GMModel.means_[2, :]
EE4=GMModel.means_[2, :]
plt.subplot(3, 3, 9)
LL=plt.plot(Wl3, EE1, Wl3, EE2, Wl3, EE3, Wl3, EE4)
#plt.show(LL)

#############
#Find vegetation cluster
red = 650
near = 850
p_red = np.size(np.argwhere(Wv_pas < red))
p_near = np.size(np.argwhere(Wv_pas < near))
HH = GMModel.means_[:, p_red]
print("_HHH_", HH)
EE = GMModel.means_[:, p_near]
print("_EEE_", EE)
NDVI = (EE-HH)/(EE+HH)
print("_NDVI_", NDVI)
plt.subplot(3, 3, 1)
plt.title(" **_NDVI_** ")
plt.plot(NDVI)
pos2 = NDVI.argmax()
#############
im_class_pos = (IM == pos2)*1
im_class_pos1 = np.equal(IM, pos2)
n_pos = np.sum(np.sum(im_class_pos))
plt.subplot(3, 3, 6)
plt.imshow(im_class_pos)
#imopen
morph = 5
if morph > 0:
    se = disk(4)
    #im_class_pos_op = opening(im_class_pos, se)
    im_class_pos_op = closing(im_class_pos, se)
    plt.subplot(3, 3, 7)
    plt.imshow(im_class_pos_op, cmap=plt.cm.gray)
#spectrum of leaf
avg2 = np.zeros(bands)
i_band = 0
for i_band in range(bands):
    avg2[i_band] = np.sum(np.sum(im_class_pos*im3[:, :, i_band]))/n_pos

plt.subplot(3,3,8)
plt.plot(Wavelength2,avg2/10000,'b')

plt.show()
