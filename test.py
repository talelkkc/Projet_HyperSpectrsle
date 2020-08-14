import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from spectrum import __Wavelength__
from sklearn.cluster import AgglomerativeClustering
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from detect_vegetation import __detect_vegetation__
samples = 1024
bands = 422
lines = 1021
#g = open('REFLECTANCE_Tomatoas_Monday.dat', 'rb')
g = open('refl_no_4300.dat', 'rb')
# im1= g.readline(samples*bands*lines)
im = np.fromfile(g, np.uint16,(samples * bands * lines))
#print("mmmm", np.size(im))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2,[0,2,1])
print("HHHHHHHH",np.shape(im2))
X=np.array(im2[739:820,629:690,:])
b=np.shape(X)
print("llllll",X)
print("***mm****",b)
plt.subplot(2,3,1)
plt.imshow(im2[250:600,450:950,234],interpolation='spline36',cmap='viridis',norm=LogNorm())
#A=plt.imshow(X[:,:,234],interpolation='gaussian',cmap='viridis',norm=LogNorm())
#plt.show(A)
#https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/interpolation_methods.html#sphx-glr-gallery-images-contours-and-fields-interpolation-methods-py
avg_X = np.squeeze(np.mean(np.mean(X, axis=1), axis=0))
print("******",np.shape(avg_X))
Wavelength=__Wavelength__()
Wavelength2=Wavelength[26:448]
plt.subplot(2,3,2)
plt.plot(Wavelength2,avg_X /10000,'b')
#plt.show(C)
##********************************##
#morph=9
#nclass = 4
#avg1=__detect_vegetation__(im2,Wavelength,morph,nclass)

nclass = 4
#bands=im2.shape[2]
## GMM
#https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
#https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.mixture.GMM.html
#https://gemfury.com/stream/python:scikit-learn/-/content/mixture/gmm.py
#https://pythonmachinelearning.pro/clustering-with-gaussian-mixture-models/
im3=im2[250:600,450:950,:]
bands = im3.shape[2]
samples = im3.shape[0]
lines = im3.shape[1]
pas = 5
Wv_pas = Wavelength2[0:bands:pas]
spl_im2 = im3[0:samples:10, 0:lines:10, 0:bands:pas]
Yc = np.reshape(spl_im2, [spl_im2.shape[0]*spl_im2.shape[1],spl_im2.shape[2]], order="F")
print("***__________***",Yc)
print("---__________---",np.shape(Yc))
GMModel = mixture.GaussianMixture(n_components=nclass,covariance_type='full',max_iter=50,n_init=20).fit(Yc)
print("_____***____",GMModel)
OP=GMModel.fit(Yc,100)
print("_________",np.size(OP))
###########
AA=GMModel.score(Yc)
print("_________###",AA)
###########

im22=im3[:,:,1:bands:pas]
Q=np.reshape(im3[:,:,1:bands:5],[im3.shape[0]*im3.shape[1],im22.shape[2]], order="F")
prediction_gmm = GMModel.predict(Q)
IM=np.reshape(prediction_gmm,[samples,lines],order="F")
print("____prediction_gmm___",prediction_gmm)
print("____**prediction_gmm**___",np.shape(prediction_gmm))
plt.subplot(2,3,3)
FF=plt.imshow(IM,interpolation='mitchell',cmap='viridis',norm=LogNorm())
#plt.show(FF)
#################
probs = GMModel.predict_proba(Yc)
print("____probs___",np.size(probs))
print("____**probs**___",np.shape(probs))

##############
#Find vegetation cluster
red = 650
near = 850
p_red = np.size(np.argwhere(Wv_pas<red))
p_near  = np.size(np.argwhere(Wv_pas<near))
#
HH=GMModel.means_[:,p_red]
EE=GMModel.means_[:,p_near]
#
NDVI = (EE-HH)/(EE+HH)
print("__NDVI__",NDVI)
print("__size_NDVI__",np.size(NDVI))


pos2 = NDVI.argmax() #index
print("maxxx0",pos2)
#############
im_class_pos= (IM==pos2)*1
print("__im_class_pos__",im_class_pos)
print("__size_im_class_pos__",np.shape(im_class_pos))
im_class_pos1=np.equal(IM,pos2)
n_pos=np.sum(np.sum(im_class_pos))
print("__n_pos__",n_pos)
print("__size_n_pos__",np.shape(n_pos))
plt.subplot(2,3,4)
plt.imshow(im_class_pos)
#plt.show(PP)
#imopen
morph=9
if (morph>0):
    se = disk(4)
    print("******", se)
    im_class_pos_op = opening(im_class_pos, se)
    #https://scikit-image.org/docs/0.13.x/auto_examples/xx_applications/plot_morphology.html
    plt.subplot(2, 3, 5)
    TT = plt.imshow(im_class_pos_op,cmap=plt.cm.gray)
    #plt.show(TT)

#spectrum of leaf
avg2 = np.zeros(bands)
i_band =0
for i_band in range(bands):
    avg2[i_band]=np.sum(np.sum(im_class_pos*im3[:,:,i_band]))/n_pos



print("__avg2__",avg2)
print("__avg2__",np.shape(avg2))
plt.subplot(2,3,6)
plt.plot(Wavelength2,avg2/10000,'b')
plt.show()