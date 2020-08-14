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
from filtre import __Filtre__
from Norm_ftr import __norm_Ftr__
from sklearn import model_selection

def __detect_vegetation__(im0,Wavelength,morph,nclass):
    im1 = im0
    im3 = __Filtre__(im0)
    im2 = __norm_Ftr__(im3, Wavelength)
    bands = im2.shape[2]
    samples = im2.shape[0]
    lines = im2.shape[1]
    pas = 5
    Wv_pas = Wavelength[0:bands:pas]
    spl_im2 = im2[0:samples:10, 0:lines:10, 0:bands:pas]
    Yc = np.reshape(spl_im2, [spl_im2.shape[0]*spl_im2.shape[1], spl_im2.shape[2]], order="F")
    param_grid = {'n_components': np.array([3, 4, 5])}
    clf = model_selection.GridSearchCV(estimator=mixture.GaussianMixture(), param_grid=param_grid, cv=2).fit(Yc)
    QQQ = list(clf.best_params_.values())[0]
    print("__QQQ__",QQQ)
    GMModel = mixture.GaussianMixture(n_components=QQQ, covariance_type='full', max_iter=100, n_init=20).fit(Yc)
    im22 = im2[:, :, 0:bands:pas]
    Q = np.reshape(im2[:, :, 0:bands:5], [im2.shape[0] * im2.shape[1], im22.shape[2]], order="F")
    prediction_gmm = GMModel.predict(Q)
    IM = np.reshape(prediction_gmm, [im2.shape[0], im2.shape[1]], order="F")
    # Find vegetation cluster
    red = 650
    near = 850
    p_red = np.size(np.argwhere(Wv_pas < red))
    p_near = np.size(np.argwhere(Wv_pas < near))
    HH = GMModel.means_[:, p_red]
    EE = GMModel.means_[:, p_near]
    NDVI = (EE - HH) / (EE + HH)
    pos2 = NDVI.argmax()
    im_class_pos = (IM == pos2)*1
    n_pos = np.sum(np.sum(im_class_pos))
    # imopen
    if (morph > 0):
        se = disk((morph-1)/2)
        se1 = disk((morph-1)/4)
        im_class_pos_op = closing(im_class_pos, se)
        im_class_pos_op = opening(im_class_pos_op, se1)
        ##
    # spectrum of leaf
    avg2 = np.zeros(bands)
    avg22 = np.zeros(bands)
    for i_band in range(0,bands):
        avg2[i_band] = np.sum(np.sum(im_class_pos * im2[:, :, i_band])) / n_pos
        avg22[i_band] = np.sum(np.sum(im_class_pos_op * im1[:, :, i_band])) / n_pos
    return (avg22,avg2,IM,im_class_pos,im_class_pos_op)





