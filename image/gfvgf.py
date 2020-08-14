import numpy as np
from sklearn import mixture
from sklearn import mixture
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
""""
A=np.random.seed(1)
print("____**____",A)
g =mixture.GaussianMixture(n_components=2)
print("*****",g)
obs = np.concatenate((np.random.randn(100, 1),10 + np.random.randn(300, 1)))
AA=g.fit(obs)
print("*****",np.shape(AA))

"""

avg2 = np.zeros(448)
print("",avg2)
print("___**__",np.size(avg2))
im2=np.transpose(avg2)
print("",im2)
print("___*im2*__",np.size(im2))

se = disk(2)
print("hahahaha",se)


