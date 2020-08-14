from sklearn.mixture import GaussianMixture
from sklearn import mixture
import  numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection


samples = 1024
bands = 422
lines = 1021
g = open('REFLECTANCE_Tomatoes_Friday.dat', 'rb')
im = np.fromfile(g, np.uint16,(samples * bands * lines))
g.close()
im2 = np.reshape(im, [samples, bands, lines], order="F")
im2=np.transpose(im2, [0, 2, 1])
im3 = np.array(im2[249:600,449:950,:])
bands = im3.shape[2]
samples = im3.shape[0]
lines = im3.shape[1]
pas = 5
spl_im2 = np.array(im3[0:samples:10, 0:lines:10, 0:bands:pas])
print("im2", np.shape(im2))
Yc = np.array(np.reshape(spl_im2, [spl_im2.shape[0]*spl_im2.shape[1], spl_im2.shape[2]], order="F"))
############
X1 = np.random.multivariate_normal([0.,0.],[[1.,0.],[0.,1.]],10000)
X2 = np.random.multivariate_normal([10.,10.],[[1.,0.],[0.,1.]],10000)
X = np.vstack((X1,X2))
np.random.shuffle(X)
###############

param_grid = {'n_components': np.array([3,4,5])}
score = 'accuracy'
#construct grid search object that uses 2 fold cross validation
clf = model_selection.GridSearchCV(estimator=mixture.GaussianMixture(),

    param_grid=param_grid,cv=2, # nombre de folds de validation croisee
    )
print("___CLF__", clf)
print(clf.fit(Yc))
#print(clf.fit(X, np.ravel(target, order='F')))
print("Meilleur(s) hyperparametre(s) sur le jeu d entrainement:", clf.best_params_)

AA = clf.best_params_
for mean, std, params in zip(clf.cv_results_['mean_test_score'],  # score moyen

                             clf.cv_results_['std_test_score'],  # écart-type du score

                             clf.cv_results_['params']  # valeur de l'hyperparamètre
                             ):
    print("\t%s = %10f (+/-%0.1f) for %r" % (score,  # critère utilisé
                                             mean,
                                             # score moyen sur tous les SEP de test dont le nombre est preske egale à 10 fois nbre de pattern de SEP et control after centrality
                                             std * 2,  # barre d'erreur

                                             params  # hyperparamètre
                                             ))

