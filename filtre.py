import numpy as np
import matplotlib.pyplot as plt
def __Filtre__(im4):
    A = im4.shape[0]
    B = im4.shape[1]
    C = im4.shape[2]
    M = im4
    M_ftr = M
    F = 0
    for i in range(0, A):
        for j in range(0, B):
            x = max(M[i, j, :])
            if (x > 10000) & (F < 500000000):
                F = F + 1
                if (M_ftr[i, j, 0] > (M_ftr[i, j, 1] + 2000)) | (M_ftr[i, j, 0] > 10000):
                    M_ftr[i, j, 0] = min(M_ftr[i, j, 1], 1000)
                if (M_ftr[i, j, C - 1] > M_ftr[i, j, C - 2] + 2000) | (M_ftr[i, j, C - 1] > 10000):
                    M_ftr[i, j, C - 1] = min(M_ftr[i, j, C - 2], 1000)
                for k in range(0, C - 1):
                    if (M_ftr[i, j, k] > (M_ftr[i, j, k - 1] + 2000)) | (M_ftr[i, j, k] > (M_ftr[i, j, k + 1] + 2000)):
                        M_ftr[i, j, k] = min(M_ftr[i, j, k - 1], M_ftr[i, j, k + 1])
                """
                if max(M_ftr[i, j, :]) > 10000:
                    if F > 0:
                        plt.subplot(3, 3, 1)
                        plt.title(" Avant** ")
                        plt.plot(M[i, j, :])
                    if F > 0:
                        plt.subplot(3, 3, 2)
                        plt.title(" Après ")
                        plt.plot(M_ftr[i, j, :])
                        """

    return (M_ftr)