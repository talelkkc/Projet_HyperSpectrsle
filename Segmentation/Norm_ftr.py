import numpy as np
def __norm_Ftr__(im4,Wavelength):
    near1 = 850
    A = im4.shape[0]
    B = im4.shape[1]
    Wv_pas1 = np.array(Wavelength[:])
    p_near1 = np.size(np.argwhere(Wv_pas1 < near1))
    M_ftr = im4
    V = 4000
    for i in range(0, A):
        for j in range(0, B):
            if (M_ftr[i, j, p_near1] > 1000):
                M_ftr[i, j, :] = (M_ftr[i, j, :] / M_ftr[i, j, p_near1]) * V

    return (M_ftr)
