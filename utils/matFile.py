import scipy.io as sio
# import matplotlib.pyplot as plt
import numpy as np
import os

# matlab文件名
matfn = u'/home/llj0571/Desktop/dataset/svhn/digitStruct.mat'
# import scipy.io as sio
#
# # reading(loading) mat file as array
# loaded_mat = sio.loadmat(matfn)
#
# data = loaded_mat['X']
#
# labels = loaded_mat['y'].astype(np.int64).squeeze()
# data = np.transpose(data, (3, 2, 0, 1))

import h5py
data=h5py.File(matfn,'r')
x=list(data.keys())
print(x)

y = data.values()
print(y)
print(data['#refs#'])
print(data['digitStruct'])



# www=data['#refs#']
# qqq = data['digitStruct']
# print(qqq)

