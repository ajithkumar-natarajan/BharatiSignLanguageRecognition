import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import signal
from tqdm import tqdm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from GaussianStatistics import *
from configure import Config
from keras.datasets import mnist
import random


Gstat  = GaussianStatistics()
config = Config()

def nstnbridx(output_size,weights,row_data):
        initial_dis = float("inf")
        index_bmu   = [0, 0]
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                dist_neuron = np.linalg.norm(row_data-weights[i, j, :])
                if dist_neuron < initial_dis:
                    initial_dis = dist_neuron
                    index_bmu = [i, j]# Best matching unit (BMU)
        return np.array(index_bmu)

def nbridxdis(nrad,output_size,index_bmu):
    nbrind = np.zeros(((2*nrad + 1)**2, 2))
    for i in range(2*nrad + 1):
        for j in range(2*nrad + 1):
            ix = i*(2*nrad+1) + j
            [i - nrad, j- nrad] + index_bmu
            nbrind[ix,:] = [i - nrad, j- nrad] + index_bmu

    # print (nbrind, (nbrind[:,1] >= 0))
    nbrind = nbrind[np.where((nbrind[:,0] >= 0) * (nbrind[:,1] >= 0))]
    nbrind = nbrind[np.where((nbrind[:,0] < output_size[0]) * (nbrind[:,1] < output_size[1]))]

    mm, _  = nbrind.shape
    nbrdist = np.zeros(mm)
    for i in range(mm):
        diff = nbrind[i,:] - index_bmu
        nbrdist[i] = diff.dot(diff)
    return nbrind, nbrdist
    
def response(output_size, X, wt, sig = 20):
    """
    """
    x = X.flatten('F')
    assert len(x) == wt.shape[2]
    Y = np.zeros(output_size)

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            diff = wt[i, j, :] - x
            dis  = diff.dot(diff)
            Y[i, j] = np.exp(-1.*dis / sig**2)
    return Y

def Normalize(mat):
    mat = mat / np.sum(abs(mat))
#    mat = (mat - np.min(mat))/ (np.max(mat) - np.min(mat))
    return mat
    
coord=np.load('Coordinates_labels.npy')
coordinates=coord[:,0:2]
labels=coord[:,2]
positions=[]
for i in range(3):
    for j in range(3):
        positions.append([i,j])
     
#data=[]
#for XX in range(4,201,4):
#    for YY in range(6,301,6):
#        coord=[XX,YY]
#        data.append(coord)
#data=np.array(data)
#coordinates=coord[:,0:2]
#np.save('Coordinates.npy', data)
#data = (data - np.min(data))/ (np.max(data) - np.min(data))

train_flag=True
test_flag=True

if train_flag:
    epochs=1000
    iterations = len(coordinates)*epochs
    output_size = [3,3]
    input_num = coordinates.shape[1]
    nrad  =1
    sig   = 3
    alpha = 0.05
    weights = np.random.rand(output_size[0], output_size[1], input_num)
        
    for itter in tqdm(range(iterations)):
        initial_dis = float("inf")
#        row_index = np.random.randint(len(data))
        row_index = np.random.randint(len(coordinates))
        learning_rate = alpha*np.exp(-itter/iterations)
#        row_data = data[row_index]
        row_data = coordinates[row_index]
#        bmu_idx  = nstnbridx(output_size,weights,row_data)
        bmu_idx = np.array(positions[int(labels[row_index]-1)])
        nbrind, nbrdist = nbridxdis(nrad,output_size,bmu_idx)
        mx, _ = nbrind.shape
        for i in range(mx):
            idx = nbrind[i,:]
            wt  = weights[int(idx[0]), int(idx[1]), :]
            diff = row_data - wt
            dis  = nbrdist[i]/sig **2
            delta = learning_rate*np.exp(-dis)*diff
            weights[int(idx[0]), int(idx[1]), :] = delta + wt
#            weights=Normalize(weights)
    print ("SOM Training done!!...")
    np.save(config.SOM_weights_path, weights)

if test_flag:
    display = True
    weights = np.load(config.SOM_weights_path)
#    for x in data:
#        N = np.sqrt(len(x))
#        X = x.reshape(2, order='F')
    X=np.array([37,201])
    X = X.reshape(2, order='F')
    y = response(output_size,X, weights)
    if display:
        plt.imshow(y)
        plt.show()
