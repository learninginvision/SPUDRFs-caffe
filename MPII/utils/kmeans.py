
from sklearn.cluster import KMeans
import operator
from functools import reduce
import numpy as np 
import scipy.stats as st

def kmeans(x, n_cluster=5):
    '''
    x has shape [samples , fdim]
    return 
    mean has shape [n_cluster, fdim]
    sigma has shape [n_cluster, fdim fdim]
    '''
    samples, fdim = x.shape
    km = KMeans(n_clusters=n_cluster)
    km.fit(x)
    sigma = np.zeros([n_cluster, fdim, fdim])
    data_cluster = [[] for i in range(n_cluster)]
    for idx, label in enumerate(km.labels_):
        data_cluster[label].append(x[idx])
    for i in range(n_cluster):
        data = np.array(data_cluster[i]).T
        sigma[i, :, :] = np.cov(data) 
    mean = km.cluster_centers_.reshape(n_cluster, fdim)
    return mean, sigma

if __name__ == "__main__":
    import numpy as np 
    
    x = np.random.rand(10000, 2).reshape(-1,2)
    mean, sigma = kmeans(x, 5)
    mean = np.expand_dims(mean, 2)
    print(mean.shape, sigma.shape)
    # print(sigma[0, :, :])
    # gauss_val = st.multivariate_normal.pdf(np.array([0, 0]), mean=mean[0, :], cov=sigma[0, :, :])
    # print(gauss_val)
