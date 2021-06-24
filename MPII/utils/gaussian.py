import numpy as np
import scipy.stats as st
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def gaussian_func(y, mu, sigma):
    #y has the shape of [samples, 1]
    #only for 1d gaussian
    samples = y.shape[0]
    num_tree, leaf_num, _, _ = mu.shape
    #res = torch.zeros(samples, num_tree, leaf_num)
    #print(y.shape)
    y = np.reshape(y, [samples, 1, 1])
    y = np.repeat(y, num_tree, 1)
    y = np.repeat(y, leaf_num, 2)   

    mu = np.reshape(mu, [1, num_tree, leaf_num])
    mu = mu.repeat(samples, 0)

    sigma = np.reshape(sigma, [1, num_tree, leaf_num])
    sigma = sigma.repeat(samples, 0)  

    res = 1.0 / np.sqrt(2 * 3.14 * (sigma + 1e-9)) * \
         (np.exp(- (y - mu) ** 2 / (2 * (sigma + 1e-9))) + 1e-9)

    return res

def multi_gaussian(y, mu, sigma):
    '''
    y has the shape of [samples, task_num]
    mu has the shape of [num_tree, leaf_node_per_tree, task_num, 1]
    sigma has the shape of [num_tree, leaf_node_per_tree, task_num, task_num]
    '''
    samples = y.shape[0]
    num_tree, leaf_num, task_num, _ = mu.shape
    gauss_val = np.zeros((samples, num_tree, leaf_num))
    
    mu = mu.reshape(num_tree, leaf_num, task_num)
    
    for i in range(num_tree):
        for j in range(leaf_num):
            t = st.multivariate_normal.pdf(y, mean=mu[i, j, :], cov=sigma[i, j, :, :], allow_singular=True)
            gauss_val[:, i, j] = t
    return gauss_val

def multi_gaussian_torch(y, mu, sigma):
    '''
    y has the shape of [samples, task_num]
    mu has the shape of [num_tree, leaf_node_per_tree, task_num, 1]
    sigma has the shape of [num_tree, leaf_node_per_tree, task_num, task_num]
    '''
    samples = y.shape[0]
    num_tree, leaf_num, task_num, _ = mu.shape
    gauss_val = np.zeros((samples, num_tree, leaf_num))
    
    mu = mu.reshape(num_tree, leaf_num, task_num)
    
    for i in range(num_tree):
        for j in range(leaf_num):
            # t = st.multivariate_normal.pdf(samples, mean=mu[i, j, :], cov=sigma[i, j, :, :])
            t = MultivariateNormal(mu[i, j, :], sigma[i, j, :, :]).log_prob(y)
            gauss_val[:, i, j] = torch.exp(t)
    return gauss_val

def update(self, x, y):
    num_tree, leaf_num, _, _  = self.mean.shape
    samples, task_num = y.shape
    for i in range(10):
        gaussian_value = gaussian_func(y, self.mean, self.sigma) # [samples, num_tree, leaf_num]
        all_leaf_prob_pi = x * (gaussian_value + 1e-9) # [samples, num_tree, leaf_num]
        # all_leaf_sum_prob = np.sum(all_leaf_prob_pi, axis=2, keepdims=True)  # [samples, num_tree, 1]
        all_leaf_sum_prob = torch.sum(all_leaf_prob_pi, dim=2, keepdim=True) # [samples, num_tree, 1]

        zeta = all_leaf_prob_pi / (all_leaf_sum_prob + 1e-9) # [samples, num_tree, leaf_num]

        # y_temp = np.expand_dims(y, 2) # [samples, task_num, 1]
        # y_temp = np.expand_dims(y_temp, 3) # [samples, task_num, 1, 1]
        y_temp = torch.unsqueeze(y, 2)
        y_temp = torch.unsqueeze(y_temp, 3)
        y_temp = np.repeat(y_temp, num_tree, 2)
        y_temp = np.repeat(y_temp, leaf_num, 3) # [samples, task_num, num_tree, leaf_num]
        zeta = np.expand_dims(zeta, 1).repeat(task_num, 1) # [samples, task_num, num_tree, leaf_num]
        zeta_y = zeta * y_temp # [samples, task_num, num_tree, leaf_num]
        zeta_y = np.sum(zeta_y, 0) # [task_num, num_tree, leaf_num]
        zeta_sum  = np.sum(zeta, 0) # [task_num, num_tree, leaf_num]

        mean = zeta_y / (zeta_sum + 1e-9) # [task_num, num_tree, leaf_num]
        self.mean = mean.transpose(1, 2, 0).reshape(num_tree, leaf_num, task_num, 1)

        mean_new = y_temp - np.expand_dims(mean, 0).repeat(samples, 0) # [samples, task_num, num_tree, leaf_num]
        m1 = np.expand_dims(mean_new.transpose(0, 2, 3, 1), 4) # [samples, num_tree, leaf_num, task_num, 1]
        m2 = np.expand_dims(mean_new.transpose(0, 2, 3, 1), 3) # [samples, num_tree, leaf_num, 1, task_num]
        cov = np.matmul(m1, m2) # [samples, num_tree, leaf_num, task_num, task_num]
        zeta_for_sigma = np.expand_dims(zeta.transpose(0, 2, 3, 1), 4).repeat(task_num, 4) * cov # [samples, num_tree, leaf_num, task_num, task_num]
        zeta_for_sigma = np.sum(zeta_for_sigma, 0) # [num_tree, leaf_num, task_num, task_num]
        zeta_sum = np.expand_dims(zeta_sum.transpose(1,2,0), 3).repeat(task_num, 3) # [num_tree, leaf_num, task_num, task_num]
        sigma = zeta_for_sigma / (zeta_sum + 1e-9)
        self.sigma = sigma

