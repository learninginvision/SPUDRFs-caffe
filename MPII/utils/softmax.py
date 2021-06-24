import numpy as np 

def softmax(x):
    '''
    : param
    x: ndarray
    x -> ntree, nleaf, N
    : return
    softmax(x, axis=2)
    '''
    x_max = x.max(axis=2)
    x_max = x_max.reshape(list(x.shape)[:-1]+[1])
    # print(x_max.shape)
    x = x - x_max
    x = np.exp(x)
    x_sum = x.sum(axis=2, keepdims=True)
    return x / x_sum

if __name__ == '__main__':
    x = np.random.randn(2, 4, 3)
    y = softmax(x)
    print(y)