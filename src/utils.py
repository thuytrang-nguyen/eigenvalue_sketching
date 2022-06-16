import numpy as np

# A very fast distance function

def get_distance(X, Y):
    """
    fast euclidean distance computation:
    X shape is n x features
    """
    num_test = X.shape[0]
    num_train = Y.shape[0]
    dists = np.zeros((num_test, num_train))
    sum1 = np.sum(np.power(X,2), axis=1)
    sum2 = np.sum(np.power(Y,2), axis=1)
    sum3 = 2*np.dot(X, Y.T)
    dists = sum1.reshape(-1,1) + sum2
    dists = np.sqrt(dists - sum3)
    dists = dists / np.max(dists)
    return dists

def GP(max_val):
	vals = []
	i = 1
	while i < max_val:
		vals.append(i)
		i = i*2
	return vals

def generate_gp_series(max_val):
	max_val = int(max_val)
	vals = GP(max_val)
	start  = 0
	vec_vals = np.zeros(max_val)
	for i in range(len(vals)):
		end = vals[i]+start
		vec_vals[start:end] = 1.0/vals[i]
		start = vals[i]+start
	return vec_vals
