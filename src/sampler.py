import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

def sample_eig(data, s, similarity_measure, scale=False, rankcheck=0):
    """
    input: original matrix
    output: sample eigenvalue
    if using some function to compute elements, this function allows
    you to run the code without instantiating the whole matrix
    """
    n = len(data)
    list_of_available_indices = range(n)
    sample_indices = np.sort(random.sample(list_of_available_indices, s))
    subsample_matrix = similarity_measure(data[sample_indices,:], data[sample_indices,:])
    # useful for only hermitian matrices
    
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    # use sparse instead
    #ssm_sparse = csr_matrix(subsample_matrix)
    #all_eig_val_estimates_t, eig_vec = eigs(ssm_sparse)
    #all_eig_val_estimates = np.real(np.asarray(all_eig_val_estimates_t))
    #all_eig_val_estimates.sort()
  
    min_eig = np.array(all_eig_val_estimates)[rankcheck]

    if scale == False:
        return min_eig
    else:
        return n*min_eig/float(s)
        
# The eigenvalue estimator
def sample_eig_default(data_matrix, s, scale=False, \
                        rankcheck=0, norm=[], nnzA=0, method="uniform random sample", multiplier=0.1):
    """
    input: original matrix
    output: sample eigenvalue
    requires data matrix to be fully instantiated
    """
    # set multiplier to 0.1

    n = len(data_matrix)
    list_of_available_indices = range(n)

    sample_indices = np.sort(np.random.choice(list_of_available_indices, \
        size=s, replace=True, p=norm))
    chosen_p = norm[sample_indices]
    
    subsample_matrix = data_matrix[sample_indices][:, sample_indices]

    sqrt_chosen_p = np.sqrt(chosen_p*s)
    D = np.diag(1 / sqrt_chosen_p)
    subsample_matrix = D @ subsample_matrix @ D        

    if "sparsity sampler" in method:
        original_nnzs = np.count_nonzero(subsample_matrix)
        subsample_matrix = subsample_matrix - np.diag(np.diag(subsample_matrix))
        pipj = np.outer(chosen_p, chosen_p)
        mask = (pipj >= 1/(s*multiplier*nnzA)).astype(int) # assuming s \geq tilde{O}(1/epsilon**2)
        subsample_matrix = subsample_matrix*mask
        nnz_subsample_matrix = np.count_nonzero(subsample_matrix)
        try:
            nnz_subsample_matrix = (nnz_subsample_matrix) / float(original_nnzs)
        except:
            nnz_subsample_matrix = 0
    
    # useful for only hermitian matrices
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(subsample_matrix))
    all_eig_val_estimates.sort()

    # use sparse instead
    #ssm_sparse = csr_matrix(subsample_matrix)
    #all_eig_val_estimates_t, eig_vec = eigs(ssm_sparse)
    #all_eig_val_estimates = np.real(np.asarray(all_eig_val_estimates_t))
    #print(all_eig_val_estimates)
    #all_eig_val_estimates.sort()


    min_eig = np.array(all_eig_val_estimates)[rankcheck]
    if scale == False or "CUR" in mode:
        if "sparsity sampler" in method:
            return min_eig, nnz_subsample_matrix
        else:
            return min_eig
    else:
        return n*min_eig/float(s)

def countSketch(data_matrix, s, flag=0, rankcheck=0, return_type="sketch"):
    n = len(data_matrix)
    tr = np.trace(data_matrix)
    tr_m = np.eye(n)*tr

    if s >= n:
        return 

    h_buckets = np.random.choice(np.arange(s), size=n)

    h_signs = np.random.choice([-1, 1], size=n)

    sketch_m = np.zeros((s,n))

    for i in range(n):
       sketch_m[h_buckets[i],i]=h_signs[i]

    if return_type == "sketch":
       return sketch_m
    
    sketch = np.matmul(sketch_m, np.matmul(data_matrix-tr_m, np.transpose(sketch_m)))
    if flag == 1:
        sketch = np.matmul(sketch_m, np.matmul(data_matrix, np.transpose(sketch_m)))

    #sketch_sparse = csr_matrix(sketch)

    #all_eig_val_estimates = np.real(np.linalg.eigvalsh(sketch))
    #all_eig_val_estimates_t, eig_vec = eigs(sketch_sparse)
    #all_eig_val_estimates = np.real(np.asarray(all_eig_val_estimates_t))
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(sketch))
    all_eig_val_estimates.sort()

    min_eig = np.array(all_eig_val_estimates)[rankcheck]

    return(min_eig)

def denseSketch(data_matrix, s,flag=0, rankcheck=0):

    n = len(data_matrix)
    tr = np.trace(data_matrix)
    tr_m = np.eye(n)*tr
    if s >= n:
        return 

    sketch_m = np.random.choice([-1,1], size=(s,n))
    sketch_m = (1/np.sqrt(s))*sketch_m

    #sketch = np.matmul(sketch_m, np.matmul(data_matrix, np.transpose(sketch_m)))
    sketch = np.matmul(sketch_m, np.matmul(data_matrix-tr_m, np.transpose(sketch_m)))
    if flag == 1:
        sketch = np.matmul(sketch_m, np.matmul(data_matrix, np.transpose(sketch_m)))
    #sketch_sparse = csr_matrix(sketch)

    #all_eig_val_estimates = np.real(np.linalg.eigvalsh(sketch))
    #all_eig_val_estimates_t, eig_vec = eigs(sketch_sparse)
    #all_eig_val_estimates = np.real(np.asarray(all_eig_val_estimates_t))
    #all_eig_val_estimates.sort()
    all_eig_val_estimates = np.real(np.linalg.eigvalsh(sketch))
    all_eig_val_estimates.sort()
    min_eig = np.array(all_eig_val_estimates)[rankcheck]

    return(min_eig)

def hybrid(data_matrix, s2, norm, rankcheck=0, nnzA=0, ratio=10, flag=0, multiplier=0.1):

    n = len(data_matrix)
    
    if s2 >= n:
        return 

    s1 = int(min(n, ratio*s2))
  
    # uniform sample of size s1
    list_of_available_indices = range(n)

    sample_indices = np.sort(np.random.choice(list_of_available_indices, size=s1, replace=True, p=norm))
    chosen_p = norm[sample_indices]
    
    subsample_matrix = data_matrix[sample_indices][:, sample_indices]
    
    sqrt_chosen_p = np.sqrt(chosen_p*s1)
    D = np.diag(1 / sqrt_chosen_p)
    subsample_matrix = D @ subsample_matrix @ D  

    # Use sparsity sampling 
    original_nnzs = np.count_nonzero(subsample_matrix)
    subsample_matrix = subsample_matrix - np.diag(np.diag(subsample_matrix))
    pipj = np.outer(chosen_p, chosen_p)
    mask = (pipj >= 1/(s1*multiplier*nnzA)).astype(int) # assuming s \geq tilde{O}(1/epsilon**2)
    subsample_matrix = subsample_matrix*mask
    #nnz_subsample_matrix = np.count_nonzero(subsample_matrix)
    #nnz_subsample_matrix = (nnz_subsample_matrix) / float(original_nnzs)

    # sketch of size s2
    tr = np.trace(subsample_matrix)
    tr_m = np.eye(len(subsample_matrix))*tr
    h_buckets = np.random.choice(np.arange(s2), size=s1)
    h_signs = np.random.choice([-1, 1], size=s1)

    sketch_m = np.zeros((s2,s1))

    for i in range(s1):
       sketch_m[h_buckets[i],i]=h_signs[i]

    sketch = np.matmul(sketch_m, np.matmul(subsample_matrix-tr_m, np.transpose(sketch_m)))
    if flag==1:
        sketch = np.matmul(sketch_m, np.matmul(subsample_matrix, np.transpose(sketch_m)))

    all_eig_val_estimates = np.real(np.linalg.eigvalsh(sketch))
    all_eig_val_estimates.sort()

    '''
    sketch_sparse = csr_matrix(sketch)

    #all_eig_val_estimates = np.real(np.linalg.eigvalsh(sketch))
    all_eig_val_estimates_t, eig_vec = eigs(sketch_sparse)
    all_eig_val_estimates = np.real(np.asarray(all_eig_val_estimates_t))
    all_eig_val_estimates.sort()
    '''
    
    min_eig = np.array(all_eig_val_estimates)[rankcheck]

    return(min_eig)
