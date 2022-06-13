
"""
create/get dataset
"""

import skimage.io
from skimage import feature
import numpy as np
from sklearn.preprocessing import normalize
from src.display_codes import display_image
import matplotlib.pyplot as plt
from random import sample
import os

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

def get_data(name, eps=0.1, plot_mat=True, raise_eps=False):
    if name == "rand_sym":
        dataset_size = 5000 
        d = np.random.choice([-1,1], size=(dataset_size,dataset_size))

        u_tr = np.triu(d)

        diag = np.diag(np.diag(d))

        xy = u_tr+(u_tr-diag).T

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        #print(xy)
        
        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "diag_1":
        num_big = 500 
        dataset_size = 5000 
        diag = np.full(5000,1)
        big_entries = np.random.choice(range(dataset_size), num_big-1)
        #print(big_entries)
        diag[0] = 1000000
        for j in big_entries:
            diag[j] == 1000000
        xy = np.diag(diag)

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        #print(xy)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "diag_power":
        dataset_size = 5000
        diag = np.arange(1,2501,0.5,dtype=float) #np.arange(1,10000,5000)
        #p_law = lambda x: 2*(x**(-2))
        p_law = lambda x: 2*(x**(-4))
        diag_p = p_law(diag)

        xy = np.diag(diag_p)

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        #print(xy)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "rand_fewbig":

        num_big = 500
        dataset_size = 5000
        #d = np.random.uniform(-1,1,(dataset_size, dataset_size))
        d=np.random.choice([-1,1], size=(dataset_size,dataset_size))


        e1 = np.random.choice(range(dataset_size), num_big)
        e2 = np.random.choice(range(dataset_size), num_big)

        for i in range(len(e1)):
            d[(e1[i])][(e2[i])] == np.random.uniform(100000,1000000)
        
        u_tr = np.triu(d)

        diag = np.diag(np.diag(d))

        xy = u_tr+(u_tr-diag).T


        #xy = (d+d.T)/2

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        #print(xy)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "kong":
        imagedrawing = skimage.io.imread('donkeykong.tn768.png')
        display_image(imagedrawing)
        dataset_size = 5000
        edges = imagedrawing
        xy = np.stack(np.where(edges == 0), axis=1)
        n_samples = dataset_size
        xy_sampled_idxs = np.random.randint(low=0, high=xy.shape[0], size=n_samples)
        xy = xy[xy_sampled_idxs, :]
        xy[:,0] = -xy[:,0]
        y_min = np.min(xy[:,0])
        xy[:,0] = xy[:,0]-y_min
        xy = xy.astype(np.float)
        xy[:, 0] = xy[:,0] / np.max(xy[:,0])
        xy[:, 1] = xy[:,1] / np.max(xy[:,1])

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "asymmetric":
        """
        experiments with asymmetrics 
        """
        dataset_size = 5000
        xy = np.random.random((dataset_size, dataset_size))

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return xy, dataset_size, min_sample_size, max_sample_size

    if name == "binary":
        """
        mimics lower bound code
        """
        dataset_size = 5000
        c = 0.50
        A = np.zeros((dataset_size, dataset_size))
        ind = np.random.choice(range(dataset_size), size=int(dataset_size*c), replace=False)
        A[ind, ind] = -1

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "random_sparse":
        """
        mimics test.m code
        """
        dataset_size = 5000
        A = np.random.random((dataset_size, dataset_size))
        A = A>0.99
        A = A.astype(int)
        A = np.triu(A) + np.triu(A).T

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "block":
        """
        uses a matrix with n/2 x n/2 block of all 1s and rest zeros
        """
        dataset_size = 5000
        A = np.zeros((dataset_size, dataset_size))
        B = np.ones((int(dataset_size/2), int(dataset_size/2)))
        A[0:len(B), 0:len(B)] = B

        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)

        return A, dataset_size, min_sample_size, max_sample_size

    if name == "deezer":
        """
        dataset: https://snap.stanford.edu/data/deezer_ego_nets.html
        """
        import pandas as pd
        import networkx as nx
        df = pd.read_csv('data/deezer_ego_nets.csv')
        G = nx.from_pandas_edgelist(df, "id_1", "id_2")
        A = nx.adjacency_matrix(G)
        A = A.todense()

        return A, len(A), 50, 2000

    if name == "MNIST":
        """
        get MNIST dataset from: http://yann.lecun.com/exdb/mnist/
        dataset: MNIST 10k
        do pip install idx2numpy prior to this
        """
        import idx2numpy
        file = 'data/t10k-images.idx3-ubyte'
        arr = idx2numpy.convert_from_file(file)
        # reshape images to vectors: the following line leads to array of size 10k x 784
        arr = arr.reshape(arr.shape[0], arr.shape[1]*arr.shape[2])
        A = get_distance(arr, arr)
        return A, len(A), 50, 2000

    if name == "arxiv" or name == "facebook" or name == "erdos":
        """
        dataset arxiv: https://snap.stanford.edu/data/ca-CondMat.html
        dataset facebook: https://snap.stanford.edu/data/ego-Facebook.html
        """
        if name == "arxiv":
            data_file = "./data/CA-CondMat.txt"
        if name == "facebook":
            data_file = "./data/facebook_combined.txt"    
        import networkx as nx
        if name == "erdos":
            from networkx.generators.random_graphs import erdos_renyi_graph
            g = erdos_renyi_graph(5000, p=0.1)
        else:
            g = nx.read_edgelist(data_file,create_using=nx.DiGraph(), nodetype = int)
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.asarray(A)
        if name == "facebook":
            A = A+A.T # symmetrizing as the original dataset is directed

        dataset_size = len(A)
        
        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)
        
        return A, dataset_size, min_sample_size, max_sample_size

    if name == "multi_block_synthetic":
        n = 5000

        A = np.ones((n,n))
        if raise_eps == False:
            num_blocks = round(1/(eps))
            sample_block_sizes = round((eps)*n)
        else:
            num_blocks = round(1/(eps**2))
            sample_block_sizes = round((eps**2)*n)

        R = np.zeros_like(A)
        Z = [-1, 1]

        sample_block = np.ones((sample_block_sizes, sample_block_sizes))

        block_start_row = []
        block_end_row = []
        block_start_col = []
        block_end_col = []
        start_row = 0
        start_col = 0
        for i in range(num_blocks):
            block_start_row.append(start_row)
            block_start_col.append(start_col)
            start_row += sample_block_sizes
            start_col += sample_block_sizes
            block_end_row.append(start_row)
            block_end_col.append(start_col)


        row_id = 0
        for i in range(num_blocks):
            col_id = 0

            for j in range(num_blocks):
                try:
                    q = int(np.unique(R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]])[-1])
                except:
                    q = 0

                if q == 0:
                    flag = sample(Z,1)[-1]
                    try:
                        R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]] \
                                    = sample_block
                    except:
                        pass
                    try:
                        R[block_start_row[j]:block_end_row[j], block_start_col[i]:block_end_col[i]] \
                                    = sample_block.T
                    except:
                        pass
        A = A+R
        
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # save figures
        if plot_mat:
            foldername = "figures/matrices/"
            if not os.path.isdir(foldername):
                os.makedirs(foldername)
            plt.imshow(A)
            plt.colorbar()
            plt.savefig(foldername+name+"_matrix.pdf")
            plt.clf()

            plt.scatter(range(n), eigvals, alpha=0.3, marker='o', s=2, edgecolors=None)
            plt.savefig(foldername+name+"_eigvals.pdf")
            plt.clf()

            V = np.abs(eigvecs)
            plt.imshow(V)
            plt.colorbar()
            plt.savefig(foldername+name+"_eigvecs.pdf")
            plt.clf()

            SIP = V.T @ V
            plt.imshow(SIP)
            plt.colorbar()
            plt.savefig(foldername+name+"_abs_IP.pdf")
            plt.clf()

        return A, n, int(n/100), int(n/5)

    if name == "multi_block_outer":
        n = 2000

        A = np.ones((n,n))
        
        if raise_eps == False:
            num_blocks = round(1/(eps))
            sample_block_sizes = round((eps)*n)
        else:
            num_blocks = round(1/(eps**2))
            sample_block_sizes = round((eps**2)*n)


        R = np.zeros_like(A)
        Z = [-1, 1]

        set_val = np.array([-1,1])

        block_start_row = []
        block_end_row = []
        block_start_col = []
        block_end_col = []
        start_row = 0
        start_col = 0
        for i in range(num_blocks):
            block_start_row.append(start_row)
            block_start_col.append(start_col)
            start_row += sample_block_sizes
            start_col += sample_block_sizes
            block_end_row.append(start_row)
            block_end_col.append(start_col)


        row_id = 0
        for i in range(num_blocks):
            col_id = 0

            for j in range(num_blocks):
                try:
                    q = int(np.unique(R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]])[-1])
                except:
                    q = 0

                if q == 0:
                    vec1 = np.random.choice(set_val, size=sample_block_sizes)
                    vec2 = np.random.choice(set_val, size=sample_block_sizes)
                    sample_block = np.outer(vec1, vec2)
                    try:
                        R[block_start_row[i]:block_end_row[i], block_start_col[j]:block_end_col[j]] \
                                    = sample_block
                    except:
                        pass
                    try:
                        R[block_start_row[j]:block_end_row[j], block_start_col[i]:block_end_col[i]] \
                                    = sample_block.T
                    except:
                        pass
        A = A+R
        
        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # save figures
        if plot_mat:
            foldername = "figures/matrices/"
            if not os.path.isdir(foldername):
                os.makedirs(foldername)
            plt.imshow(A)
            plt.colorbar()
            plt.savefig(foldername+name+"_matrix.pdf")
            plt.clf()

            plt.scatter(range(n), eigvals, alpha=0.3, marker='o', s=2, edgecolors=None)
            plt.ylim((-250,250))
            plt.savefig(foldername+name+"_eigvals.pdf")
            plt.clf()

            V = np.abs(eigvecs)
            plt.imshow(V)
            plt.colorbar()
            plt.savefig(foldername+name+"_eigvecs.pdf")
            plt.clf()

            SIP = V.T @ V
            plt.imshow(SIP)
            plt.colorbar()
            plt.savefig(foldername+name+"_abs_IP.pdf")
            plt.clf()

        return A, n, int(n/100), int(n/5)

    if name == "synthetic_tester":
        """
        uses a matrix with 1/eps^2 eigvals of size  +-eps*n and 1 eigenvalue of size +-n/2
        """
        n = 5000
        
        L = list(eps*n*np.ones(100))
        L = np.array([(n/2.0)] + L)
        mask = np.random.random(101) < 0.5
        L = np.where(mask, -1*L, L)
        L = np.diag(L)

        V = np.random.random((n,101))
        num = 0.4
        mask = np.random.random((n,101)) < num
        V = np.where(mask, 0*V, V)
        I = (1/np.sqrt(n))*np.ones(n)
        V[:,0] = I

        Q, R= np.linalg.qr(V)
        A = Q @ L @ Q.T

        eigvals, eigvecs = np.linalg.eig(A)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # save figures
        foldername = "figures/matrices/"
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        plt.imshow(A)
        plt.colorbar()
        plt.savefig(foldername+name+"_matrix.pdf")
        plt.clf()

        plt.scatter(range(n), eigvals, alpha=0.3, marker='o', s=2, edgecolors=None)
        plt.savefig(foldername+name+"_eigvals.pdf")
        plt.clf()

        V = np.abs(eigvecs)
        plt.imshow(V)
        plt.colorbar()
        plt.savefig(foldername+name+"_eigvecs.pdf")
        plt.clf()
        
        SIP = V.T @ V
        plt.imshow(SIP)
        plt.colorbar()
        plt.savefig(foldername+name+"_abs_IP.pdf")
        plt.clf()
        return A, n, int(n/100), int(n/5)

    if name == "random_equal_signs":
        """
        dataset for tracking frobenius norm error of BSSB-BB
        """
        dataset_size = 2000
        A = np.random.random((dataset_size, dataset_size))
        A = A.T @ A
        w, v = np.linalg.eig(A)
        w = np.array(list(range(2000))).astype(float)
        w = w - 1000
        w[0:500] = w[0:500] - 100.0*np.ones(w[0:500].shape) + np.random.rand(len(w[0:500]))
        w[-500:] = w[-500:] + 100.0*np.ones(w[-500:].shape) + np.random.rand(len(w[-500:]))
        w[500:1501] = np.zeros(w[500:1501].shape) + np.random.rand(len(w[500:1501]))

        # plot eigenvalues
        plt.plot(np.array(list(range(2000))), w)
        plt.xlabel("eigenvalue indices")
        plt.ylabel("eigenvalues")
        plt.title("eigenvalues of random matrix")
        plt.savefig("./figures/random_equal/eigenvalues/eigvals.pdf")

        w_half = np.lib.scimath.sqrt(w)
        B = v @ np.diag(w_half)

        return B

    if name == "tridiagonal":
        """
        matrix for zeros in the main diagonal and 1s in the 1st off-diagonal
        """
        from scipy.sparse import diags
        dataset_size = 1000
        diagonals = [list(np.zeros(dataset_size)), list(np.ones(dataset_size-1)), list(np.ones(dataset_size-1))]
        A = diags(diagonals, [0, -1, 1]).toarray()

        min_sample_size = 50
        max_sample_size = dataset_size

        return A, dataset_size, min_sample_size, max_sample_size

