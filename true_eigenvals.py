import numpy as np
import networkx as nx
import idx2numpy
from sklearn.preprocessing import normalize
from random import sample
import os
import pickle
import random
import matplotlib.pyplot as plt
from src.main_approximator import approximator
from src.viz import plot_all_errors, plot_all_nnz, plot_eigval_vs_nnzA
from src.utils import get_distance
from src.display_codes import disply_prob_histogram
from src.get_dataset import get_data
from src.similarities import hyperbolic_tangent, thin_plane_spline
from copy import copy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

#random.seed()
# Parameters

search_rank = [0,1,2,3,-4,-3,-2,-1]
dataset_name = "diag_power"
# dataset_name = "erdos", "MNIST", "block", "facebook", "kong", "multi_block_outer", "arxiv", "tridiagonal"
if dataset_name == "kong":
    similarity_measure = "tps" # "tps", "ht", 

# Get the dataset
if dataset_name == "kong":
    xy, dataset_size, min_samples, max_samples = get_data(dataset_name)
    if similarity_measure == "ht":
        similarity = hyperbolic_tangent
    if similarity_measure == "tps":
        similarity = thin_plane_spline
    true_mat = similarity(xy, xy)
if dataset_name != "kong":
    true_mat, dataset_size, min_samples, max_samples = get_data(dataset_name)
true_spectrum = np.real(np.linalg.eigvals(true_mat))
print("||A||_infty:", np.max(true_mat))
true_spectrum.sort()
#print(true_spectrum)
chosen_eig = true_spectrum[search_rank]
print("chosen eigs:", chosen_eig)
