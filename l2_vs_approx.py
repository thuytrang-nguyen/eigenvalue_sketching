import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from src.sampler import countSketch 

# generate a zeros matrix
n = 1000
eps = 0.01
trials = 10

diag_A = eps*np.ones(n)
diag_A[0] = n

A = np.diag(diag_A)

A_half = sqrtm(A)

for i in range(trials):
	sketch = countSketch(A, return_type="sketch")
