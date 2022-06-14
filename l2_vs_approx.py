import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from src.sampler import countSketch 
from tqdm import tqdm
import os

# generate a zeros matrix
sketch_type = "countSketch"
sizes = range(400,5000,200)

#eps = 0.01
epses = list(np.arange(0.05, 0.11, 0.01))
trials = 10

for eps in tqdm(epses):
	
	L2_all = []
	EE_all = []

	for n in tqdm(sizes):
		diag_A = eps*np.ones(n)
		diag_A[0] = n

		A = np.diag(diag_A)

		A_half = sqrtm(A)

		L2 = []
		EE = []

		for i in range(trials):
			# get the sketch matrix
			sketch_m = countSketch(A, s=int(1/(eps**2)), return_type="sketch")

			# get corresponding L2 error this round
			L2_error = np.linalg.norm(A_half @ sketch_m.T @ sketch_m @ A_half - A)
	
			# compute eigenvalue estimate
			est_eig = np.max(np.real(np.linalg.eigvals(sketch_m @ A @ sketch_m.T)))

			# compute error
			est_error = np.abs(n - est_eig)

			L2.append(L2_error)
			EE.append(est_error)
	
		# compute mean of the L2 error and estimation error
		L2_all.append(np.mean(L2))
		EE_all.append(np.mean(EE))

	plt.plot(L2_all, EE_all, label=eps)

plt.xlabel("mean L2 error")
plt.ylabel("mean L eigval error")
plt.title("L2 error vs eigval error of estimation")
plt.legend(loc="upper right")
dir_to_save = "figures/sketching_trial_results/"
if not os.path.isdir(dir_to_save):
	os.makedirs(dir_to_save)
filename = os.path.join(dir_to_save, sketch_type+"_L2vEE.pdf")
plt.savefig(filename)
