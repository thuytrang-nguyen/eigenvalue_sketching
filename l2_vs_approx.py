import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from src.sampler import countSketch, denseSketch 
from tqdm import tqdm
import os

# generate a zeros matrix
# "countSketch", "denseSketch"
# sketch_type = "countSketch"
sketch_type = "denseSketch"
# sizes = range(500,5000,200)
sizes = [5000]


#eps = 0.01
epses = list(np.arange(0.05, 0.11, 0.01))
# epses = [0.1]
trials = 10

L2_all = []
EE_all = []
E_all = []

for eps in tqdm(epses):
	"""
	L2_all = []
	EE_all = []
	E_all = []
	"""
	for n in tqdm(sizes):
		diag_A = eps*n*np.ones(n)
		diag_A[int(1/eps**2)+1:] = 0		
		diag_A[0] = n
		
		# plt.plot(diag_A)
		# plt.show()

		A = np.diag(diag_A)

		A_half = sqrtm(A)

		L2 = []
		EE = []
		E = []

		for i in range(trials):
			# get the sketch matrix
			if sketch_type == "countSketch":
				sketch_m = countSketch(A, s=int(1/(eps**2)), return_type="sketch")
			if sketch_type == "denseSketch":
				sketch_m = denseSketch(A, s=int(1/(eps**2)), return_type="sketch")
			
			# check
			# print(sketch_m @ sketch_m.T)
			# print(sketch_m.T @ sketch_m)
			# print(sketch_m.shape)
			
			# get corresponding L2 error this round
			L2_error = np.linalg.norm(A_half @ sketch_m.T @ sketch_m @ A_half - A)
			
			local_mat = np.matmul(sketch_m, np.matmul(A, sketch_m.T))
			# print(local_mat)	
			# compute eigenvalue estimate
			# est_eig = np.max(np.real(np.linalg.eigvals(np.matmul(sketch_m, np.matmul(A, sketch_m.T)))))
			est_eig = np.max(np.real(np.linalg.eigvals(local_mat)))
			
			# print(est_eig)	
			# exit(0)
			# compute error
			est_error = np.abs(n - est_eig)

			L2.append(L2_error)
			EE.append(est_error)
			E.append(est_eig)
	
		# compute mean of the L2 error and estimation error
		L2_all.append(np.mean(L2))
		EE_all.append(np.mean(EE))
		E_all.append(np.mean(E))

	#plt.plot(L2_all, EE_all, label="{:.2f}".format(eps))

# print(L2_all, EE_all)
# print(E_all)
# print(sketch_m @ A @ sketch_m.T)
# LSSL = A_half @ sketch_m.T @ sketch_m @ A_half

# plt.imshow(LSSL)
# plt.show()
# '''
plt.plot(L2_all, EE_all)

plt.xlabel("mean L2 error")
plt.ylabel("mean L eigval error")
plt.title("L2 error vs eigval error of estimation")
# plt.legend(loc="upper right")
dir_to_save = "figures/sketching_trial_results/"
if not os.path.isdir(dir_to_save):
	os.makedirs(dir_to_save)
filename = os.path.join(dir_to_save, sketch_type+"_L2vEE.pdf")
plt.savefig(filename)
# '''
