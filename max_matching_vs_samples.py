import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from src.sampler import countSketch, denseSketch 
from tqdm import tqdm
import os
from src.utils import generate_gp_series as ggs

# generate a zeros matrix
# "countSketch", "denseSketch"
# sketch_type = "countSketch"
sketch_type = "denseSketch"
# sizes = range(500,5000,200)
sizes = [20000]


#eps = 0.01
# epses = list(np.arange(0.01, 0.11, 0.01))
# epses = [0.1]
# eps = 0.03
samples = np.array(list(range(100,1100,10)))
trials = 10

#L2_all = []
EE_all = []
EE_1p = []
EE_2p = []
#E_all = []


diag_A = np.zeros(sizes[-1])
v = np.sort(ggs(sizes[0]))
diag_A[0:sizes[0]] = v
diag_A = np.sort(diag_A)[::-1]          # descending order sort

# plt.plot(diag_A)
# plt.show()

A = np.diag(diag_A)

for j in tqdm(range(len(samples))):
	"""
	L2_all = []
	EE_all = []
	E_all = []
	"""
	for n in sizes:
		#L2 = []
		EE = []
		#E = []

		for i in range(trials):
			# get the sketch matrix
			if sketch_type == "countSketch":
				sketch_m = countSketch(A, s=samples[j], return_type="sketch")
			if sketch_type == "denseSketch":
				sketch_m = denseSketch(A, s=samples[j], return_type="sketch")
			
			# check
			# print(sketch_m @ sketch_m.T)
			# print(sketch_m.T @ sketch_m)
			# print(sketch_m.shape)
			
			# get corresponding L2 error this round
			# L2_error = np.linalg.norm(A_half @ sketch_m.T @ sketch_m @ A_half - A)
			
			local_mat = np.matmul(sketch_m, np.matmul(A, sketch_m.T))
			# print(local_mat)	
			# compute eigenvalue estimate
			# est_eig = np.max(np.real(np.linalg.eigvals(np.matmul(sketch_m, np.matmul(A, sketch_m.T)))))
			est_eigs = np.real(np.linalg.eigvals(local_mat))
			vals = np.zeros(len(diag_A))
			vals[0:len(est_eigs)] = est_eigs
			vals = np.sort(vals)[::-1]
			max_est_error = np.max(np.abs(diag_A - vals))
			
			# print(est_eig)	
			# exit(0)
			# compute error
			# est_error = np.abs(n - est_eig)

			# L2.append(L2_error)
			EE.append(max_est_error)
			#E.append(est_eig)
	
		# compute mean of the L2 error and estimation error
		# L2_all.append(np.mean(L2))
		EE_all.append(np.mean(EE))
		EE_1p.append(np.percentile(EE, 20, axis=0))
		EE_2p.append(np.percentile(EE, 80, axis=0))
		#E_all.append(np.mean(E))

	#plt.plot(L2_all, EE_all, label="{:.2f}".format(eps))

# print(L2_all, EE_all)
# print(E_all)
# print(sketch_m @ A @ sketch_m.T)
# LSSL = A_half @ sketch_m.T @ sketch_m @ A_half

# plt.imshow(LSSL)
# plt.show()
# '''
# plt.plot(L2_all, EE_all)
samples = samples/sizes[-1]
ratios = np.array(EE_all)
EE_1p = np.array(EE_1p)
EE_2p = np.array(EE_2p)
#print(ratios, EE_1p, EE_2p)
plt.plot(np.log(samples), np.log(ratios))
plt.fill_between(np.log(samples), np.log(ratios-EE_1p+1e-60), np.log(ratios+EE_2p+1e-60), alpha=0.2)
plt.ylabel("log mean eig matching error")
plt.xlabel("log proportion of samples")
# plt.title("L2 error vs eigval error of estimation")
# plt.legend(loc="upper right")
dir_to_save = "figures/sketching_trial_results/"
if not os.path.isdir(dir_to_save):
	os.makedirs(dir_to_save)
filename = os.path.join(dir_to_save, sketch_type+"_MaxMatch_v_samples_GP.pdf")
plt.savefig(filename)
# '''
