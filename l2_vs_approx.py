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
sizes = [5000]


#eps = 0.01
epses = list(np.arange(0.02, 0.11, 0.01))
# epses = [0.1]
# eps = 0.03
# samples = np.array(list(range(100,2500,10)))
trials = 10

L2_all = []
EE_all = []
#E_all = []

for j in tqdm(range(len(epses))):
	eps = epses[j]
	for n in sizes:
		"""
		diag_A = eps*n*np.ones(n)
		diag_A[int(1/eps**2)+1:] = 0		
		diag_A[0] = n
		diag_A[1] = -n
		"""
		diag_A = np.zeros(n)
		v = ggs(1/eps**2)
		diag_A[0:int(1/eps**2)] = v
		diag_A = np.sort(diag_A)[::-1]		#descending order sort

		# plt.plot(diag_A)
		# plt.show()

		A1 = np.diag(diag_A)

		A2 = np.diag(np.sort(v)[::-1])

		#A_half = sqrtm(A)

		L2 = []
		EE = []
		#E = []

		for i in range(trials):
			# get the sketch matrix
			if sketch_type == "countSketch":
				sketch_m = countSketch(A1, s=(int(1/eps**2)), return_type="sketch")
			if sketch_type == "denseSketch":
				sketch_m = denseSketch(A1, s=(int(1/eps**2)), return_type="sketch")
			
			# check
			# print(sketch_m @ sketch_m.T)
			# print(sketch_m.T @ sketch_m)
			# print(sketch_m.shape)
			
			sketched_matrix = sketch_m @ A1 @ sketch_m.T

			# get corresponding L2 error this round
			L2_error = np.linalg.norm(sketched_matrix - A2)
			
			# print(local_mat)	
			# compute eigenvalue estimate
			max_est_error = np.max(np.abs(np.sort(v)[::-1] - np.real(np.linalg.eigvals(sketched_matrix))))
			
			# print(est_eig)	
			# exit(0)
			# compute error
			# est_error = np.abs(n - est_eig)

			L2.append(L2_error)
			EE.append(max_est_error)
			#E.append(est_eig)

	# compute mean of the L2 error and estimation error
	print(L2)
	print(EE)
	L2_all.append(np.mean(L2))
	EE_all.append(np.mean(EE))
	#EE_1p.append(np.percentile(EE, 20, axis=0))
	#EE_2p.append(np.percentile(EE, 80, axis=0))
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
#samples = samples/sizes[-1]

print(len(EE_all), len(L2_all))
L2_all = np.array(L2_all)
EE_all = np.array(EE_all)
ratios = np.array(L2_all/EE_all)
#EE_1p = np.array(EE_1p)
#EE_2p = np.array(EE_2p)
#print(ratios, EE_1p, EE_2p)
#plt.plot(np.log(samples), np.log(ratios))
#plt.fill_between(np.log(samples), np.log(ratios-EE_1p+1e-60), np.log(ratios+EE_2p+1e-60), alpha=0.2)
plt.plot(epses, ratios)
plt.ylabel("meanL2 error / mean eig matching error")
plt.xlabel("Epsilons")
# plt.title("L2 error vs eigval error of estimation")
# plt.legend(loc="upper right")
dir_to_save = "figures/sketching_trial_results/"
if not os.path.isdir(dir_to_save):
	os.makedirs(dir_to_save)
filename = os.path.join(dir_to_save, sketch_type+"_L2_v_EE_GP.pdf")
plt.savefig(filename)
# '''
