import numpy as np
from pdb import set_trace as bp
from NOVAS_multilink_v2 import MultiLinkBody


class TestMultiLink():
	def __init__(self,MLB):
		self.seeds = [0,1,2]
		self.MLB = MLB()

	def test_design_for_n_targets(self,n_nodes,max_targets, threshold = 1.):
		SR = np.zeros((max_targets+1,))
		Loss = np.zeros((max_targets+1,))
		Loss[0:2] = np.inf
		for n_targets in range(2,max_targets+1):
			targets, orient = self.MLB.TargetGeneration(n_nodes=6,n_targets=n_targets)
			old_perc = 0
			old_loss = np.inf
			print('-----DESIGN: Testing ', n_nodes-1,' links, ', n_targets, ' targets --------')
			for seed in self.seeds: #repeat opt for a few seeds or 100% reached
				lopt, Qopt, loss, perc = self.MLB.Optimize_design(n_nodes=n_nodes,targets=targets, orientations=orient, seed = seed, n_iter = 300, n_samples = 1000, sigma = 1.0, gamma = 10.0, threshold= threshold,  verbose = False) 
				#self.MLB.plot_configuration(lopt,Qopt,targets=targets,orient=orient)
				print('sr: ', perc)
				perc = max(old_perc,perc)
				print('max sr', perc)
				loss = min(old_loss,loss)
				old_perc = perc
				old_loss = loss
				if perc == 1:
					break
			#if n_targets == 4: 
				#test_exec_sr = self.test_execution_for_random_targets(lopt,M=100)
			SR[n_targets] = perc
			Loss[n_targets] = loss

		return SR, Loss#, test_exec_sr 


	def test_execution_for_random_targets(self,lopt, M, threshold = 1.):
		print('------------------------------ TESTING EXECUTION ---------------------------------')
		np.random.seed(1)
		new_targets = 0.1+0.8*np.random.rand(M,2)
		new_orientations = np.pi/180*(-45+90*np.random.rand(M,))
		LOSS = np.zeros((M,))
		for i in range(M):
			print('---Running iteration: ', i+1,'---')
			target = np.array(new_targets[i,:])
			target = target[np.newaxis,:] 
			orient = np.array([new_orientations[i]])
			Qnew, loss = self.MLB.exec_NOVAS(lopt=lopt,new_targets=target,new_orientations=orient,n_iter = 150, n_samples = 200, sigma = 1.0, gamma = 10.0, verbose = False) 
		#idx, losses = sess.plot_configuration(lopt,Qnew,targets=target,orientations=orient)
		#print('Link lengths: ', sess.l)
		#print('Q: ', Qnew)
			LOSS[i] = loss
		success = np.nonzero(LOSS<threshold)
		success_rate = len(success[0])/M
		print('Success rate: ', success_rate)
		#import matplotlib.pyplot as plt
		#plt.hist(LOSS,bins=[0,1,2,4,10,max(LOSS)])
		#plt.show()
		return success_rate




if __name__ =='__main__':
	MLB = MultiLinkBody
	sess = TestMultiLink(MLB)
	max_nodes = 9 # nr of nodes = nr of links + 1
	max_targets = 6
	n_nodes = np.arange(3,max_nodes+1)
	SR = np.zeros((max_nodes+1,max_targets+1))
	Loss = SR.copy()
	Exec_SR = np.zeros((max_nodes+1,))
	for i in n_nodes:
		#SR[i,:], Loss[i,:], Exec_SR[i] = sess.test_design_for_n_targets(n_nodes=i,max_targets=max_targets)
		SR[i,:], Loss[i,:] = sess.test_design_for_n_targets(n_nodes=i,max_targets=max_targets)
	
	print('SR: ', SR)
	#print('Execution SR: ', Exec_SR)

	bp()

