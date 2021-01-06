import torch
import numpy as np
from pdb import set_trace as bp


class MultiLinkBody():
	def __init__(self,targets,n_nodes=10,verbose = True):
		self.n_nodes = n_nodes
		self.n_targets = targets.shape[0]
		self.targets = torch.tensor(targets)
		dtype = torch.float
		torch.manual_seed(1)
		self.l = 0.1
		self.qbar = torch.randn(self.n_nodes-1, dtype = dtype, requires_grad = True)
		self.q = torch.randn(self.n_nodes-1, self.n_targets, dtype= dtype, requires_grad = True)
		self.verbose = verbose

	def get_node_locs(self,q):
		xy = np.zeros((self.n_nodes,2))
		xtemp = 0.0
		ytemp = 0.0
		for i in range(1,self.n_nodes):
			xtemp += self.l*np.cos(np.sum(q[:i]))
			ytemp += self.l*np.sin(np.sum(q[:i]))
			xy[i,0] = xtemp
			xy[i,1] = ytemp
		#print(xy)
		return xy 

	def get_end_eff_loc(self,Q):
		xy = torch.zeros((self.n_targets,2))
		for i in range(1,self.n_nodes):
			xy[:,0] += self.l*torch.cos(torch.sum(Q[:i,:],dim=0))
			xy[:,1] += self.l*torch.sin(torch.sum(Q[:i,:],dim=0))
		return xy


	def plot_configuration(self,q,Q=None):
		xy = self.get_node_locs(q)
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(xy[:,0],xy[:,1])
		plt.plot(xy[:,0],xy[:,1],'o')
		if Q is not None:
			for i in range(Q.shape[1]):
				xy = self.get_node_locs(Q[:,i])
				plt.plot(xy[:,0],xy[:,1],'--')
				plt.plot(xy[:,0],xy[:,1],'o')
		plt.plot(self.targets.numpy()[:,0],self.targets.numpy()[:,1],'*r',markersize = 14)
		plt.show()

	def loss(self,Q,qbar):
		end_eff_xys = self.get_end_eff_loc(Q)
		loss = torch.sum(200.*(end_eff_xys - self.targets).pow(2)) # reach the targets!
		loss += torch.sum(1.*(Q.T-qbar).pow(2)) # don't deviate much from the home pose!
		loss += self.L1_norm(Q[1:,:])#+ self.L1_norm(qbar) #keep robot straight if possible!
		#loss += torch.sum(torch.where(torch.abs(Q[1:,:])>0.52,10.0,0.0)) #pay penalty for each angle bigger than 45deg (except for the first angle)
		loss += torch.sum(torch.nn.functional.softplus(torch.abs(Q[1:,:])-0.79))
		return loss

	def L1_norm(self,A):
		return torch.linalg.norm(A,ord=1)

	def optimize(self,n_iter = 4000, lr = 1e-3):
		for i in range(n_iter):
			loss = self.loss(self.q, self.qbar)
			if self.verbose: print(i+1,loss.item())
			loss.backward()
			with torch.no_grad():
				self.q -= lr*self.q.grad
				self.qbar -= lr*self.qbar.grad
				self.q.grad.zero_()
				self.qbar.grad.zero_()
		return  self.q.detach().numpy(), self.qbar.detach().numpy(), loss.item()











if __name__ =='__main__':
	targets = 0.5*np.array([[1.2, 1.0], 
						[0.8, 0.1],
						[1.5, 0.5]])
						#[0.8, 0.7]])
	node_budget = 10
	sess = MultiLinkBody(targets = targets, n_nodes=node_budget)
	sess.plot_configuration(sess.qbar.detach().numpy(),sess.q.detach().numpy())
	Q,qbar, loss = sess.optimize()
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	print('Home joint angles: ', qbar*180.0/np.pi)
	for j in range(targets.shape[0]):
		print('Target ', j+1, ' joint angles: ', Q[:,j]*180.0/np.pi)
	#print('-------')
	#print('Home link lengths: ', lbar)
	#for j in range(targets.shape[0]):
	#	print('Target ', j+1, ' link lengths: ', L[:,j]) 
	##sess.plot_configuration(lbar, qbar,L,Q)

	
	sess.plot_configuration(qbar,Q)

	bp()