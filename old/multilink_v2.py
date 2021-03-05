import torch
import numpy as np
from pdb import set_trace as bp


class MultiLinkBody():
	def __init__(self,targets,n_nodes=10, end_eff_orient=None,verbose = True):
		self.n_nodes = n_nodes
		self.n_targets = targets.shape[0]
		self.targets = torch.tensor(targets)
		dtype = torch.float
		torch.manual_seed(1)
		self.l = 0.1
		#self.qbar = torch.randn(self.n_nodes-1, dtype = dtype, requires_grad = True)
		#self.q = torch.randn(self.n_nodes-1, self.n_targets, dtype= dtype, requires_grad = True)
		self.qbar = torch.zeros(self.n_nodes-1, dtype = dtype, requires_grad = True) #home pose angles
		self.Q = torch.zeros(self.n_nodes-1, self.n_targets, dtype= dtype, requires_grad = True) # angles for all targets
		self.end_eff_orient = torch.tensor(end_eff_orient) if end_eff_orient is not None else None #if not None, this specifies required end effector orientations for each target
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

	def get_torch_node_locs(self,Q):
		xy = torch.zeros((self.n_nodes, self.n_targets, 2))
		temp = torch.zeros((self.n_targets,2))
		for i in range(1,self.n_nodes):
			temp[:,0] += self.l*torch.cos(torch.sum(Q[:i,:],dim=0))
			temp[:,1] += self.l*torch.sin(torch.sum(Q[:i,:],dim=0))
			xy[i,:,0] = temp[:,0]
			xy[i,:,1] = temp[:,1]
		return xy

	def get_link_orientations(self,Q):
		orient = torch.zeros((self.n_nodes-1,self.n_targets))
		for i in range(self.n_nodes-1):
			orient[i,:] = torch.sum(Q[:i+1,:],dim=0)
		return orient 

	def plot_configuration(self,q,Q=None,indices=None):
		xy = self.get_node_locs(q)
		import matplotlib.pyplot as plt
		if indices is not None:
			ind = max(indices)+1
		else:
			ind = len(q)+1
		plt.figure()
		plt.plot(xy[:ind,0],xy[:ind,1])
		plt.plot(xy[:ind,0],xy[:ind,1],'o')
		if Q is not None:
			for i in range(Q.shape[1]):
				if indices is not None:
					ind = indices[i]+1
				else:
					ind = len(q)+1
				xy = self.get_node_locs(Q[:,i])
				plt.plot(xy[:ind,0],xy[:ind,1],'--')
				plt.plot(xy[:ind,0],xy[:ind,1],'o')
		plt.plot(self.targets.numpy()[:,0],self.targets.numpy()[:,1],'*r',markersize = 14)
		plt.show()

	def loss(self,Q,qbar):
		node_xys = self.get_torch_node_locs(Q)
		squared_diff = torch.sum((node_xys - self.targets).pow(2), dim = 2)
		#bp()
		loss = 200.*torch.sum(torch.min(squared_diff,dim=0).values)
		indices = torch.min(squared_diff,dim=0).indices
		if self.verbose: print('Minimizers: ', indices)
		if self.end_eff_orient is not None:
			orient = self.get_link_orientations(Q)
			end_eff_link_orient = torch.zeros(self.n_targets)
			#bp()
			for i in range(self.n_targets):
				end_eff_link_orient[i] = orient[indices[i]-1,i]
			loss += 20*torch.sum((end_eff_link_orient-self.end_eff_orient).pow(2))
		#temp = 200.*torch.sum(torch.nn.functional.softmin(squared_diff, dim = 0)

		#end_eff_xys = self.get_end_eff_loc(Q)
		#loss = torch.sum(200.*(end_eff_xys - self.targets).pow(2)) # reach the targets!
		loss += 0.001*torch.sum(1.*(Q.T-qbar).pow(2)) # don't deviate much from the home pose!
		loss += 0.1*self.L1_norm(Q[1:,:])#+ self.L1_norm(qbar) #keep robot straight if possible!
		#loss += torch.sum(torch.where(torch.abs(Q[1:,:])>0.79,10.0,0.0)) #pay penalty for each angle bigger than 45deg (except for the first angle)
		loss += 0.1*torch.sum(torch.nn.functional.softplus(torch.abs(Q[1:,:])-0.79))
		return loss, indices

	def L1_norm(self,A):
		return torch.linalg.norm(A,ord=1)

	def optimize(self,n_iter = 4000, lr = 1e-3):
		for i in range(n_iter):
			loss, indices = self.loss(self.Q, self.qbar)
			if self.verbose: print(i+1,loss.item())
			loss.backward()
			with torch.no_grad():
				self.Q -= lr*self.Q.grad
				self.qbar -= lr*self.qbar.grad
				self.Q.grad.zero_()
				self.qbar.grad.zero_()
		return  self.Q.detach().numpy(), self.qbar.detach().numpy(), loss.item(), indices











if __name__ =='__main__':
	targets = 0.5*np.array([[1.2, 1.0], 
						[0.8, 0.1],
						[1.5, 0.5]])
						#[0.8, 0.7]])
	node_budget = 10
	required_endeff_orientations = np.pi/180*np.array([0.0, -10.0, 45.0])
	sess = MultiLinkBody(targets = targets, n_nodes=node_budget,end_eff_orient=required_endeff_orientations)
	sess.plot_configuration(sess.qbar.detach().numpy(),sess.Q.detach().numpy())
	Q,qbar, loss, indices = sess.optimize()
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	print('Home joint angles: ', qbar*180.0/np.pi)
	for j in range(targets.shape[0]):
		print('Target ', j+1, ' joint angles: ', Q[:,j]*180.0/np.pi)
	#print('-------')
	#print('Home link lengths: ', lbar)
	#for j in range(targets.shape[0]):
	#	print('Target ', j+1, ' link lengths: ', L[:,j]) 
	##sess.plot_configuration(lbar, qbar,L,Q)
	print(180/np.pi*sess.get_link_orientations(torch.tensor(Q)))
	sess.plot_configuration(qbar,Q)
	sess.plot_configuration(qbar,Q,indices)

	bp()