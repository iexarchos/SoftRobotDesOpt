import torch
import numpy as np
from pdb import set_trace as bp


class MultiLinkBody():
	def __init__(self,n_nodes,targets,verbose = False):
		self.n_nodes = n_nodes
		self.n_targets = targets.shape[0]
		self.targets = torch.tensor(targets)
		dtype = torch.float
		torch.manual_seed(1)
		self.lbar = torch.rand(self.n_nodes-1, dtype= dtype, requires_grad = True)
		self.qbar = torch.randn(self.n_nodes-1, dtype = dtype, requires_grad = True)

		self.l = torch.rand(self.n_nodes-1, self.n_targets, dtype= dtype, requires_grad = True)
		self.q = torch.randn(self.n_nodes-1, self.n_targets, dtype= dtype, requires_grad = True)
		self.verbose = verbose

	def get_node_locs(self,l,q):
		xy = np.zeros((self.n_nodes,2))
		xtemp = 0.0
		ytemp = 0.0
		for i in range(1,self.n_nodes):
			xtemp += l[i-1]*np.cos(np.sum(q[:i]))
			ytemp += l[i-1]*np.sin(np.sum(q[:i]))
			xy[i,0] = xtemp
			xy[i,1] = ytemp
		#print(xy)
		return xy 

	def get_end_eff_loc(self,L,Q):
		xy = torch.zeros((self.n_targets,2))
		for i in range(1,self.n_nodes):
			xy[:,0] += L[i-1,:]*torch.cos(torch.sum(Q[:i,:],dim=0))
			xy[:,1] += L[i-1,:]*torch.sin(torch.sum(Q[:i,:],dim=0))
		return xy


	def plot_configuration(self,l,q,L=None,Q=None):
		xy = self.get_node_locs(l,q)
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(xy[:,0],xy[:,1])
		plt.plot(xy[:,0],xy[:,1],'o')
		if L is not None:
			for i in range(L.shape[1]):
				xy = self.get_node_locs(L[:,i],Q[:,i])
				plt.plot(xy[:,0],xy[:,1],'--')
				plt.plot(xy[:,0],xy[:,1],'o')
		plt.plot(self.targets.numpy()[:,0],self.targets.numpy()[:,1],'*r',markersize = 14)
		plt.show()

	def loss(self,L,Q,lbar,qbar):
		end_eff_xys = self.get_end_eff_loc(L,Q)
		loss = torch.sum(100.*(end_eff_xys - self.targets).pow(2)) # reach the targets!
		loss += torch.sum(1.*(L.T-lbar).pow(2) + 1.*(Q.T-qbar).pow(2)) # don't deviate much from the home pose!
		loss += self.L1_norm(L)+self.L1_norm(Q)#+self.L1_norm(lbar)+ self.L1_norm(qbar) #don't extend and bend too much!
		return loss

	def L1_norm(self,A):
		return torch.linalg.norm(A,ord=1)

	def optimize(self,n_iter = 2000, lr = 1e-3):
		for i in range(n_iter):
			loss = self.loss(self.l, self.q, self.lbar, self.qbar)
			if self.verbose: print(i+1,loss.item())
			loss.backward()
			with torch.no_grad():
				self.l -= lr*self.l.grad
				self.q -= lr*self.q.grad
				self.lbar -= lr*self.lbar.grad
				self.qbar -= lr*self.qbar.grad
				self.l.grad.zero_()
				self.q.grad.zero_()
				self.lbar.grad.zero_()
				self.qbar.grad.zero_()
		return self.l.detach().numpy(), self.q.detach().numpy(), self.lbar.detach().numpy(), self.qbar.detach().numpy(), loss.item()











if __name__ =='__main__':
	targets = np.array([[1.2, 1.0], 
						[0.8, 0.1],
						[1.5, 0.5]])
						#[0.8, 0.7]])
	node_budget = 10
	Loss = np.zeros((node_budget+1,))
	Loss[0:2] = np.inf
	Res = []
	for i in range(2,node_budget+1):
		#print('Testing ', i , 'nodes...')
		sess = MultiLinkBody(n_nodes = i, targets = targets)
		##sess.plot_configuration(sess.lbar.detach().numpy(), sess.qbar.detach().numpy(),sess.l.detach().numpy(),sess.q.detach().numpy())
		L,Q,lbar,qbar, loss = sess.optimize()
		del sess
		print(i,' nodes, optimal loss: ', loss)
		Loss[i] = loss
		Res.append([L, Q, lbar, qbar, loss])
		#print('Home joint angles: ', qbar*180.0/np.pi)
		#for j in range(targets.shape[0]):
		#	print('Target ', j+1, ' joint angles: ', Q[:,j]*180.0/np.pi)
		#print('-------')
		#print('Home link lengths: ', lbar)
		#for j in range(targets.shape[0]):
		#	print('Target ', j+1, ' link lengths: ', L[:,j]) 
		##sess.plot_configuration(lbar, qbar,L,Q)

	opt_nodes = np.argmin(Loss)
	print('Optimal node number: ', opt_nodes, ', optimal loss: ', Loss[opt_nodes])
	L_opt, Q_opt, lbar_opt, qbar_opt, loss_opt = Res[opt_nodes-2]
	sess = MultiLinkBody(n_nodes = opt_nodes, targets = targets)
	sess.plot_configuration(sess.lbar.detach().numpy(), sess.qbar.detach().numpy(),sess.l.detach().numpy(),sess.q.detach().numpy())
	sess.plot_configuration(lbar_opt, qbar_opt,L_opt,Q_opt)

	bp()