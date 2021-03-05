import numpy as np
from pdb import set_trace as bp
from scipy import special
#import nlopt
#NOVAS shared link length except for final link which is optimized individually
class MultiLinkBody():
	def __init__(self,n_nodes,targets,orientations,verbose = False):
		self.n_nodes = n_nodes
		self.n_targets = targets.shape[0]
		self.targets = np.array(targets)
		self.orientations = orientations
		
		np.random.seed(1)
		#self.lbar = torch.rand(self.n_nodes-1, dtype= dtype, requires_grad = True)
		#self.qbar = np.zeros(self.n_nodes-1, dtype = dtype, requires_grad = True)

		self.l = np.random.rand(self.n_nodes-2,)
		self.l_final = np.random.rand(self.n_targets,)
		self.q = np.zeros((self.n_nodes-1, self.n_targets))
		self.verbose = verbose

	def get_node_locs(self,l,l_final,q):
		xy = np.zeros((self.n_nodes,2))
		xtemp = 0.0
		ytemp = 0.0
		for i in range(1,self.n_nodes):
			if i==self.n_nodes-1:
				xtemp += l_final*np.cos(np.sum(q[:i]))
				ytemp += l_final*np.sin(np.sum(q[:i]))
			else:
				xtemp += l[i-1]*np.cos(np.sum(q[:i]))
				ytemp += l[i-1]*np.sin(np.sum(q[:i]))
			xy[i,0] = xtemp
			xy[i,1] = ytemp
		#print(xy)
		return xy 

	def get_end_eff_loc(self,l,l_final,Q):
		xy = np.zeros((self.n_targets,2))
		for i in range(1,self.n_nodes):
			if i==self.n_nodes-1:
				xy[:,0] += l_final*np.cos(np.sum(Q[:i,:],axis=0))
				xy[:,1] += l_final*np.sin(np.sum(Q[:i,:],axis=0))
			else:
				xy[:,0] += l[i-1]*np.cos(np.sum(Q[:i,:],axis=0))
				xy[:,1] += l[i-1]*np.sin(np.sum(Q[:i,:],axis=0))
		return xy

	def get_end_eff_orientation(self,Q):
		orient = np.sum(Q,axis=0)
		return orient


	def plot_configuration(self,l,l_final,Q=None):
		#xy = self.get_node_locs(l,q)
		import matplotlib.pyplot as plt
		#plt.figure()
		#plt.plot(xy[:,0],xy[:,1])
		#plt.plot(xy[:,0],xy[:,1],'o')
		#if Q is not None:
		for i in range(Q.shape[1]):
			xy = self.get_node_locs(l,l_final[i],Q[:,i])
			plt.plot(xy[:,0],xy[:,1],'--')
			plt.plot(xy[:,0],xy[:,1],'o')
		plt.plot(self.targets[:,0],self.targets[:,1],'*r',markersize = 14)
		plt.show()

	def loss(self,l,l_final,Q):
		end_eff_xys = self.get_end_eff_loc(l,l_final,Q)
		end_eff_orient = self.get_end_eff_orientation(Q)
		loss = np.sum(100.*(end_eff_xys - self.targets)**2) # reach the targets!
		loss +=  20*np.sum((end_eff_orient-self.orientations)**2)
		#loss += np.sum(1.*(Q.T-qbar).pow(2)) # don't deviate much from the home pose!
		loss += 0.1*(np.sum(Q*Q)+np.sum(l*l)+np.sum(l_final*l_final))
		#loss += 0.1*self.L1_norm(Q)#+self.L1_norm(lbar)+ self.L1_norm(qbar) #don't bend too much!
		#loss += 0.1*self.L1_norm(l)
		#loss += 0.1*torch.sum(torch.nn.functional.softplus(torch.abs(Q[1:,:])-0.79))
		return loss


#	def NLOPT_loss(self,x):
#		l = x[0:self.n_nodes-1]
#		q = x[self.n_nodes-1:].reshape((self.n_nodes-1, self.n_targets))
#		return self.loss(l,q) 
			
	def NOVAS_loss(self,x):
		idx1 = self.n_nodes-2
		idx2 = idx1+self.n_targets
		L = np.zeros(x.shape[0],)
		for i in range(x.shape[0]):
			l = x[i,0:idx1]
			l_final = x[i,idx1:idx2]
			q = x[i,idx2:].reshape((self.n_nodes-1, self.n_targets))
			L[i]= self.loss(l,l_final,q)
		return L 



	def L1_norm(self,A):
		return torch.linalg.norm(A,ord=1)

#	def optimize(self,n_iter = 2000, lr = 1e-3):
#		#dim = self.l.size()+self.q.size()
#		x = np.concatenate((self.l,self.q.reshape((self.n_nodes-1)* self.n_targets)),axis=0)
#		opt = nlopt.opt(nlopt.GN_CRS2_LM,x.shape[0])
#		opt.set_min_objective(self.NLOPT_loss)
#		lb = np.concatenate((np.zeros(self.n_nodes-1,),-np.pi/4*np.ones((self.n_nodes-1)*self.n_targets, )),axis=0)
#		ub = np.concatenate((np.ones(self.n_nodes-1,),np.pi/4*np.ones((self.n_nodes-1)*self.n_targets, )),axis=0)
#		opt.set_lower_bounds(lb)
#		opt.set_upper_bounds(ub)
#		opt.verbose=1
#		xopt = opt.optimize(x)
#		return xopt[:self.n_nodes-1], xopt[self.n_nodes-1:].reshape((self.n_nodes-1, self.n_targets))

	def NOVAS(self, n_iter = 1000, n_samples = 200, sigma = 1, gamma = 10.0):
		mu = np.concatenate((self.l,self.l_final,self.q.reshape((self.n_nodes-1)* self.n_targets)),axis=0)
		dim = len(mu)
		#bp()
		#bp()
		for i in range(n_iter+1):
			deltax = sigma*np.random.normal(size=(n_samples,dim))
			x = mu[np.newaxis,:]+deltax
			fx = -self.NOVAS_loss(x)
			fx = fx - fx.min() 
			fx = fx/(fx.max() - fx.min())
			S = special.softmax(gamma*fx)
			mu = mu + (S[:,np.newaxis]*deltax).sum(axis=0) 
			sigma = np.sqrt((S[:,np.newaxis]*deltax**2).sum(axis=0)+1e-8)
			obj = self.NOVAS_loss(mu[np.newaxis,:])
			if (i+1)%1==0:
				print('Iter:{}, obj. value={:.3f}, avg. sigma={:.3e}'.format(i, np.squeeze(obj), sigma.mean()))

		idx1 = 	self.n_nodes-2
		idx2 = idx1 + self.n_targets

		return mu[:idx1], mu[idx1:idx2],mu[idx2:].reshape((self.n_nodes-1, self.n_targets)), obj













if __name__ =='__main__':
	targets = np.array([[1.2, 1.0], 
						[0.8, 0.1],
						[1.5, 0.5]])
						#[0.8, 0.7]])
	required_endeff_orientations = np.pi/180*np.array([0.0, -120.0, 45.0])
	#node_budget = 10
	#Loss = np.zeros((node_budget+1,))
	#Loss[0:2] = np.inf
	#Res = []
	#for i in range(2,node_budget+1):
		#print('Testing ', i , 'nodes...')
	sess = MultiLinkBody(n_nodes = 4, targets = targets,orientations=required_endeff_orientations)
		##sess.plot_configuration(sess.lbar.detach().numpy(), sess.qbar.detach().numpy(),sess.l.detach().numpy(),sess.q.detach().numpy())
	l, l_final, Q, Loss = sess.NOVAS()
	#l,Q = sess.optimize()
	#	del sess
#		print(i,' nodes, optimal loss: ', loss)
#		Loss[i] = loss
#		Res.append([l, Q, qbar, loss])
		#print('Home joint angles: ', qbar*180.0/np.pi)
		#for j in range(targets.shape[0]):
		#	print('Target ', j+1, ' joint angles: ', Q[:,j]*180.0/np.pi)
		#print('-------')
		#print('Home link lengths: ', lbar)
		#for j in range(targets.shape[0]):
		#	print('Target ', j+1, ' link lengths: ', L[:,j]) 
		##sess.plot_configuration(lbar, qbar,L,Q)

	#opt_nodes = np.argmin(Loss)
	#print('Optimal node number: ', opt_nodes, ', optimal loss: ', Loss[opt_nodes])
	#l_opt, Q_opt, qbar_opt, loss_opt = Res[opt_nodes-2]
	sess = MultiLinkBody(n_nodes = 4, targets = targets,orientations=required_endeff_orientations)
	sess.plot_configuration(sess.l,sess.l_final,sess.q)
	sess.plot_configuration(l,l_final,Q)
	print('End effector orientations: ')
	print(180/np.pi*sess.get_end_eff_orientation(Q))

	bp()
