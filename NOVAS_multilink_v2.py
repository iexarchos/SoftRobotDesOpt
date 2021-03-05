import numpy as np
from pdb import set_trace as bp
from scipy import special


class MultiLinkBody():
	def __init__(self):
		pass


	def TargetGeneration(self,n_nodes,n_targets,render = False):
		self.n_nodes = n_nodes
		n_links = n_nodes - 1
		n_targets = n_targets
		total_length = 1.
		np.random.seed(4)
		av_link_length = total_length/n_links
		L = 0.5*av_link_length+av_link_length*np.random.rand(n_links,) # each link random in 0.5*av, 1.5*av
		Q = np.zeros((n_links, n_targets))
		
		Q[0,:] = 90.*np.random.rand(n_targets,)
		Q[1:,:] = -30.+60.*np.random.rand(n_links-1,n_targets)
		print(L)
		print('Total length: ', L.sum())
		print(Q)
		#bp()
		Q = Q*np.pi/180
		#tt = np.random.rand(n_targets,)
		tt = np.linspace(0.1,0.9,n_targets)
		#print(tt)
		points = L[0]+(L.sum()-L[0])*tt #sample points along the robot
		targets = np.zeros((n_targets,2))
		orientations = np.zeros((n_targets,))
		for i in range(n_targets):
			P = points[i]
			kk = 0
			while P > L[0:kk+1].sum():
				kk+=1
			diff = P - L[0:kk].sum()
			xy = self.get_single_node_locs(L,Q[:,i])
			diff = diff/np.sqrt((xy[kk+1,0]-xy[kk,0])**2+(xy[kk+1,1]-xy[kk,1])**2)
			targets[i,:] = xy[kk,:]+diff*(xy[kk+1,:]-xy[kk,:])
			orientations[i] = np.sum(Q[:kk+1,i],axis=0)

		if render:
			self.plot_configuration(L,Q,targets,orientations)
		
		return targets, orientations
		


		

	def get_single_node_locs(self,l,q):
		#get node locations for a single configuration/target
		xy = np.zeros((self.n_nodes,2))
		xtemp = 0.0
		ytemp = 0.0
		for i in range(1,self.n_nodes):
			xtemp += l[i-1]*np.cos(np.sum(q[:i]))
			ytemp += l[i-1]*np.sin(np.sum(q[:i]))
			xy[i,0] = xtemp
			xy[i,1] = ytemp
		return xy

	def get_multi_node_locs(self,l,Q):
		# get node locations for all configurations/targets
		xy = np.zeros((self.n_nodes, Q.shape[1], 2))
		temp = np.zeros((Q.shape[1],2))
		for i in range(1,self.n_nodes):
			temp[:,0] += l[i-1]*np.cos(np.sum(Q[:i,:],axis=0))
			temp[:,1] += l[i-1]*np.sin(np.sum(Q[:i,:],axis=0))
			xy[i,:,0] = temp[:,0]
			xy[i,:,1] = temp[:,1]
		return xy

	def distance_to_segment(self,p,v,w, loss_mode = True, return_point = False):
		#calculate the distance of point p to the line segment vw#

		def dist_squared(v,w):
			#squared distance between two points
			return (v[0]-w[0])**2+(v[1]-w[1])**2
		
		line_seg_length_squared = dist_squared(v,w) #line segment length
		if line_seg_length_squared == 0:
			return np.sqrt(dist_squared(p,v))
		#extended line v+t(w-v), parameterized by t. Point p projects on line at
		# t = (p-v)*(w-v)/(w-v)**2
		t = ((p[0]-v[0])*(w[0]-v[0])+(p[1]-v[1])*(w[1]-v[1]))/line_seg_length_squared #  
		if loss_mode:
			t = max(0.3,min(0.9,t)) # clamp between [0.3,0.9]
		else:
			t = max(0,min(1.0,t)) # clamp between [0,1]
		vw_closest_to_p = v+t*(w-v) # find point on vw closest to p
		dist_to_seg = np.sqrt(dist_squared(p,vw_closest_to_p))
		if return_point:
			return vw_closest_to_p
		else:
			return dist_to_seg



	def get_dist_and_orient(self,l,Q,targets):
		xy = self.get_multi_node_locs(l,Q)
		dist = np.zeros((self.n_nodes-1,targets.shape[0]))
		orient = np.zeros((self.n_nodes-1,targets.shape[0]))
		for i in range(targets.shape[0]):
			for j in range(self.n_nodes-1):
				dist[j,i]=self.distance_to_segment(targets[i,:],xy[j+1,i,:],xy[j,i,:])
				orient[j,i] = np.sum(Q[:j+1,i],axis=0)
		return dist,orient


	def plot_configuration(self,l,Q,targets,orient):
		import matplotlib.pyplot as plt
		from matplotlib.collections import LineCollection
		from matplotlib.colors import ListedColormap
		inv_dist = 1./np.sum(targets**2,axis=1)
		idx, losses = self.loss(l,Q,targets,orient,inv_dist,return_idx_losses=True)
		print('idx: ',idx)
		al = 0.1
		cmap = ListedColormap(['r', 'g', 'b'])
		fig, ax = plt.subplots()
		CMap = 'Dark2'
		for i in range(targets.shape[0]):
			xy = self.get_single_node_locs(l,Q[:,i])
			points = xy.reshape(-1,1,2)
			segments = np.concatenate([points[:-1], points[1:]], axis=1)
			lc = LineCollection(segments, cmap=CMap, linestyles='dashed')
			lc.set_array(np.arange(0,self.n_nodes))
			ax.add_collection(lc)
			plt.plot(xy[:,0],xy[:,1],'o',color='grey')
			xy_new = xy[0:idx[i]+2,:]
			xy_new[-1,:] = self.distance_to_segment(targets[i,:],xy[idx[i],:],xy[idx[i]+1,:],loss_mode = False,return_point = True)
			print('Last link length: ', np.sqrt((xy_new[-1,0]-xy_new[-2,0])**2+(xy_new[-1,1]-xy_new[-2,1])**2))
			points = xy_new.reshape(-1,1,2)
			segments = np.concatenate([points[:-1], points[1:]], axis=1)
			lc = LineCollection(segments, cmap=CMap, linewidths=5)
			lc.set_array(np.arange(0,self.n_nodes))
			ax.add_collection(lc)
			plt.arrow(targets[i,0],targets[i,1],al*np.cos(orient[i]),al*np.sin(orient[i]),width=0.01,color = 'black')

		plt.plot(targets[:,0],targets[:,1],'*r',markersize = 22)	
		plt.axis('equal')
		plt.show()
		return idx, losses



	def loss(self,l,Q,targets,orientations,inv_dist,return_idx_losses = False, print_idx = False):
		dist,orient = self.get_dist_and_orient(l,Q,targets)
		losses = 100*(dist+0.5*abs(orient[:,:]-orientations[np.newaxis,:]))
		loss = np.sum(inv_dist**2*np.min(losses[1:,:],axis =0)) #closer targets are more difficult, penalize them more
		if print_idx:
			idxs = np.argmin(losses,axis = 0)
			print('End-eff links: ', idxs)
		if return_idx_losses:
			return np.argmin(losses,axis = 0), np.min(losses,axis =0)
		else:
			return loss


	def NOVAS_loss(self,x,inv_dist,print_idx = False):
		L = np.zeros(x.shape[0],)
		for i in range(x.shape[0]):
			l = x[i,0:self.n_nodes-1]
			q = x[i,self.n_nodes-1:].reshape((self.n_nodes-1, self.n_targets))
			L[i]= self.loss(l,q,self.targets,self.orientations,inv_dist,print_idx = print_idx)
		return L 

	def SetDesignPar(self,n_nodes):
		self.n_nodes = n_nodes

	def Optimize_design(self,n_nodes, targets, orientations, seed=0, n_iter = 200, n_samples = 1000, sigma = 1.0, gamma = 10.0, threshold=1., verbose = True):
		self.SetDesignPar(n_nodes) #this overwrites n_nodes from TargetGeneration!
		self.n_targets = targets.shape[0]
		self.targets = np.array(targets)
		self.orientations = orientations
		
		np.random.seed(seed)
		max_tar_dist = max(np.sum(self.targets**2,axis=1))
		
		robot_length = 0.25*max_tar_dist
		av_link_length = robot_length/(n_nodes-1)
		self.l = 0.5*av_link_length+av_link_length*np.random.rand(self.n_nodes-1,)
		l_min = np.array([0.1]*(self.n_nodes-1)) #minimum link length
		l_max = np.array([1.0]*(self.n_nodes-1)) #maximum link length
		self.a = 30*np.pi/180
		Q_min = -self.a*np.ones((self.n_nodes-1,self.n_targets)) #minimum bending angle
		Q_min[0,:] = -np.pi #base can rotate freely
		Q_max = self.a*np.ones((self.n_nodes-1,self.n_targets)) #minimum bending angle
		Q_max[0,:] = np.pi
		self.lower_bounds = np.concatenate((l_min,Q_min.reshape((self.n_nodes-1)* self.n_targets)),axis=0)
		self.upper_bounds = np.concatenate((l_max,Q_max.reshape((self.n_nodes-1)* self.n_targets)),axis=0)
		self.Q = np.zeros((self.n_nodes-1, self.n_targets))
		#self.Q[0,:] = 45*np.pi/180
		#self.Q[0,:] = (90.*np.random.rand(self.n_targets,))*np.pi/180
		#self.Q[1:,:] = (-30.+60.*np.random.rand(self.n_nodes-2,self.n_targets))*np.pi/180
		
		self.verbose = verbose
		lopt, Qopt, loss, perc = self.NOVAS(n_iter = n_iter, n_samples = n_samples, sigma = sigma, gamma = gamma, threshold=threshold)
		return lopt, Qopt, loss, perc



	def NOVAS(self, n_iter, n_samples, sigma , gamma, threshold = 1.):
		mu = np.concatenate((self.l,self.Q.reshape((self.n_nodes-1)* self.n_targets)),axis=0)
		dim = len(mu)
		sigma = np.ones((dim,))
		sigma[0:self.n_nodes-1] = 0.1
		inv_dist = 1./np.sum(self.targets**2,axis=1)
		for i in range(n_iter+1):
			deltax = sigma*np.random.normal(size=(n_samples,dim))
			x = mu[np.newaxis,:]+deltax
			x = np.clip(x,self.lower_bounds,self.upper_bounds)
			fx = -self.NOVAS_loss(x,inv_dist)
			fx = fx - fx.min() 
			fx = fx/(fx.max() - fx.min())
			S = special.softmax(gamma*fx)
			mu = mu + (S[:,np.newaxis]*deltax).sum(axis=0) 
			mu = np.clip(mu,self.lower_bounds,self.upper_bounds)
			sigma = np.sqrt((S[:,np.newaxis]*deltax**2).sum(axis=0)+1e-8)
			obj = self.NOVAS_loss(mu[np.newaxis,:],inv_dist,print_idx=self.verbose)
			if (i+1)%1==0 and self.verbose:
				print('Iter:{}, obj. value={:.3f}, avg. sigma={:.3e}'.format(i, np.squeeze(obj), sigma.mean()))
			if obj<0.5:
				break

		lopt = mu[:self.n_nodes-1]
		Qopt = mu[self.n_nodes-1:].reshape((self.n_nodes-1, self.n_targets))
		_,losses = self.loss(lopt,Qopt,self.targets,self.orientations,inv_dist,return_idx_losses = True)
		print('Losses: ', losses)
		success = np.nonzero(losses<threshold)
		success_rate = len(success[0])/self.n_targets

		return lopt, Qopt, obj, success_rate


	def Link_budget_search(self,targets,orientations,link_budget=8):
		node_budget = link_budget+1
		Loss = np.zeros((node_budget+1,))
		Loss[0:3] = np.inf
		SR = np.zeros((node_budget+1,))
		Res = []
		for i in range(3,node_budget+1):
			print('Testing ', i , 'nodes...')
			l, Q, loss, perc = self.Optimize_design(n_nodes=i,targets=targets, orientations=orientations, n_iter = 400, n_samples = 1000, sigma = 1.0, gamma = 10.0, verbose = True)
			print('-------------------',i,' nodes, optimal loss: ', loss,' success rate: ', perc,' ----------------------')
			Loss[i] = loss
			SR[i] = perc
			Res.append([l, Q, loss, perc])

		opt_nodes = np.argmin(Loss)
		print('Optimal node number: ', opt_nodes, ', optimal loss: ', Loss[opt_nodes], ', SR: ', SR[opt_nodes])
		lopt, Qopt, loss_opt, SR_opt = Res[opt_nodes-3]
		self.SetDesignPar(opt_nodes)
		self.plot_configuration(lopt,Qopt,targets,orientations)
		print('Link lengths: ', lopt)
		print('Robot length: ', lopt.sum())
		print('Q: ', Qopt)



	def exec_NOVAS(self, lopt, new_targets, new_orientations, n_iter = 150, n_samples = 500, sigma = 1.0, gamma = 10.0, threshold=1., verbose = False):
		self.l = lopt
		mu = np.zeros(((self.n_nodes-1)*new_targets.shape[0],))
		self.a = 30*np.pi/180
		Q_min = -self.a*np.ones((self.n_nodes-1,new_targets.shape[0])) #minimum bending angle
		Q_min[0,:] = -np.pi #base can rotate freely
		Q_max = self.a*np.ones((self.n_nodes-1,new_targets.shape[0])) #minimum bending angle
		Q_max[0,:] = np.pi
		lower_bounds = Q_min.reshape((self.n_nodes-1)*new_targets.shape[0])
		upper_bounds = Q_max.reshape((self.n_nodes-1)*new_targets.shape[0])
		dim = len(mu)
		inv_dist = 1.0
		for i in range(n_iter+1):
			deltax = sigma*np.random.normal(size=(n_samples,dim))
			x = mu[np.newaxis,:]+deltax
			x = np.clip(x,lower_bounds,upper_bounds)
			fx = -self.NOVAS_exec_loss(x,new_targets,new_orientations,inv_dist)
			fx = fx - fx.min() 
			fx = fx/(fx.max() - fx.min())
			S = special.softmax(gamma*fx)
			mu = mu + (S[:,np.newaxis]*deltax).sum(axis=0) 
			mu = np.clip(mu,lower_bounds,upper_bounds)
			sigma = np.sqrt((S[:,np.newaxis]*deltax**2).sum(axis=0)+1e-8)
			obj = self.NOVAS_exec_loss(mu[np.newaxis,:],new_targets,new_orientations,inv_dist,print_idx=verbose)
			if (i+1)%1==0 and verbose:
				print('Iter:{}, obj. value={:.3f}, avg. sigma={:.3e}'.format(i, np.squeeze(obj), sigma.mean()))
			if obj<1.0:
				break
		print('Obj. value={:.3f}, avg. sigma={:.3e}'.format(np.squeeze(obj), sigma.mean()))
		return mu.reshape((self.n_nodes-1, new_targets.shape[0])), obj  


	def NOVAS_exec_loss(self,x,targets,orientations,inv_dist,print_idx=False):
		L = np.zeros(x.shape[0],)
		for i in range(x.shape[0]):
			q = x[i,:].reshape((self.n_nodes-1, targets.shape[0]))
			L[i]= self.loss(self.l,q,targets,orientations,inv_dist,print_idx = print_idx)
		return L 


	def Execution_phase_test(self,lopt, M=1000):
		np.random.seed(1)
		self.n_nodes = len(lopt)+1
		new_targets = 0.1+0.8*np.random.rand(M,2)
		new_orientations = np.pi/180*(-90+180*np.random.rand(M,))
		LOSS = np.zeros((M,))
		for i in range(M):
			print('-- Running iteration: ', i+1,' --')
			target = np.array(new_targets[i,:])
			target = target[np.newaxis,:] 
			orient = np.array([new_orientations[i]])
			Qnew, loss = self.exec_NOVAS(lopt=lopt,new_targets=target,new_orientations=orient,n_iter = 150, n_samples = 800, sigma = 1.0, gamma = 10.0)
			LOSS[i] = loss
		success = np.nonzero(LOSS<1.)
		success_rate = len(success[0])/M
		print('Success rate: ', success_rate)
		return success_rate

#	def ws_recursion(self,v,l,th,theta_range,n,debth=0):
#		if debth == len(l):
#			return
#		else:
#			lc = l[debth]
#			for theta in theta_range:
#				th1 = th + theta
#				w = v + np.array([lc*np.cos(th), lc*np.sin(th)])
#				P = list(self.sample_line_segment(v,w,3))
#				self.W.extend([(th1, p[0], p[1]) for p in P])
#				#bp()
#				self.ws_recursion(w,l,th1,theta_range,n,debth+1)
#
#
#	def sample_line_segment(self,v,w,n):
#		t = np.linspace(0.,1.,num=n)
#		P = np.zeros((n,2))
#		P[:,:] =v[np.newaxis,:]+t[:,np.newaxis]*(w[np.newaxis,:]-v[np.newaxis,:])
#		return P
#
#
#	def Calculate_workspace(self,l, th0, th1, n):
#		self.W = []
#		theta = np.linspace(-30,30,num=n)*np.pi/180
#		self.ws_recursion(np.array([0.,0.]),l,0,theta,n)
#		WS = np.array(self.W)
#		print(WS.shape)
#		bp()
#		from matplotlib import pyplot as plt
#		fig = plt.figure()
#		ax = fig.add_subplot(111, projection='3d')
#		ax.scatter(WS[:,1], WS[:,2], WS[:,0]*180./np.pi)
#		ax.set_xlabel('x ')
#		ax.set_ylabel('y ')
#		ax.set_zlabel('theta')
#
#		#ax = plt.axes(projection='3d')
#		#ax.plot_trisurf(WS[:,1], WS[:,2], WS[:,0]*180./np.pi, cmap='viridis', edgecolor='none')
#		plt.show()
#		bp()





if __name__ =='__main__':
	sess = MultiLinkBody()

	##generate targets automatically:
	#targets, orientations = sess.TargetGeneration(n_nodes=6,n_targets=4)
	#bp()

	#or pick specific targets
	targets = np.array([[0.8, 0.6], 
						[0.6, 0.25],
						[0.9, 0.4],
						[0.4, 0.65]])					
	orientations = np.pi/180*np.array([0.0, 15.0, -30.0, 90.0])
	seed = 0 #optimization seed
											#	#nodes = #links+1
	lopt, Qopt, loss, perc = sess.Optimize_design(n_nodes=6,targets=targets, orientations=orientations,seed=seed, n_iter = 250, n_samples = 2000, sigma = 1.0, gamma = 10.0, verbose = True)
	print('Design: Target success rate: ', perc)
	idx, losses = sess.plot_configuration(lopt,Qopt,targets=targets,orient=orientations)
	print('Link lengths: ', lopt)
	print('Robot length: ', lopt.sum())
	print('Q: ', Qopt)

	#bp() #continue to run design optimization for different link budgets
	#sess.Link_budget_search(targets,orientations,link_budget = 8)



	bp() #continue to run testing of execution phase
	#lopt = np.array([0.44436697, 0.1740868,  0.26990801, 0.1,  0.11443017])
	sr = sess.Execution_phase_test(lopt)
	bp()

	#bp() #continue to run robot workspace calculation
	#lopt = np.array([0.44436697, 0.1740868,  0.26990801, 0.1,  0.11443017])
	#th0 = -45*np.pi/180.
	#th1 = 45*np.pi/180.
	#n = 5
	#sess.Calculate_workspace(lopt, -30, 30, n)



