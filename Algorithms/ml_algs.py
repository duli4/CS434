import DataSet as ds
import numpy as np
import math

###################################################
#                Linear Regression                #
###################################################
	
def get_data_mwlin(mx, my):
	w0 = np.linalg.inv(np.matmul(mx.transpose(), mx))
	w1 = np.matmul(w0, mx.transpose())
	
	return np.matmul(w1, my)
	
def get_data_mglin(mw, mx):
	return np.matmul(mw.transpose(), mx.transpose()).transpose()
	
###################################################
#                Perceptron                       #
###################################################

#Batch 
def get_data_mwpb(mx, my, lr = 0.1, er = 0.01):
	mw = np.zeros((np.shape(mx)[1], 1))
	delta = np.zeros((np.shape(mx)[1], 1))
	
	while 1:
		for i in xrange(np.shape(mx)[0]):
			u = np.matmul(mw.transpose(), mx[i].transpose())
			if np.matmul(my[i], u) <=0:
				delta -= np.matmul(my[i], mx[i]).transpose()
		delta /= np.shape(mx)[0]
		mw -= lr*delta
		if np.linalg.norm(delta) < er:
			return mw
	
#Online
def get_data_mwpo(mx, my):
	mw = np.zeros((np.shape(mx)[1], 1))
	
	while 1:
		old_mw = np.copy(mw)
		for i in xrange(np.shape(mx)[0]):
			u = np.matmul(mw.transpose(), mx[i].transpose())
			if np.matmul(my[i], u) <=0:
				mw += np.matmul(my[i], mx[i]).transpose()
		if np.array_equal(old_mw, mw):
			return mw
			
#Voted
def get_data_mwpv(mx, my, epochs = 100):
	l_mw = []
	c_mw = np.zeros((np.shape(mx)[1], 1))
	surv_time = 0
	
	for i in xrange(epochs):
		for i in xrange(np.shape(mx)[0]):
			u = np.matmul(c_mw.transpose(), mx[i].transpose())
			if np.matmul(my[i], u) <=0:
				l_mw.append((surv_time, np.copy(c_mw)))
				surv_time = 0
				c_mw += np.matmul(my[i], mx[i]).transpose()
			else:
				surv_time += 1
	wa = np.zeros((np.shape(mx)[1], 1))
	for w in [l_mw[i][0]*l_mw[i][1] for i in xrange(len(l_mw))]:
		wa += w
	return wa
	

def get_data_mgp(mw, mx):
	return np.matrix([np.sign(val) for val in np.matmul(mw.transpose(), mx.transpose()).tolist()[0]]).transpose()

###################################################
#              Logisitic Regression               #
###################################################
	
#Batch
def get_data_mwlogb(mx, my, lr = 0.1, er = 1.0):
	mw = np.zeros((np.shape(mx)[1], 1))
	
	while 1:
		delta = np.zeros((np.shape(mx)[1], 1))
		for i in xrange(np.shape(mx)[0]):
			y_hat = get_sigmoid(mw, mx[i])
			delta +=  np.matmul((y_hat - my[i]), mx[i]).transpose()
		mw -= lr * delta
		
		if np.linalg.norm(delta) < er:	
			return mw
	
#Online
def get_data_mwlogo(mx, my, lr = 0.01, epochs = 100):
	mw = np.zeros((np.shape(mx)[1], 1))
	
	for i in xrange(epochs):
		old_mw = np.copy(mw)
		for i in xrange(np.shape(mx)[0]):
			y_hat = get_sigmoid(mw, mx[i])
			mw -= (lr * np.matmul((y_hat-my[i]), mx[i])).transpose()
	
	return mw
	
def get_data_mglog(mw, mx):
	return [0.0 if y < -700 else (1.0/(1.0+(math.exp(-1.0*y)))) for y in [np.matmul(mw.transpose(), mx[i].transpose()) for i in xrange(np.shape(mx)[0])]]
	
def get_sigmoid(mw, mx_row):
	y = np.matmul(mw.transpose(), mx_row.transpose())
	return 0.0 if y < -700 else (1.0 / (1.0 + (math.exp(-1.0*y))))
	
###################################################
#             K-Nearest Neighbour                 #
###################################################
def get_best_k_cv(mx, my, max_k, loo = 0, minf = 1, maxf = 20):
	folds = ds.get_highest_factor(minf,maxf,np.shape(mx)[0])
	lx = np.vsplit(mx, folds)
	ly = np.vsplit(my, folds)
	errs = []
	for i in xrange(folds):
		vx = lx[i]
		vy = ly[i]
		trn_xs = get_joined_matrices(lx[:i]+lx[i+1:])
		trn_ys = get_joined_matrices(ly[:i]+ly[i+1:])
		errs.append(get_k_errs(trn_xs, trn_ys,vx, vy, max_k, loo))
	
	total_avg_errs = get_avg_errs(errs, folds, (max_k-1)/2)
	print total_avg_errs
	
	return list(sorted(total_avg_errs, key=lambda t: t[1]))[0][0]
			
def get_avg_errs(errs, folds, ks):
		raw_errs = [[errs[i][j][1] for i in xrange(len(errs))] for j in xrange(ks)]
		return [(1+2*i, sum(raw_errs[i])/(folds-1.)) for i in xrange(len(raw_errs))]

def get_k_errs(mx_trn, my_trn, mx_tst, my_tst, max_k = 21, loo = 0):
	l_ke = []
	for k_cur in xrange(1, max_k, 2):
		errs = 0
		gs = [get_knn_g(mx_tst[i], mx_trn, my_trn, k_cur, loo) for i in xrange(np.shape(mx_tst)[0])]
		errs += sum([abs(my_tst[i]-gs[i])/2 for i in xrange(np.shape(gs)[0])])
		l_ke.append((k_cur, errs))
		
	return l_ke

def get_knn_g(p, mx, my, k, loo = 0):
	l = [(i, get_similarity(p, mx[i])) for i in xrange(np.shape(mx)[0])]
	lp = list(sorted(l, key=lambda t: t[1]))[loo:k+loo]
	vote = sum([my[lp[i][0]] for i in xrange(k)])
	
	return np.sign(vote)

def get_similarity(mx0, mx1):
	return np.linalg.norm(mx0 - mx1)
	
def get_joined_matrices(lm):
	m = np.copy(lm[0])
	for i in xrange(1, len(lm)):
		m = np.concatenate((np.copy(m), lm[i]))
		
	return m

###################################################
#                  Decision Trees                 #
###################################################

class TreeNode: 
	def __init__(self, lm, depth = 0, max_depth = 6, prediction = None):
		self.depth = depth
		self.max_depth = max_depth
		self.prediction = prediction
		self.left = None
		self.right = None
		if prediction is not None:
			self.is_leaf = true
			self.threshold = None
		else:
			self.is_leaf = false
			self.threshold = self.get_threshold(lm)
		
	def predict(self, p):
		if self.is_leaf:
			return self.prediction
		else:
			if self.threshold(p):
				return self.left.predict(p)
			else: 
				return self.right.predict(p)
	
	def get_threshold(self, m):
		print "GT"

#def get_decision_tree():

def get_best_split(mx, my):
	print "GBS"
	
def get_best_thresh(lm):
	ly = [row[0] for row in lm]
	slm = list(sorted(lm, key=lambda l: l[1]))
	g = []
	for i in xrange(len(slm)-1):
		lys = []
		lys.append([slm[j][0] for j in xrange(i+1)])
		lys.append([slm[j][0] for j in xrange(i, len(slm) - i)])
		print i
		g.append((slm[i][1], get_info_gain(ly, lys)))
		del lys
		
	print g
	sg = list(sorted(g, key=lambda t: -t[1]))
	return sg[0]
	
def get_info_gain(ly, lys):
	us = [get_uncertainty(lys[i]) for i in xrange(len(lys))]
	us_sum = sum([float(len(lys[i]))/len(ly)*us[i] for i in xrange(len(us))])
	
	return get_uncertainty(ly) - us_sum

def get_uncertainty(ly, type = 'entropy'):	
	p_plus = float(sum([1 for p in ly if p == 1])) / len(ly)
	p_minus = float(sum([1 for p in ly if p == -1])) / len(ly)
	if type == 'entropy':
		uncertainty = -1*((p_plus*(math.log(p_plus, 2))) if p_plus > 0 else 0) + ((p_minus*(math.log(p_minus, 2))) if p_minus > 0 else 0)
	elif type == 'error':
		uncertainty = min(p_plus, p_minus)
	elif type == 'gini':
		uncertainty = p_minus*p_plus
		
	return uncertainty
	
