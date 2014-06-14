import numpy as np
import pylab as pl

class SMP():

### this is a soft margin perceptron also supporting kernels

	def __init__(self, max_it = 4400, kernel = 'inner', degree=3, disp=1):
		self.max_it = max_it
		self.disp = disp
		if kernel == 'poly':
			self.kernel = lambda x, y: (np.inner(x, y) + 1)**degree
		elif kernel == 'gaussian':
			self.kernel = lambda x, y: np.exp(np.inner(x - y, x - y)/1)
		else: # kernel == 'inner'
			self.kernel = np.inner
			
	def train(self, X, y, C = 0.1):
		self.fit(X, y, C)
		
	def fit(self, X, y, C = 0.1, lam = 1, sC = 0.1, mu = 0.1, learning_rate = 0.02):
		
		# coefficients
		alpha = np.zeros(np.shape(X)[0])
		b = 0.0
		xi = np.zeros(np.shape(X)[0])
		
		# build kernel matrix
		K = np.asarray( self.kernel(X, X) )
		
		# index fields
		indices = [i for i in range(len(y))]
		index_count = [0] * len(y)
		first_recount = True
		threshold = min(len(y), self.max_it)/5
		
		# loop
		k = 0; loop = True; 
		while loop and k < self.max_it:
			k = k + 1; loop = False
			
			alpha *= (1.0 - C * learning_rate)
			
			# cycle through samples
			for i in indices:
				update = 0. if xi[i] >= 0 else mu
				
				# misclassification
				if 1 - xi[i] - y[i] * (np.inner(y*alpha,K[:, i]) + b) >= 0 :
					loop = True
					update += lam
					alpha[i] += learning_rate * lam
					b += learning_rate * lam*y[i]
					index_count[i] = 0	
												
				# correct classification 
				else:
					index_count[i] += 1
					if index_count[i] > threshold:
						indices.remove(i)
				
				xi[i] -= learning_rate * (sC - update)
		
			# use all arrays before reaching the end. 
			if k == self.max_it - threshold or loop == False:
				if first_recount : loop = True
				first_recount = False
				indices = [i for i in range(len(y))]
		
		# save results
		self.alpha = alpha
		self.b = b
		self.X = X
		self.y = y
		
		if self.disp > 0:
			print  'k =', k
		if self.disp > 1:
			print 'alpha =', self.alpha
		
	# get posterior probability for a single samples mainly	
	def classify(self, z):
		z = np.asarray(z)
		K = np.asarray( map(lambda x: self.kernel(x, z), self.X) )
		return np.dot(self.y*self.alpha, K) + self.b

	# classify for lists of samples
	def predict(self, grid): 
		t = []
		for z in grid:
			if self.classify(z) < 0:
				t.append(0)
			else: t.append(1)
		return np.asarray(t);
		
