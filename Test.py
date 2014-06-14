from SMP import SMP
import sklearn.datasets 
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap


class Classifier_1vsA():
	def __init__(self, classifier, max_it = 400, disp=1):
		self.disp = disp
		self.classifier = classifier
		self.classifiers = []
	
	def fit(self, X, y):
		self.classlabels = list(set(y))
		if len(self.classlabels) < 2:
			self.classlabels = []
			print "There is nothing to classify here! Less than two class labels"
			return
		if len(self.classlabels) == 2:
			self.classlabels = [self.classlabels[0]]
		for l in self.classlabels:
			self.classifiers.append( self.classifier(disp=self.disp) )
			OneVsAll_y = map(lambda x: 1 if x == l else -1, y)
			self.classifiers[l].fit(X, OneVsAll_y)
			
	def classify(self, z):
		if len(self.classlabels) == 1:
			return [ self.classifiers[0].classify(z), 1 - self.classifiers[0].classify(z) ]
			
		posteriors = []
		sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x*0.0001)) 
		for l in self.classlabels:
			posterior = sigmoid(self.classifiers[l].classify(z))
			posteriors.append(posterior)
		#print posteriors
		return np.asarray(posteriors)
		
	def predict(self, grid): 
		t = []
		for z in grid:
			t.append(np.argmax( self.classify(z) ))
		return np.asarray(t);


''' get light and bold colour maps '''
def getColourMaps():
	cmaps = []
	cmaps.append(ListedColormap(['#FFFEFF', '#AAFFbb', '#AAAAFF', '#AB0AFF', '#ABEAFF', '#EBEAFF', '#FBEA11', '#FBEA99']))
	cmaps.append(ListedColormap(['#00FE00', '#00FFbb', '#0000FF', '#AA0AFF', '#ABEAFF', '#EBEAFF', '#FBEA11', '#FBEA99']))
	return cmaps[0], cmaps[1]
	
def plot_classification(X, y, classifier, name='std', transform='id'):
	steps = 10.0
	x_min, x_max = X[:, 0].min() , X[:, 0].max() 
	y_min, y_max = X[:, 1].min() , X[:, 1].max() 
	h = min( abs(x_max - x_min), abs(y_max - y_min)) / steps
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	grid = np.c_[xx.ravel(), yy.ravel()]
	if transform != 'id':
		grid = np.asarray(map(transform, grid))
	Z = classifier.predict(grid)
	Z = Z.reshape(xx.shape)
	
	cmap_light, cmap_bold = getColourMaps()
	pl.figure()		
	pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
	pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
	print("Saving to file ./plots/%s.png" %name)
	pl.savefig('./plots/%s.png' %name)
	
def Test() :
	iris = sklearn.datasets.load_iris()
	X, y = iris.data, iris.target
	#y = map(lambda x: 1 if x == 2 else x, y)
	#X = X[:,1:3]
	cl3 = Classifier_1vsA(SMP)
	cl3.fit(X, y)
	print "Error rate:", sum(y != cl3.predict(X))*1.0/len(y)*100, "%"
	#plot_classification(X, y, cl3)

# run test
Test()

