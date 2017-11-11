import numpy as np

class Classifier():

	def __init__(self,**dict):
		from sklearn import svm
		self.clf = svm.SVC()


	@classmethod
	def create_classifier(cls,**dict):
		if(dict['name']!='SVM'):
			print 'name %s not defined as a valid classifier'%(dict['name'])
			return None
		return cls()

	
	def fit(self,X,y):
		self.clf.fit(X,y)


	def predict(self,X):
		return self.clf.predict(X)
			

cl = Classifier.create_classifier(name="SVM")
X=np.array([[2,3],[4,5]])
y = np.array([[0],[1]])
cl.fit(X,y)
print cl.predict(np.array([10,2]))	