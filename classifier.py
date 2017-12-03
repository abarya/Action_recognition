import numpy as np

class Classifier():

	def __init__(self,**dict):
		if dict['name']=="SVM":
			from sklearn import svm
			self.clf = svm.SVC(kernel='rbf')
		else:
			from sklearn import linear_model
			self.clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
													intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear',
													max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
				

	@classmethod
	def create_classifier(cls,**dict):
		if(dict['name']!='SVM'):
			return cls(name=dict['name'])
		return cls(name="")

	
	def fit(self,X,y):
		self.clf.fit(X,y)


	def predict(self,X):
		return self.clf.predict(X)
			

# cl = Classifier.create_classifier(name="SVM")
# X=np.array([[2,3],[4,5]])
# y = np.array([[0],[1]])
# cl.fit(X,y)
# print cl.predict(np.array([10,2]))	