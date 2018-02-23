import numpy as np
import os
import csv
import random
import sys

import visualization
import hist
import fourier_ag
import classifier
from sklearn.decomposition import PCA


activityToClassNumDict = {'rinsing mouth with water': 7, 'writing on whiteboard': 1, 'opening pill container': 2,
						'random': 12, 'talking on couch': 5, 'drinking water': 3, 'cooking (chopping)': 13, 'working on computer': 9,
						'relaxing on couch': 10, 'brushing teeth': 8, 'still': 11, 'talking on the phone': 0, 'cooking (stirring)': 6,
						'wearing contact lenses': 4}

directory = "/media/arya/54E4C473E4C458BE/Action_dataset/data";

# print sys.argv[1]
# num = int(sys.argv[1]) #number of the person for which the feature is extracted
def normalize(histogram):
	for i in range(len(histogram)):
		mean = np.mean(histogram[i]);
		var  = np.var(histogram[i]);
		if(var!=0):
			histogram[i] = (histogram[i]-mean)/var;
		else:
			histogram[i]= histogram[i] - mean;
	return histogram


def getFeatureVector(histogram):
	fout = np.zeros(20);
	for i in range(len(histogram)):
		fourier_pyramid = fourier_ag.fourier(histogram[i],2,10)
		fout += fourier_pyramid.createFeature()
	return fout


def getLabelDict(file_name):
	activityLabel = {}
	activity_file = open(file_name,'r')
	for line in activity_file:
		if line=="END":
			break
		line = line.split(',')[:-1]	
		activityLabel[line[0]+'.txt']=line[1]
	return activityLabel	


def createDescriptorAndSave(v,X,Y,activity_num,trans):
	histogram,hist_descriptor = hist.createHistogram(v,trans);
	#histogram = normalize(histogram);
	# feature_vec = getFeatureVector(histogram)
	#save feature vector and label to file
	# np.savetxt(X,feature_vec.reshape((1,feature_vec.size)),delimiter=',') 
	# np.savetxt(Y,activity_num)
	#save hist_descriptor feature vector and label to file
	np.savetxt(X,hist_descriptor.reshape((1,hist_descriptor.size)),delimiter=',') 
	np.savetxt(Y,activity_num)


def extractFeatures(directory):
	for x in range(num,num+1):
		X = open("train/train_x"+str(num)+".data","w")
		Y = open("train/train_y"+str(num)+".data","w")
		path = directory + str(x);
		label_file_path = path + '/activityLabel.txt';
		activityLabelDict = getLabelDict(label_file_path)
		files = []
		for file in os.listdir(path):
			if file.endswith(".txt"):
				files.append(file)
		i=0
		for file_name in files:
			print i, file_name
			i=i+1
			if file_name!='activityLabel.txt':
				file_path = path + '/' + file_name
				v = visualization.visualization(file_path,5)
				trans=False
				activity_num = np.array([activityToClassNumDict[activityLabelDict[file_name]]])
				createDescriptorAndSave(v,X,Y,activity_num,trans)
				# # apply translation to points about y-axis
				# trans=True
				# createDescriptorAndSave(v,X,Y,activity_num,trans)
                    

def computeAccuracy(x,y,model):
	counter=0
	for i in range(len(y)) :
		prediction=model.predict(x[i].reshape(1,-1))
		if(prediction==y[i]) :
			counter+=1
	accuracy=float(counter*100)/len(y)
	return accuracy


# extractFeatures(directory)

train_accuracy = 0
test_accuracy = 0
maximum = 0
for x in range(10):
	data_x = []
	label = []
	for count in range(1,5):
		filename_x = "train/train_x"
		filename_y = "train/train_y"
		filename_x = filename_x+str(count)+".data"
		with open(filename_x, "rb") as csvfile:
			lines = csv.reader(csvfile)
			lines = list(lines)
			data_x += lines
			for i in range(len(lines)*(count-1),len(data_x)):
				data_x[i] = [float(x) for x in data_x[i]]
		filename_y = filename_y+str(count)+".data"
		with open(filename_y, "rb") as csvfile:
			lines = csv.reader(csvfile)
			lines = list(lines)
			print lines,filename_y
			label += lines
			for i in range(len(lines)*(count-1),len(label)):
				print label[i],count,i
				label[i] = int(float(label[i][0]))


	data_x = np.array(data_x)
	# print data_x.shape
	# #reduce dimensionality
	# pca = PCA(n_components=500)
	# pca.fit(data_x)
	# data_x = pca.transform(data_x)
	# print data_x.shape


	trainingSet=[]
	training_label=[]
	testSet=[]
	test_label=[]
	for i in range(len(data_x))	 :
		if random.random() < 0.7:
	        	trainingSet.append(data_x[i])
	        	training_label.append(label[i])
		else :
			testSet.append(data_x[i])
			test_label.append(label[i])

	training_label = np.array(training_label)
	print training_label.shape,len(trainingSet)

	trainingSet=np.array(trainingSet)
	'''pca = PCA(n_components=20)
	pca.fit(trainingSet);
	trainingSet = pca.transform(trainingSet);
	testSet = np.array(testSet);
	testSet = pca.transform(testSet);'''

	model = classifier.Classifier.create_classifier(name="SVM")
	model.fit(trainingSet,training_label)

	current_train_accuracy = computeAccuracy(trainingSet,training_label,model)
	train_accuracy+=current_train_accuracy

	current_test_accuracy = computeAccuracy(testSet,test_label,model)
	test_accuracy+=current_test_accuracy
	maximum = max(maximum,current_test_accuracy)
	#train accuracy
	print('training accuracy is %d%% with training samples = %d'%(current_train_accuracy,len(training_label)))
	#test accuracy
	print('test accuracy is %d%% with test samples = %d'%(current_test_accuracy,len(test_label)))


	print np.array(data_x).shape,np.array(label).shape

# print feature_vector.shape
print 'training accuracy',train_accuracy/10,'%'
print 'test accuracy',test_accuracy/10,'%'
print 'maximum',maximum,'%'

