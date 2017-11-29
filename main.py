import numpy as np
import os
import csv
import random
import visualization
import hist
import fourier_ag
import classifier

FOURIER_COEFF_LENGTH = 60
activityToClassNumDict = {'rinsing mouth with water': 7, 'writing on whiteboard': 1, 'opening pill container': 2,
						'random': 12, 'talking on couch': 5, 'drinking water': 3, 'cooking (chopping)': 13, 'working on computer': 9,
						'relaxing on couch': 10, 'brushing teeth': 8, 'still': 11, 'talking on the phone': 0, 'cooking (stirring)': 6,
						'wearing contact lenses': 4}
directory = "/media/arya/54E4C473E4C458BE/Action_dataset/data"

def normalize(histogram):
	for i in range(len(histogram)):
		mean = np.mean(histogram[i]);
		var  = np.var(histogram[i]);
		histogram[i] = (histogram[i]-mean)/var;
	return histogram


def getFeatureVector(histogram):
	feature_vector = np.zeros((len(histogram)*FOURIER_COEFF_LENGTH),dtype='float')
	#histogram = normalize(histogram)
	for i in range(len(histogram)):
		fourier_pyramid = fourier_ag.fourier(histogram[i],2,20)
		fourier_pyramid.createFeature()
		s = FOURIER_COEFF_LENGTH*i
		e = FOURIER_COEFF_LENGTH*(i+1)
		feature_vector[s:e] = fourier_pyramid.feature_out
	return feature_vector


def getLabelDict(file_name):
	activityLabel = {}
	activity_file = open(file_name,'r')
	for line in activity_file:
		if line=="END":
			break
		line = line.split(',')[:-1]	
		activityLabel[line[0]+'.txt']=line[1]
	return activityLabel	


def extractFeatures(directory):
	for x in range(1,5):
		path = directory + str(x)
		label_file_path = path + '/activityLabel.txt'
		activityLabelDict = getLabelDict(label_file_path)
		files = []
		for file in os.listdir(path):
			if file.endswith(".txt"):
				files.append(file)

		X = open("train/train_x.data","a")
		Y = open("train/train_y.data","a")

		i=0
		for file_name in files:
			print i, file_name
			i=i+1
			if file_name!='activityLabel.txt':
				file_path = path + '/' + file_name
				v = visualization.visualization(file_path,3)
				histogram = hist.createHistogram(v)
				feature_vec = getFeatureVector(histogram)
				np.savetxt(X,feature_vec.reshape((1,feature_vec.size)),delimiter=',') 
				np.savetxt(Y,np.array([activityToClassNumDict[activityLabelDict[file_name]]]))


#extractFeatures(directory)
filename_x = "train/train_x.data"
filename_y = "train/train_y.data"
data_x = []
label = []

text_file = open(filename_x, "r")

with open(filename_x, 'rb') as csvfile:
	lines = csv.reader(csvfile)
	data_x = list(lines)
	for i in range(len(data_x)):
		data_x[i] = [float(x) for x in data_x[i]]

with open(filename_y, 'rb') as csvfile:
	lines = csv.reader(csvfile)
	label = list(lines)
	for i in range(len(label)):
		label[i] = int(float(label[i][0]))

trainingSet=[]
training_label=[]
testSet=[]
test_label=[]
for i in range(len(data_x))	 :
	if random.random() < 0.8:
        	trainingSet.append(data_x[i])
        	training_label.append(label[i])
	else :
		testSet.append(data_x[i])
		test_label.append(label[i])

training_label= np.array(training_label)
trainingSet=np.array(trainingSet)

model = classifier.Classifier.create_classifier(name="SVM")
model.fit(trainingSet,training_label)

counter=0
for i in range(len(training_label)) :
	prediction=model.predict(trainingSet[i])
	if(prediction==training_label[i]) :
		counter+=1

accuracy=float(counter)/len(training_label)
print "counter", counter, len(training_label)
print "accuracy", accuracy

print np.array(data_x).shape,np.array(label).shape

# print feature_vector.shape


