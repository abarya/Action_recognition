import numpy as np
import visualization
import hist
import fourier_ag

FOURIER_COEFF_LENGTH = 60

v = visualization.visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512164800.txt',3)

histogram = hist.createHistogram(v)

feature_vector = np.zeros((len(histogram)*FOURIER_COEFF_LENGTH),dtype='float')

for i in range(len(histogram)):
	fourier_pyramid = fourier_ag.fourier(histogram[i],2,20)
	fourier_pyramid.createFeature()
	s = FOURIER_COEFF_LENGTH*i
	e = FOURIER_COEFF_LENGTH*(i+1)
	feature_vector[s:e] = fourier_pyramid.feature_out

print feature_vector.shape	


