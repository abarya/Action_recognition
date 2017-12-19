import numpy as np
import matplotlib.pyplot as plt
from fourier_hist import fHist

class fourier:

	def __init__(self,time_series,num_levels,num_bins):
		self.time_series = time_series;
		self.num_levels = num_levels;
		self.num_bins = num_bins;

	def calcFourier(self, input_series):
		coef = np.fft.fft(input_series); #Extracts all the coeffficients!
		return coef;


	def createPyramid(self,level,fobj): #It will return histogram of a particular level
		segment_size = int(len(self.time_series)/(1<<(level)));
		initial_pos = 0;
		while(initial_pos<self.time_series.size):
			final_pos = initial_pos+segment_size;
			if(final_pos>self.time_series.size):
				final_pos = self.time_series.size;
				break;
			fourier_input = self.time_series[initial_pos:final_pos];
			fourier_output = self.calcFourier(fourier_input);
			fobj.create_hist(level,fourier_output);
			initial_pos = final_pos;
			
		return fobj


	def createFeature(self): #Total feature vector construction
		
		fobj = fHist(self.num_bins,self.num_levels);
		for i in range(0,self.num_levels):
			fobj = self.createPyramid(i,fobj);	
			#print fobj.fourier_hist
		return fobj.fourier_hist		
		

#a = np.arange(-np.pi,np.pi,0.01);
#a = np.cos(a);
#fourier_pyramid = fourier(a,2,10);
#fans =fourier_pyramid.createFeature();
#print len(fans)
# print fourier_pyramid.feature_out.size;



