import numpy as np
import matplotlib.pyplot as plt

class fourier:


	def __init__(self,time_series,num_levels,num_samples):
		self.time_series = time_series;
		self.num_levels = num_levels;
		self.num_samples = num_samples;


	def calcFourier(self, input_series):
		real_coef = np.fft.fft(input_series); #Extracts all the coeffficients!
		fourier_mag = np.abs(real_coef); #Converts complex coef to magnitude of complex coefs
		start = fourier_mag[:(self.num_samples/2)];
		end = fourier_mag[(fourier_mag.size-self.num_samples/2):];
		ans = np.concatenate((start,end));
		return ans;


	def createPyramid(self,level): #It will return concatenated array of a particular level
		segment_size = int(len(self.time_series)/(1<<level));
		initial_pos = 0;
		while(initial_pos<self.time_series.size):
			final_pos = initial_pos+segment_size;

			if(final_pos>self.time_series.size):
				final_pos = self.time_series.size;
				break;
			fourier_input = self.time_series[initial_pos:final_pos];
			fourier_output = self.calcFourier(fourier_input);
			if(initial_pos == 0):
				fourier_ans = fourier_output;
			else:
				fourier_ans = np.concatenate((fourier_ans,fourier_output));

			initial_pos = final_pos;
		return fourier_ans


	def createFeature(self): #Total feature vector construction

		for i in range(0,self.num_levels):
			feature = self.createPyramid(i);
			if(i==0):
				self.feature_out = feature;
			else:
				self.feature_out = np.concatenate((self.feature_out,feature));
		

# a = np.arange(-np.pi,np.pi,0.01);
# a = np.cos(a);
# fourier_pyramid = fourier(a,2,20);
# fourier_pyramid.createFeature();

# print fourier_pyramid.feature_out;
# print fourier_pyramid.feature_out.size;



