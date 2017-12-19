# IT CREATES THE FOURIER HISTOGRAM #

import numpy as np

class fHist():
	
	def __init__(self,num_bins,num_levels):
		self.bin_size = 50/num_bins; # we use 50 because max possible freq*100 is 50, and divide it by num_bins
		self.num_bins = num_bins;
		self.num_levels = num_levels;
		self.fourier_hist = np.zeros(num_bins*num_levels);
	
	def create_hist(self,level,farray):
		start = (level)*self.num_bins;
		freq = np.fft.fftfreq(len(farray))*100;
		#print freq
		bin_num = start;
		com_val = self.bin_size;
		i=0;
		for num in freq:
	 		if num<0:
				break;
			elif num<=com_val:
				self.fourier_hist[bin_num]+= np.abs(farray[i]);
			else:
				com_val+=self.bin_size;
				bin_num+=1;
				self.fourier_hist[bin_num]+=np.abs(farray[i]);
			i+=1;
		
			
			
	


