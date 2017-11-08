import numpy as np


def calcFourier(input_series , num_samples):
	fourier  = np.fft.fft(input_series);
	start = fourier[:(num_samples/2)];
	end = fourier[(fourier.size-num_samples/2):];
	ans = np.concatenate((start,end));
	print ans.size
	return ans;

def createPyramid(time_series, level):

	segment_size = int(len(time_series)/(1<<level));
	initial_pos = 0;
	while(initial_pos<time_series.size):
		final_pos = initial_pos+segment_size;
		if(final_pos>time_series.size):
			final_pos = time_series.size;
		fourier_input = time_series[initial_pos:final_pos];
		print fourier_input;
		initial_pos = final_pos;

a = np.arange(-np.pi,np.pi,0.01);
ans = calcFourier(a,20);
