import numpy as np

Range = 1.0 # -0.5 to +0.5

def createHistogramPerFrame(num_divisions,points):
	hist = np.zeros((num_divisions,num_divisions,num_divisions))
	for i in range(len(points)):
		idx = min(int(((points[i][0]+0.5)/Range)*num_divisions),num_divisions-1)
		idy = min(int(((points[i][1]+0.5)/Range)*num_divisions),num_divisions-1)
		idz = min(int(((points[i][2]+0.5)/Range)*num_divisions),num_divisions-1)
		hist[idx][idy][idz]+=1
	return hist;


def createHistogram(v):
	hist_size = v.num_divisions * v.num_divisions * v.num_divisions;
	#TODO: Is the first column of zeros required?
	final_hist = np.zeros((int(hist_size),1),np.float32);

	for line in v.data_file:
		if line=="END":
			break	
		v.getPointsAndNormalize(line);
		hist = createHistogramPerFrame(int(v.num_divisions),np.concatenate((v.p_x.reshape(len(v.p_x),1),v.p_y.reshape(len(v.p_x),1),v.p_z.reshape(len(v.p_x),1)),axis=1));
		final_hist = np.concatenate((final_hist,hist.reshape(hist.size,1)),axis=1) # Now, each row will become a time series by the end!!!

	# shape of final_hist will be (hist.size*total_frames)
	return final_hist[:,1:]


#sample run
import visualization
v = visualization.visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512174930.txt',3)
final_hist = createHistogram(v)