import numpy as np

Range = 2.0 # -1 to +1

def createHistogramPerFrame(num_divisions,points):
	hist = np.zeros((num_divisions,1,num_divisions))
	minimum_coord = np.amin(points, axis=0)
	for i in range(len(points)):
		points[i]-=minimum_coord
		idx = min(int(((points[i][0])/Range)*num_divisions),num_divisions-1)
		idy = 0#min(int(((points[i][1])/Range)*num_divisions),num_divisions-1)
		idz = min(int(((points[i][2])/Range)*num_divisions),num_divisions-1)
		if idx<0 or idz<0:
			print "hello    jasdksdjk",idx,idz
		hist[idx][idy][idz]+=1
	return hist


def createHistogram(v,trans):
	hist_size = v.num_divisions * 1 * v.num_divisions
	#TODO: Is the first column of zeros required?
	final_hist = np.zeros((int(hist_size),1),np.float32)

	for line in v.data_file:
		if line=="END":
			break	
		v.getPointsAndNormalize(line)
		v.getSkeletonPoints()
		if trans==True:
			v.translate()
		hist = createHistogramPerFrame(int(v.num_divisions),np.array(v.connecter_points))
		final_hist = np.concatenate((final_hist,hist.reshape(hist.size,1)),axis=1) # Now, each row will become a time series by the end!!!
	v.restoreDataFile()
	# shape of final_hist will be (hist.size*total_frames)
	return final_hist[:,1:]


#sample run
# import visualization
# v = visualization.visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512174930.txt',3)
# final_hist = createHistogram(v)