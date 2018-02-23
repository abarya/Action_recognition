import numpy as np

Range = 2.0 # -1 to +1

def createHistogramPerFrame(num_divisions,points):
	hist = np.zeros((num_divisions,1,num_divisions))
	minimum_coord = np.amin(points, axis=0)
	for i in range(len(points)):
		points[i]-=minimum_coord
		idx = min(int(((points[i][0])/Range)*float(num_divisions)),num_divisions-1)
		idy = 0#min(int(((points[i][1])/Range)*num_divisions),num_divisions-1)
		idz = min(int(((points[i][2])/Range)*float(num_divisions)),num_divisions-1)
		if idx<0 or idz<0:
			print "hello    jasdksdjk",idx,idz
		hist[idx][idy][idz]+=1
	return hist


def saveHistogram(hist,filepath):
	filename = filepath.split('/')[-2]+'/'+filepath.split('/')[-1]
	X = open('hist_data/'+filename,'w')
	np.savetxt(X,hist.reshape((1,hist.size)),delimiter=',')

def createHistogram(v,trans):
	hist_size = v.num_divisions * 1 * v.num_divisions
	#TODO: Is the first column of zeros required?
	final_hist = np.zeros((int(hist_size),1),np.float32)
	hist_descriptor = np.zeros((1,int(hist_size)),np.float32) #descriptor when fourier is not used

	for line in v.data_file:
		if line=="END":
			break	
		v.getPointsAndNormalize(line)
		v.getSkeletonPoints()
		if trans==True:
			v.translate()
		hist = createHistogramPerFrame(int(v.num_divisions),np.array(v.connecter_points))
		hist_descriptor+=hist.reshape((1,hist.size))
		final_hist = np.concatenate((final_hist,hist.reshape(hist.size,1)),axis=1) # Now, each row will become a time series by the end!!!
	# saveHistogram(hist_descriptor,v.filename)
	v.restoreDataFile()
	# shape of final_hist will be (hist.size*total_frames)
	return final_hist[:,1:],hist_descriptor


#sample run
# import visualization
# v = visualization.visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512174930.txt',20)
# final_hist = createHistogram(v,False)