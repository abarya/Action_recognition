import numpy as np
import visualization

v = visualization.visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512164800.txt',3)
Range = 1.0 # -0.5 to +0.5


def createHistogram(num_divisions,points):
	hist = np.zeros((num_divisions,num_divisions,num_divisions))
	for i in range(len(points)):
		idx = min(int(((points[i][0]+0.5)/Range)*num_divisions),num_divisions-1)
		idy = min(int(((points[i][1]+0.5)/Range)*num_divisions),num_divisions-1)
		idz = min(int(((points[i][2]+0.5)/Range)*num_divisions),num_divisions-1)
		print idx,idy,idz
		hist[idx][idy][idz]+=1


for line in v.data_file:
	if line=="END":
		break
	v.getPointsAndNormalize(line)
	createHistogram(int(v.num_divisions),np.concatenate((v.p_x.reshape(len(v.p_x),1),v.p_y.reshape(len(v.p_x),1),v.p_z.reshape(len(v.p_x),1)),axis=1))	
