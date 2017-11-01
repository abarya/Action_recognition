import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class visualization():

	def __init__(self,filename):
		self.filename = filename
		self.enableInteractivePlot()
		self.ax = plt.figure().add_subplot(111, projection='3d') #create Axes3D object
		self.data_file = open(self.filename,'r') #'/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512164529.txt'
		self.points = np.array([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]])
	

	def enableInteractivePlot(self):
		plt.ion()
	
	
	def getPointsAndNormalize(self,line):
		self.getPoints(line)
		self.normalize()


	def normalize(self):
		from sklearn.preprocessing import normalize
		self.p_x = normalize(self.p_x,axis=0).ravel()
		self.p_y = normalize(self.p_y,axis=0).ravel()
		self.p_z = normalize(self.p_z,axis=0).ravel()


	def getPoints(self,line):
		line=line.split(',')
		line=[float(x) for x in line[:len(line)-1]]
		self.p_x = []
		self.p_y = []
		self.p_z = []
		k=0
		while(k<11):
			i=11+k*14
			self.p_x.append(line[i])
			self.p_z.append(line[i+1])
			self.p_y.append(line[i+2])
			k=k+1
		k=0	
		while(k<4):
			i=155+k*4
			self.p_x.append(line[i])
			self.p_z.append(line[i+1])
			self.p_y.append(line[i+2])
			k=k+1	

		self.p_x = np.array(self.p_x).reshape(len(self.p_x),1)
		self.p_y = np.array(self.p_y).reshape(len(self.p_y),1)
		self.p_z = np.array(self.p_z).reshape(len(self.p_z),1)


	def setLimits(self):
		self.ax.set_xlim(0,1)
		self.ax.set_ylim(0,1)
		self.ax.set_zlim(0,1)


	def setLabels(self):
		self.ax.set_xlabel('xlabel')
		self.ax.set_ylabel('ylabel')
		self.ax.set_zlabel('zlabel')


	def plot(self,j1,j2):
		self.ax.plot([self.p_x[j1],self.p_x[j2]],[self.p_y[j1],self.p_y[j2]],[self.p_z[j1],self.p_z[j2]],'b')


	def plotJoints(self):
		for i in range(4):
			self.plot(i,i+1)
		self.plot(5,6)	
		self.plot(1,2)
		self.plot(2,5)
		self.plot(3,5)
		self.plot(3,4)
		self.plot(4,11)
		self.plot(6,12)
		self.plot(7,8)
		self.plot(7,9)
		self.plot(9,10)
		self.plot(2,7)
		self.plot(2,9)
		self.plot(10,14)
		self.plot(8,13)


	def show(self):
		for line in self.data_file:
			if line=="END":
				break
			self.setLimits()
			self.setLabels()
			self.getPointsAndNormalize(line)
			self.plotJoints()
			self.ax.plot(self.p_x,self.p_y,self.p_z,'o')
			r = [-1,1]
			X, Y = np.meshgrid(r, r)
			# plot vertices
			self.ax.scatter3D(self.points[:, 0], self.points[:, 1], self.points[:, 2])
			plt.draw()
			plt.pause(0.02)
			self.ax.cla()

#sample object
visual = visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512164529.txt')
visual.show()