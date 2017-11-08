import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class visualization():

	def __init__(self,filename,num_divisions,visualize=False):
		self.filename = filename
		self.enableInteractivePlot()
		self.joints_dict()
		if visualize == True:
			self.ax = plt.figure().add_subplot(111, projection='3d') #create Axes3D object
		self.data_file = open(self.filename,'r') #'/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512164529.txt'
		self.num_divisions = float(num_divisions)
	

	def joints_dict(self):
		self.dict = {1 : 'HEAD', 2 : 'NECK', 3 : 'TORSO', 4 : 'LEFT_SHOULDER', 5 : 'LEFT_ELBOW', 6 : 'RIGHT_SHOULDER', 7 : 'RIGHT_ELBOW', 8 : 'LEFT_HIP', 9 : 'LEFT_KNEE',
				    10 : 'RIGHT_HIP', 11 : 'RIGHT_KNEE', 12 : 'LEFT_HAND', 13 : 'RIGHT_HAND', 14 : 'LEFT_FOOT', 15 : 'RIGHT_FOOT'}


	def annotate(self):
		for i, txt in enumerate(self.dict):
			print txt
			self.ax.text(self.p_x[i],self.p_y[i],self.p_z[i],self.dict[txt])


	def enableInteractivePlot(self):
		plt.ion()
	
	
	def getPointsAndNormalize(self,line):
		self.getPoints(line)
		self.normalize()
		self.shiftOrigin()


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


	def shiftOrigin(self):
		self.hip_center = []
		self.hip_center.append((self.p_x[7]+self.p_x[9])/2)
		self.hip_center.append((self.p_y[7]+self.p_y[9])/2)
		self.hip_center.append((self.p_z[7]+self.p_z[9])/2)
		self.p_x-=self.hip_center[0]
		self.p_y-=self.hip_center[1]
		self.p_z-=self.hip_center[2]


	def setLimits(self):
		self.ax.set_xlim(-1,1)
		self.ax.set_ylim(-1,1)
		self.ax.set_zlim(-1,1)


	def setLabels(self):
		self.ax.set_xlabel('xlabel')
		self.ax.set_ylabel('ylabel')
		self.ax.set_zlabel('zlabel')


	def plot(self,j1,j2):
		self.ax.plot([self.p_x[j1],self.p_x[j2]],[self.p_y[j1],self.p_y[j2]],[self.p_z[j1],self.p_z[j2]],'#ff3300',linewidth=4)


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

	def getCoordinates(self,x,y,z):
		Z = []
		count_x = np.array([1,0,-1,0],dtype='float')/self.num_divisions
		count_y = np.array([0,1,0,-1],dtype='float')/self.num_divisions
		x_coord = float(x)/self.num_divisions
		y_coord = float(y)/self.num_divisions
		for k in [z,z+1]:
			for i in range(4):
				next_coord = [-0.5+x_coord,-0.5+y_coord,-0.5+float(k)/self.num_divisions]
				Z.append(next_coord)
				x_coord+=count_x[i]
				y_coord+=count_y[i]
		return np.array(Z)		


	def getSides(self,Z):
		verts = [[Z[0],Z[1],Z[2],Z[3]],
				 [Z[4],Z[5],Z[6],Z[7]], 
				 [Z[0],Z[1],Z[5],Z[4]], 
				 [Z[2],Z[3],Z[7],Z[6]], 
				 [Z[1],Z[2],Z[6],Z[5]],
				 [Z[4],Z[7],Z[3],Z[0]]]
		return verts
				 

	def drawCubes(self):
		for i in range(int(self.num_divisions)):
			for j in range(int(self.num_divisions)):
				for k in range(int(self.num_divisions)):
					Z = self.getCoordinates(i,j,k)
					verts = self.getSides(Z)
					# plot sides
					from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

					collection = Poly3DCollection(verts, 
					 facecolors=None, linewidths=1, edgecolors='g', alpha=.2)
					face_color = [1, 1, 1] # alternative: matplotlib.colors.rgb2hex([0.5, 0.5, 1])
					collection.set_facecolor(face_color)
					collection.set_edgecolor('k')
					self.ax.add_collection3d(collection)

	def rotate_points(self):
		#Only considering x-y coordinates
		right_hip = [self.p_x[9],self.p_y[9]]
		left_hip = [self.p_x[7],self.p_y[7]]
		orientation = np.array([left_hip[0]-right_hip[0],left_hip[1]-right_hip[1]],dtype='float32')
		theta = math.acos(orientation[0]/np.linalg.norm(orientation))

		for i in range(len(self.p_x)):
			self.p_x[i],self.p_y[i] = self.rotate((0,0),(self.p_x[i],self.p_y[i]),-theta)


	def rotate(self,origin, point, angle):
	    """
	    Rotate a point counterclockwise by a given angle around a given origin.

	    The angle should be given in radians.
	    """
	    ox, oy = origin
	    px, py = point

	    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	    return qx, qy


	def show(self):
		for line in self.data_file:
			if line=="END":
				break
			self.setLimits()
			self.setLabels()
			self.getPointsAndNormalize(line)
			# self.rotate_points()
			self.drawCubes()
			self.plotJoints()
			self.annotate()
			self.ax.plot(self.p_x,self.p_y,self.p_z,'o')
			r = [-1,1]
			X, Y = np.meshgrid(r, r)
			# plot vertices
			#self.ax.scatter3D(self.points[:, 0], self.points[:, 1], self.points[:, 2])
			plt.draw()
			plt.pause(0.02)
			self.ax.cla()

#sample object
# visual = visualization('/media/arya/54E4C473E4C458BE/Action_dataset/data1/0512171649.txt',3,False)
# visual.show()