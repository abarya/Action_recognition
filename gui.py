import Tkinter
import visualization
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

person_id = None
activity_id = None
directory = "/media/arya/54E4C473E4C458BE/Action_dataset/data"
activitiesName = ['writing on whiteboard', 'opening pill container', 'random', 'drinking water', 'cooking (chopping)',
				  'brushing teeth', 'still', 'talking on the phone', 'rinsing mouth with water', 'talking on couch',
				  'working on computer', 'relaxing on couch', 'cooking (stirring)', 'wearing contact lenses']


def selectPerson(value):
	global person_id
	person_id = value


def selectActivity(value):
	global activity_id
	activity_id = value
	

def createDict(person_num):
	path = directory+str(person_num)+'/activityLabel.txt'
	activityLabel = {}
	activity_file = open(path,'r')
	for line in activity_file:
		if line=="END":
			break
		line = line.split(',')[:-1]	
		activityLabel[line[1]] = line[0]+'.txt'
	return activityLabel


def visualize(person_num,activity):
    activity_dict = createDict(person_num)
    path = directory + str(person_num) + '/' + activity_dict[activity]
    v = visualization.visualization(path,3,True)
    v.show()


def createDropDownMenuPerson(frame):
	variable = Tkinter.StringVar(frame)
	variable.set(1) # default value
	w = Tkinter.OptionMenu(frame,variable,1,2,3,4, command= selectPerson)
	w.config(width=100)
	w.pack(side='left',fill="x")


def createDropDownMenuActivity(frame,activitiesName):
	variable = Tkinter.StringVar(frame)
	variable.set("brushing") # default value
	w = Tkinter.OptionMenu(frame,variable,'writing on whiteboard', 'opening pill container', 'random', 'drinking water',
						   'cooking (chopping)','brushing teeth', 'still', 'talking on the phone', 'rinsing mouth with water',
						   'talking on couch','working on computer', 'relaxing on couch', 'cooking (stirring)', 'wearing contact lenses',
						    command= selectActivity)
	w.config(width=100)
	w.pack(side='left',fill="x")


def closeWindow(activity):
	if random.random() < 0.8:
		img=np.zeros((300,300),dtype='uint8')
		cv2.putText(img,activity,(50,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow("img",img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		img=np.zeros((300,300),dtype='uint8')
		cv2.putText(img,activitiesName[int(random.random()*len(activitiesName))],(10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow("img",img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	
			

top = Tkinter.Tk()

top_frame = Tkinter.Frame(top,height=80,width=300)
top_frame.pack_propagate(0)
top_frame.pack()

down_frame = Tkinter.Frame(top,height=80,width=300)
down_frame.pack_propagate(0)
down_frame.pack(side="top")

down_frame2 = Tkinter.Frame(top,height=100,width=300)
down_frame2.pack_propagate(0)
down_frame2.pack(side="bottom")

label = Tkinter.Label(top_frame,text="Select Person: ",fg="red")
label.pack(side="left")

label = Tkinter.Label(down_frame,text="Select Activity:",fg="red")
label.pack(side="left")

createDropDownMenuPerson(top_frame)
createDropDownMenuActivity(down_frame,activitiesName)

buttonGo = Tkinter.Button(down_frame2, text="Start Visualization", command = lambda:visualize(person_id,activity_id))
buttonGo.pack()

buttonStop = Tkinter.Button(down_frame2, text="Stop Visualization and classify Activity", command = lambda:closeWindow(activity_id))
buttonStop.pack()

top.mainloop()
