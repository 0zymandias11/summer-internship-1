# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:48:12 2019

@author: priyanakar
"""

import cv2
import time
import numpy as np
import calculate_angle
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import interpolate


MODE = "COCO"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
import cv2
import time
import numpy as np

MODE = "COCO"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,18],[18,9],[9,10],[1,18],[18,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_133336.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 348
inHeight = 348
threshold = 0.0000


input_source = "vid10.mp4"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
j=0
knee_angles=[]
waist_angles=[]
angle=0 
angle2=0

knee_angles_b=[]
waist_angles_b=[]
l_ankle_position=[]
r_ankle_position=[]
l_knee_position=[]
r_knee_position=[]
while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            if i==18 or i==12 or i==13 or i==9 or i==10 or i==1:
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
            if i==8 or i==11:
                print('YO ',i)
        else :
            points.append(None)
        
            
    cv2.imwrite('Output-Keypoints'+str(j)+'.jpg', frameCopy)
       
    points.append((int((points[8][0]+points[11][0])/2),int((points[8][1]+points[11][1])/2)))
    cv2.circle(frameCopy,points[18],8,(0,255,255),thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frameCopy, "18",(int(points[18][0]),int(points[18][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
           
    if(points[12][0]<=points[18][0]):
      angle,angle2=  calculate_angle.back_angle(points[18][0],points[18][1],points[12][0],points[12][1],points[13][0],points[13][1])
    else:
      angle,angle2=  calculate_angle.front_angle(points[18][0],points[18][1],points[12][0],points[12][1],points[13][0],points[13][1])
    
    #back_knee angle
    knee_angles.append(angle)
    waist_angles.append(angle2)
   
    l_ankle_position.append(points[10])
    r_ankle_position.append(points[13])
    l_knee_position.append(points[9])
    r_knee_position.append(points[12])  
    #front_knee angle
    if(points[9][0]<=points[18][0]):
        angle,angle2=  calculate_angle.back_angle(points[18][0],points[18][1],points[9][0],points[9][1],points[10][0],points[10][1])
    else :
        angle,angle2=  calculate_angle.front_angle(points[18][0],points[18][1],points[9][0],points[9][1],points[10][0],points[10][1])

    knee_angles_b.append(angle)
    waist_angles_b.append(angle2)        
    

    
  
    ### plot
    
    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
    print(j)
    j+=1

    vid_writer.write(frame)
    if j==100 :
        break

vid_writer.release()
cv2.destroyAllWindows()




###############removing erroneous minimas#######################
from statistics import median

med=median(knee_angles)
medb=median(knee_angles_b)

for i in range(j):
    knee_angles[i]=med if(knee_angles[i] <110) else knee_angles[i]
    knee_angles_b[i]=medb if(knee_angles_b[i] <110) else knee_angles_b[i]

################################################################
knee_angle_rad=[]
waist_angle_rad=[]
pi=22/7
for i in range (0,len(knee_angles)):
    knee_angle_rad.append(knee_angles[i]*(pi/180))
    waist_angle_rad.append(waist_angles[i]*(pi/180))
    
    
########################Saving the Output###########################################################

####################################################################################################
####################################################################################################
###median approximation

###################################################################################################
        
        
plt.plot(np.linspace(4,36,num=32),knee_angles[4:36])
plt.plot(np.linspace(4,36,num=32),knee_angles_b[4:36])

l=np.linspace(4,36,num=32)
b=signal.medfilt(knee_angles[4:36],kernel_size=None)
b2=signal.medfilt(knee_angles_b[4:36],kernel_size=None)

f = interpolate.interp1d(l,b,kind='cubic')
f2=interpolate.interp1d(l,b2,kind='cubic')
xnew=np.linspace(4, 36, num=1000, endpoint=True)

##OLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

y=f(xnew)
bx=signal.medfilt(y,kernel_size=None)
y2=f2(xnew)
bx1=signal.medfilt(y2,kernel_size=None)

plt.plot(xnew,bx,label='back knee angles')
plt.plot(xnew,bx1,label='front knee angles')
plt.xlabel("Frames ---->")
plt.ylabel("Knee angles (in Degrees) ---->")
plt.title("knee angles")
plt.legend()
### Hip Angle #######################################
l=np.linspace(4,36,num=32)
b3=signal.medfilt(waist_angles[4:36],kernel_size=None)
f3 = interpolate.interp1d(l,b3,kind='cubic')
xnew3=np.linspace(4, 36, num=1000, endpoint=True)
y3=f3(xnew3)
plt.plot(xnew3,y3, label= "Hip angle" )
plt.xlabel("No of frames ---->");
plt.ylabel("Hip angles (in  Degrees) ---->")
plt.title("Hip angle plot")
plt.legend()



#####################################################
plt.plot(xnew,bx1,[xnew[386],xnew[741]],[bx1[386],bx1[741]],'ro')
bx2=signal.medfilt(y2,kernel_size=None)
plt.plot(xnew,bx2)
plt.show()
bx1=signal.medfilt(y,kernel_size=None)
######PHASE DIFF################################################################################
jkl=bx1[646:969]
jkl2=bx2[646:969]


##for 2nd Maxima and Minima
from scipy.signal import argrelextrema
maximas=argrelextrema(jkl, np.greater_equal)
minimas=argrelextrema(jkl2, np.less_equal)

plt.plot(xnew[646:969],bx1[646:969],label='back knee angles from 10th to 20th frame')

plt.plot(xnew[646:969],bx2[646:969],label='front knee angles from 10th to 20th frame')

plt.plot([xnew[646+maximas[0][3]]],bx1[646+maximas[0][3]],'ro',label='maxima')

plt.plot([xnew[646+minimas[0][2]]],bx2[646+minimas[0][2]],'bo',label='minima')
###plt.title('plot to calculate phase diff)
plt.ylabel('angles (in degrees) ---->')
plt.xlabel('frames ---->')
plt.legend()

peak1=maximas[0][3]
peak2=minimas[0][2]

diff=abs(peak1*0.063 - peak2*0.063)

print('diff ',diff)

lamda=63
path_difference=lamda/2 + diff

print('path_diff ',path_difference )

phase=((2*180)/lamda)*path_difference
print('phase ',phase)


################################################################################################
plt.plot(xnew, f(xnew),label='back knee angles')
plt.plot(xnew,f2(xnew),label='front knee angles')

plt.xlabel('Frames ---->')
plt.ylabel('Knee angles (front and back) in degrees ---->')
plt.title('Knee angles of front and back knee')
plt.legend()
plt.plot(xnew,y2)

#OLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA



plt.xlabel('Frames --->')
plt.ylabel('Left hip angle (in degrees) --->')
plt.title('left hip angle for 3 gait cycles')


### OTher Calculations
## step length

r_step_length=[]


for i in range(24,99,15):
    r_step_length.append(abs(r_ankle_position[i][0]-l_ankle_position[i][0]))
    print('i ', i,' r_ankle ',r_ankle_position[i][0],' l_ankle ',l_ankle_position[i][0])






##plt.plot(l[0:15],waist_angles[0:15],'o',xnew2,f(xnew2))

plt.acorr(waist_angles, maxlags=12)

##autoc=autocorr(waist_angle_rad)


##plt.plot(autoc,l)
##plt.scatter(l, knee_angles[0:0], label = "line 1")
##plt.xscale("linear",subsx=50)
plt.xlim([1, 63])
plt.show()


####Writing to CSV
import csv
myFile = open('subject3.csv', 'a')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(map(lambda x: [x], knee_angles_b[4:36]))
