# -*- coding: utf-8 -*-
import cv2
import glob #module that finds all the pathnames matching
from matplotlib import pyplot as plt # importing library for plotting 


'''                                 SIFT PERFORMING 

Detecting some relevant points by using SIFT detector and placing a SIFT descriptor 
on each of those point, in order to obtain a bag of SIFT descriptors (i.e. intensity 
gradient histograms) to represent each image (for a total of 8 images).

'''

#load images 
collection = []
for im in glob.glob('selected_images/*.bmp'): 
    img = cv2.imread(im)
    collection.append(img)

#convert them into gray scale 
image_list=[]
for i in collection:
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    image_list.append(gray)

#create feature extraction object
sift = cv2.xfeatures2d.SIFT_create() #load algorythm
# optional: passing parameters like numbers of features and threshold
# install opencv-contrib-python to make the attribute works

# Detect: find relevant keypoints
kp = [] #list of keypoints
for img in image_list:
    k = sift.detect(img, None) # passing no mask 
    kp.append(k)
    
des = [] #list of descriptors
for n in range(len(image_list)):
    des.append(n)
    #Compute a descriptor for each keypoint detected
    kp[n], des[n] = sift.compute(image_list[n], kp[n])
    #output: keypoint and corresponding descriptor --> MATRIX (rows=kp, col=128)
    
    # Draw keypoints detected on the image
    image_list[n] = cv2.drawKeypoints(image_list[n], kp[n], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #the flag draw a circle with size of keypoint and show its orientation

# for each image, we obtained a number of descriptors equal to the number of keypoints detected
    


'''                             HISTOGRAM COMPARISON
'''

images = []
for im in glob.glob('result/*.bmp'): 
    image = cv2.imread(im)
    images.append(image)


''' 
Performing comparison between 
- first two descriptors, one from a face image and one from a car image (respectively, img6 and img7) 
- other two descriptors, one from a horse image and one from a cow image (respectively, img1 and img5)

The fist comparison will be simply visual, while the second will be quantitative,
using three different metrics of comparison.

'''
  

'''                             1. VISUAL COMPARISON

                    Plotting the two SIFT descriptors closeby as barplots 
''' 
    

#creating multiple plots
fig = plt.figure(figsize=(20, 20)) # create a figure object    

ax1 = plt.subplot(321) # face
ax2 = plt.subplot(322) # horse
ax3 = plt.subplot(323) # car
ax4 = plt.subplot(324) # cow
ax5 = plt.subplot(325) # face-car comparison
ax6 = plt.subplot(326) # horse-cow comparison

#Vertically correlated
ax1.set_title('A face descriptor ')
ax1.set_xlabel('Descriptors')
ax1.set_ylabel('Gradient frequency')

ax2.set_title('A horse descriptor ')
ax2.set_xlabel('Descriptors')
ax2.set_ylabel('Gradient frequency')

ax3.set_title('A car descriptor ')
ax3.set_xlabel('Descriptors')
ax3.set_ylabel('Gradient frequency')

ax4.set_title('A cow descriptor ')
ax4.set_xlabel('Descriptors')
ax4.set_ylabel('Gradient frequency')

ax5.set_title('Face - Car histograms comparison ')
ax5.set_xlabel('Gradient frequency')
ax5.set_ylabel('Power')

ax6.set_title('Horse - Cow histograms comparison ')
ax6.set_xlabel('Gradient frequency')
ax6.set_ylabel('Power')

''' plotting each histogram
for n in range(len(kp)): # sovrapposizione istogrammi di ogni keypoint
    face_hist = des[5][n] # face descriptor
    car_hist = des[6][n] # car descriptor
    horse_hist = des[0][n] # horse descriptor
    cow_hist = des[4][n] # cow descriptor
    ax1.plot(face_hist, 'g-')
    ax3.plot(car_hist, 'b-')
    ax2.plot(horse_hist, 'r-')
    ax4.plot(cow_hist, 'y-')
'''

#chose any descriptor (keypoints) from each histogram
face_hist = des[5][76] # face histogram (i.e. keypoint at position 42 of the image at position 5 - 6th image - )
car_hist = des[6][27] # car histogram
horse_hist = des[0][34] # horse histogram
cow_hist = des[4][87] # cow histogram

ax1.plot(face_hist, 'g-')
ax3.plot(car_hist, 'b-')
ax2.plot(horse_hist, 'r-')
ax4.plot(cow_hist, 'y-')

#comparison (without overlapping bars)
ax5.hist((face_hist, car_hist), bins=10, color=('b', 'g'), alpha=0.5) 
ax6.hist((horse_hist, cow_hist), bins=10, color=('r', 'y'), alpha=0.5) 

#if you need to save plot as an image in the folder
#plt.savefig("plot.png")


'''
                            2. OTHERS HISTOGRAMS COMPARISON METRICS
'''

comp1 = []
comp2 = []
comp3 = []

for method in range(3): # 3 metriche di confronto
    comparison1 = cv2.compareHist(horse_hist, horse_hist, method) 
    comp1.append(comparison1)
    comparison2 = cv2.compareHist(horse_hist, car_hist, method) 
    comp2.append(comparison2)
    comparison3 = cv2.compareHist(horse_hist, cow_hist, method) 
    comp3.append(comparison3)
  
    
methods = ["Correlation ", "Chi-Square  ", "Intersection"]

# !!! HISTOGRAMS HAVE TO BE NORMALIZED AT 1 TO PERFORM INTERSECTION CORRECTLY 

print('Method: ' + '\t\t' + 'Horse - Horse' + '\t\t' + 'Horse - Car ' + '\t\t' + 'Horse - Cow ' + '\n')

for c1, c2, c3, m in zip(comp1, comp2, comp3, methods):
    print(m + '\t\t' + str(c1) + '\t\t' + str(c2) + '\t\t' + str(c3) + '\n\n')


print('Number of keypoint detected --> \t face image: ' + '\t' + str(len(kp[5])) + '\n')  
print('Number of keypoint detected --> \t car image: ' + '\t' + str(len(kp[6])) + '\n')  
print('Number of keypoint detected --> \t horse image: ' + '\t' + str(len(kp[0])) + '\n')  
print('Number of keypoint detected --> \t cow image: ' + '\t' + str(len(kp[4])) + '\n')  