#      ▄▀▄     ▄▀▄
#     ▄█░░▀▀▀▀▀░░█▄
# ▄▄  █░░░░░░░░░░░█  ▄▄
#█▄▄█ █░░▀░░┬░░▀░░█ █▄▄█

#######################################
##### Authors:                    #####
##### Stephane Vujasinovic        #####
##### Frederic Uhrweiller         ##### 
#####                             #####
##### Creation: 2017              #####
##### Optimization: David Castillo#####
##### Rv: FEB:2018                #####
#######################################


#***********************
#**** Main Programm ****
#***********************

# Package importation
import time
import numpy as np
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Pool

# =========================sub Process===========================

def doWork(st): #j=1 es izquierdo , j=2 es derecho
    grayL = st[0] 
    grayR = st[1]
    j = st[2]

    # Create StereoSGBM and prepare all parameters
    window_size = 5
    min_disp = 2
    num_disp = 130-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        preFilterCap = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)

    # Used for the filtered image
    if j == 1 :
        disp= stereo.compute(grayL,grayR)
    
    if j == 2 :
        stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
        disp= stereoR.compute(grayR,grayL)

    return disp

#====================================================

cv2.useOptimized()
wb=Workbook()
ws=wb.active  

# write into the excell worksheet

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        """
				p p p
				p p p
				p p p
        """
        average=0
        for u in range (-1,2):     # (-1 0 1)
            for v in range (-1,2): # (-1 0 1)
                average += disp[y+u,x+v]
        average=average/9
        #Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        #Distance= np.around(Distance*0.01,decimals=2)
        #print('Distance: '+ str(Distance)+' m')
        print('Average: '+ str(average))
        counterdist = int(input("ingresa distancia (cm): "))
        ws.append([counterdist, average])

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

##===========================================================
filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/", "{}.npy".format("distL"))
filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))

# Read
imgpointsR = np.load(filenameR)
imgpointsL = np.load(filenameL)
objpoints = np.load(filename_op)
mtxR = np.load(filename_mtR)
distR = np.load(filename_dR)
mtxL = np.load(filename_mtL)
distL = np.load(filename_dL)
ChessImaR = np.load(filename_chR)

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)


#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    preFilterCap = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)


# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000#80000
sigma = 1.8 #1.8
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
CamR= cv2.VideoCapture(1)   # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL= cv2.VideoCapture(2)

while True:
    with Pool(processes=2) as pool :
        #mark the start time
        startTime = time.time()
        # Start Reading Camera images
        retR, frameR= CamR.read()
        retL, frameL= CamL.read()
        
        # Rectify the images on rotation and alignement
        # Rectify the image using the calibration parameters founds during the initialisation
        Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  
        Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Convert from color(BGR) to gray
        #grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        #grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

        grayR= Right_nice
        grayL= Left_nice
        #=======================================================================================
        
        # Compute the 2 images for the Depth_image
        # Run the pool in multiprocessing
        st1 = (grayL,grayR,1 )
        st2 = (grayL,grayR,2 )

        # Computo para el stereo
        disp , dispR = pool.map(doWork, (st1,st2))
        
        dispL= disp

        #=======================================================================================
    
        dispL= np.int16(dispL)
        dispR= np.int16(dispR)
        
        # Using the WLS filter
        filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)

        # Change the Color of the Picture into an Ocean Color_Map
        filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

        cv2.imshow('Filtered Color Depth',filt_Color)

        # Draw Red lines
        for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            Left_nice[line*20,:]= (0,0,255)
            Right_nice[line*20,:]= (0,0,255)
      
        cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
        
        # Mouse click
        cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
        
        #mark the end time
        endTime = time.time()    
        
        pool.terminate()
        pool.join() 
        #calculate the total time it took to complete the work
        workTime =  endTime - startTime
        
        #print results
    print ("The job took " + str(workTime) + " sconds to complete")

        # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

    
# Save excel
wb.save("readvsdist.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
