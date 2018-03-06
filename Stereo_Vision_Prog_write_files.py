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

#====================================================

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0,65):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)
    ChessImaR= cv2.imread('images/chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('images/chessboard-L'+t+'.png',0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9,6),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

#   Right Side
_, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints,imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)

#   Left Side
_, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints,imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)

##===========================================================
filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/", "{}.npy".format("distL"))
filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))

# Write
np.save(filenameL,imgpointsL)
np.save(filenameR,imgpointsR)
np.save(filename_op,objpoints)
np.save(filename_mtR,mtxR)
np.save(filename_dR,distR)
np.save(filename_mtL,mtxL)
np.save(filename_dL,distL)
np.save(filename_chR,ChessImaR)

print('Cameras Ready to use')