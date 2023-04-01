import cv2
import numpy as np
import os
import glob

# Specify the pattern
CHECKERBOARD = (9, 7) # The pattern has 9 corners per column and 7 corners per row

# Store the world 3D coordinates of corners
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Get the path to all the images
image_paths = glob.glob('./images/*.jpg')

# Create arrays to store the 3D world coordinate points and the camera coordinate points
objpoints = []
imgpoints = []

# Iterate over all the images and store the corner coordinates for them
for fname in image_paths:
    
    # Read the image
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to extract out the corners in the image
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # Check if the code was able to find the corners
    if ret == True:
        
        # Refine the estimate of corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Run for maximum 30 iterations and stop if the changes is less than or equal to 0.001
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
        
        # Store the refined corners and the world coordinates
        imgpoints.append(corners_refined)
        objpoints.append(objp)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        
    # Save the image
    corner_path = fname.replace("images", "corners")
    cv2.imwrite(corner_path, img)

# h,w = img.shape[:2]
# 
# """
# Performing camera calibration by
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the
# detected corners (imgpoints)
# """
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# 
# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)
