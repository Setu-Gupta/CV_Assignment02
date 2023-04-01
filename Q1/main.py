import cv2
import numpy as np
import os
import glob
import csv

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
images_with_corners = []    # An array to store the images in which OpenCV was able to detect the corners
images = []                 # An array to store the images which are later used for removing the distortion

# Iterate over all the images and store the corner coordinates for them
for fname in image_paths:
    
    # Read the image
    img = cv2.imread(fname)
    
    # Convert to greyscale
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
        images_with_corners.append(fname)
        images.append(img)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        
    # Save the image
    corner_path = fname.replace("images", "corners")
    cv2.imwrite(corner_path, img)

# Perform camera calibration
reproj_error, int_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Extract the intrinsic parameters out of the intrinsic matrix
fx = int_mtx[0][0]
fy = int_mtx[1][1]
skew = int_mtx[0][1]
cx = int_mtx[0][2]
cy = int_mtx[1][2]

print("Estimated fx and fy focal lengths:", fx, fy)
print("Estimated skew (OpenCV does NOT estimate the skew. It assumes it to be zero):", skew)
print("Estimated optical center coordinates:", cx, cy)
print("Estimated re-projection error:", reproj_error)
print("============================================")

print("The extrinsic parameters for all the images:")

# Dump the extrinsic parameters to a CSV file
with open("extrinsics.csv", 'w', newline='') as csvfile:
    fieldnames = ['Image name', 'Rotation angle', 'Rotation axis', 'Translation vector']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(rvecs)):
        image_name = images_with_corners[i]
        translation = tvecs[i]
        rotation = rvecs[i]
        rotation_angle = np.linalg.norm(rotation)
        rotation_axis = rotation / rotation_angle
        print(f"{image_name}: Rotation angle->{rotation_angle} Rotation axis->{rotation_axis.reshape(-1)} Translation->{translation.reshape(-1)}")

        row = {'Image name': image_name,
               'Rotation angle': rotation_angle,
               'Rotation axis': rotation_axis.reshape(-1),
               'Translation vector': translation.reshape(-1),
              }
        writer.writerow(row)
print("============================================")

print("Estimated Distortion coefficients:", dist)

for img, fname in zip(images, images_with_corners):
    h, w = img.shape[:2]
    
    # Refine the camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(int_mtx, dist, (w,h), 1, (w,h))
    
    # Undistort and save the image
    dst = cv2.undistort(img, int_mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    output_image_name = fname.replace("images", "undistorted")
    cv2.imwrite(output_image_name, dst)
print("============================================")
