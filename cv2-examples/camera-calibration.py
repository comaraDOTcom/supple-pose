import numpy as np
import cv2 as cv
import glob
import os
import re

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
#images = glob.glob("cv2-examples/iphone-xr/chessboard/*.jpeg")
images = [
    "cv2-examples/iphone-xr/chessboard/IMG_3910.jpeg",
    "cv2-examples/iphone-xr/chessboard/IMG_3911.jpeg",
    "cv2-examples/iphone-xr/chessboard/IMG_3914.jpeg",
    "cv2-examples/iphone-xr/chessboard/IMG_3917.jpeg",
]

for fname in images:
    print(f"On image {fname}")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"True for image: {fname}")
        objpoints.append(objp)
        # refines corners location.
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Use regular expression pattern to match "IMG_" followed by any number
        pattern = r"IMG_(\d+)"
        # Search for the pattern in the path
        match = re.search(pattern, fname)
        if match:
            extracted_string = match.group(0)
            print(extracted_string)  # Output: IMG_3910
        else:
            print("No match found.")
        np.savetxt( os.getcwd() + "/cv2-examples/calibrate/" + extracted_string + ".txt",
            corners2.reshape(42,2),
        )
        imgpoints.append(corners2)
        # Draw and display the corners
        if fname == images[-1]:
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(10)
cv.destroyAllWindows()
print(f"Number of imgpoint entries {len(imgpoints)}")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"Imgpoints are: {imgpoints}")


cwd = os.getcwd()

np.savetxt(
    cwd + "/cv2-examples//calibrate/camera_matrix.txt",
    mtx,
)
np.savetxt(
    cwd + "/cv2-examples//calibrate/distortion_matrix.txt",
    dist,
)

np.save(cwd + "/cv2-examples//calibrate/rotation_vector.npy", rvecs)
np.save(cwd + "/cv2-examples//calibrate/translation_vector.npy", tvecs,)

print("Camera matrix")
print("Return: \n")
print(ret)
print("Camera matrix: \n")
print(mtx)
print("distortion coefficients: \n")
print(dist)
print("Rotation vectors: \n")
print(rvecs)
print("Translation vectors: \n")
print(tvecs)
