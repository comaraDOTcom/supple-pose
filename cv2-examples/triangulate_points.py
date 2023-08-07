import numpy as np
import cv2 as cv

image_3910_camera_matrix = np.loadtxt('/Users/conoromara/code/supple-pose/cv2-examples/iphone-xr/chessboard/IMG_3910_camera_matrix.txt')
image_3914_camera_matrix = np.loadtxt('/Users/conoromara/code/supple-pose/cv2-examples/iphone-xr/chessboard/IMG_3914_camera_matrix.txt')

# projected points in IMG_3910
img_3910_projected_imgpoints = np.loadtxt('/Users/conoromara/code/supple-pose/cv2-examples/calibrate/IMG_3910.txt')
# projected points in IMG_3914
img_3914_projected_imgpoints = np.loadtxt('/Users/conoromara/code/supple-pose/cv2-examples/calibrate/IMG_3914.txt')

# reconstruct 3d points in homogenous coordinates.
output = cv.triangulatePoints(
    projMatr1=image_3910_camera_matrix,
    projMatr2=image_3914_camera_matrix,
    projPoints1=img_3910_projected_imgpoints,
    projPoints2=img_3914_projected_imgpoints,
)

np.savetxt('/Users/conoromara/code/supple-pose/cv2-examples/calibrate/triangulate_points_IMG3910_IMG3914.txt', output)
print(f"Output is: {output}")