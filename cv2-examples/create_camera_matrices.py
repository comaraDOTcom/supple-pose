import numpy as np
import cv2 as cv
import os

cwd = os.getcwd()

# First image
images = [
    "cv2-examples/iphone-xr/chessboard/IMG_3910.jpeg",
    "cv2-examples/iphone-xr/chessboard/IMG_3911.jpeg",
    "cv2-examples/iphone-xr/chessboard/IMG_3914.jpeg",
    "cv2-examples/iphone-xr/chessboard/IMG_3917.jpeg",
]


# instrinsic matrix
K = np.array(
    [
        [3.09199949e03, 0.00000000e00, 1.52308981e03],
        [0.00000000e00, 3.10166355e03, 1.86807015e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# distortion coefficients
dist = np.array(
    [[2.18683640e-01, 5.33372619e-01, -2.05956490e-02, -3.38301720e-03, -6.32434590e00]]
)

# rotation vectors
rvecs = np.array(
    [
        [-0.11893077, 0.09625741, 2.02047069],
        [-0.21840696, 0.02228696, 1.48292746],
        [0.13866245, -0.02150365, 1.45823303],
        [-0.24697967, 0.2530784, 1.53941822],
    ]
)

# translation vectors
tvecs = np.array(
    [
        [2.51905268, -0.54756773, 15.52981619],
        [0.25183613, -2.67597754, 16.24146717],
        [2.21810863, -2.80810259, 15.0016107],
        [1.73947752, -0.29315962, 17.66658854],
    ]
)


def compute_camera_matrix(K, translation_vector, rotation_vector):
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    # Extrinsix matrix (R|T)
    RT = np.concatenate((rotation_matrix, translation_vector.reshape(3, -1)), axis=1)
    # camera matrix is the matrix multiplication of intrinsic matrix and Extrinsic matrix
    C = np.matmul(K, RT)
    return C


def write_camera_matrices(camera_matrix_dict, write_location):
    for key in camera_matrix_dict.keys():
        key_path = key.split(".")[0]
        print("Conor")
        np.savetxt(
            write_location + "/" + key_path + "_camera_matrix.txt",
            camera_matrix_dict[key],
        )
        print(f"Written to {write_location + '/' + key}")
    return True


image_camera_matrix_dict = {
    image: compute_camera_matrix(K, tvecs[index], rvecs[index])
    for index, image in enumerate(images)
}

# save camera matrices for each image
write_camera_matrices(image_camera_matrix_dict, write_location=cwd)
