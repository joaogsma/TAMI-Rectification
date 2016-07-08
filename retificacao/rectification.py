import numpy as np
from copy import deepcopy
from numpy.linalg import inv
from point import Point

def affine_rectification(image, infinity_line):
    # Normalize the infinity line. This is done to avoid numerical problems
    infinity_line = deepcopy(infinity_line)
    infinity_line.normalize()
    
    # Compute the projective transformation H_p, which returns the infinity
    # line to its correct position, removing projective distortion on the image
    a = infinity_line.x
    b = infinity_line.b
    c = infinity_line.c
    H_p = np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])

    # Compute the transformed image and return it
    return transform_image(H_p, image)


# Returns an image, which is the application of the given transformation 'H' on
# the given image 'image'. The transformation is applied through the inverse of
# H, thus avoiding the presence of holes in the transformed image
def transform_image(H, image):
    # Compute the number of rows and columns in the image
    (row_num, col_num, _) = np.shape(f)
    # Initialize the transformed image
    f_ret = np.zeros(f.shape, dtype=np.uint8)
    # Compute the inverse transformation
    H_inv = inv(H)

    # Compute transformed image
    for col in range(0, col_num):
        for row in range(0, row_num):
            # Compute the point p in the original image which is mapped to the 
            # (row, col) pixel in the transformed image
            p = Point(col, row, 1)
            p.to_img_coord(col_num, row_num)
            p.transform(H_inv)
            p.normalize()
            # Get the (row_px, col_px) pixel coordinates of p
            (row_px, col_px)  = p.get_pixel_coord(col_num, row_num)

            # If the pixel is part of the image, get it's color
            if ((0 < col_px < col_num) and (0 < row_px < row_num)):
                # Original image
                pixelValue = f[row_px][col_px]
                # Rectified image
                f_ret[row][col][0] = pixelValue[0] #R
                f_ret[row][col][1] = pixelValue[1] #G
                f_ret[row][col][2] = pixelValue[2] #B

    return f_ret