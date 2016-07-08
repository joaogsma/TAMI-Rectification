import numpy as np
from copy import deepcopy
from math import sqrt
from numpy.linalg import inv
from point import Point


# =============================================================================
# ============================= UTILITY FUNCTIONS =============================
# =============================================================================
# Returns an image, which is the application of the given transformation 'H' on
# the given image 'image'. The transformation is applied through the inverse of
# H, thus avoiding the presence of holes in the transformed image. Set invert=
# False if H is already the precomputed inverse.
def transform_image(H, image, invert=True):
    # Compute the number of rows and columns in the image
    (row_num, col_num, _) = image.shape
    # Initialize the transformed image
    image_ = np.zeros(image.shape, dtype=np.uint8)
    # Compute the inverse transformation
    if invert:
        H = inv(H)

    # Compute transformed image
    for col in range(0, col_num):
        for row in range(0, row_num):
            # Compute the point p in the original image which is mapped to the 
            # (row, col) pixel in the transformed image
            p = Point(col, row, 1)
            p.to_img_coord(col_num, row_num)
            p.transform(H)
            p.normalize()
            # Get the (row_px, col_px) pixel coordinates of p
            (row_px, col_px)  = p.get_pixel_coord(col_num, row_num)

            # If the pixel is part of the image, get it's color
            if ((0 < col_px < col_num) and (0 < row_px < row_num)):
                # Original image
                pixelValue = image[row_px][col_px]
                # Rectified image
                image_[row][col][0] = pixelValue[0] #R
                image_[row][col][1] = pixelValue[1] #G
                image_[row][col][2] = pixelValue[2] #B

    return image_

def is_line_pair_list(list_, ):
    for pair in perpendicular_line_pairs:
        if (type(pair) != tuple or type(pair[0]) != Point or 
                type(pair[1]) != Point):
            return False

    return True

def solve_system(perpendicular_line_pairs):
    pair1 = perpendicular_line_pairs[0]
    pair2 = perpendicular_line_pairs[1]
    
    (l, m) = pair1
    (r, s) = pair2
    
    (l1, l2, l3) = [l.x, l.y, l.z]
    (m1, m2, m3) = [m.x, m.y, m.z]
    (r1, r2, r3) = [r.x, r.y, r.z]
    (s1, s2, s3) = [s.x, s.y, s.z]

    b = r1*s1*l2*m2 - r2*s2*l1*m1
    b /= l1*m1*(r1*s2 + r2*s1) - r1*s1*(l1*m2 + l2*m1)

    c = -(l1*m2 + l2*m1)*b - l2*m2
    c /= l1*m1

    return b, c

# =============================================================================



# =============================================================================
# ========================= STRATIFIED RECTIFICATION ==========================
# =============================================================================
def remove_projective_distortion(image, parallel_line_pairs):
    # Check input format
    if len(parallel_lines) != 2 or not is_line_pair_list(parallel_line_pairs):
        raise Exception("Incorrect set of parallel line pairs")

    # Compute the vanishing points
    vanishing_points = list()
    for pair in parallel_lines:
        vp = pair[0].cross(pair[1])
        vanishing_points.append(vp)

    # Compute the infinity line
    infinity_line = vanishing_points[0].cross(vanishing_points[1])
    # Normalize the infinity line. This is done to avoid numerical errors
    infinity_line.normalize()

    # Compute the projective transformation H_p, which returns the infinity
    # line to its correct position, removing projective distortion on the image
    a = infinity_line.x
    b = infinity_line.y
    c = infinity_line.z
    H_p = np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])

    # Compute the transformed image and return it
    return transform_image(H_p, image) 


def remove_affine_distortion(image, perpendicular_line_pairs):
    # Check input format
    if (len(perpendicular_line_pairs) != 2 or 
            not is_line_pair_list(perpendicular_line_pairs)):
        raise Exception("Incorrect set of perpendicular line pairs")

    # Compute the elements which specify the conic dual to the circular points 
    (b, c) = solve_system(perpendicular_line_pairs)
    a = sqrt(c - b*b)

    # Compute the matrix H_a_inv. This is the transformation that caused the 
    # affine distortion of the matrix, as opposed to its inverse (H_a, not 
    # computed), which which would remove the distortion.
    H_a_inv = np.array([[a, b, 1], [0, 1, 0], [0, 0, 1]])

    # Compute the transformed image and return it. There is no need to compute
    # H_a, since transform_image would utilize its inverse to avoid having
    # holes in the rectified image
    return transform_image(H_a_inv, image, invert=False)


def stratified_metric_rect(image, parallel_line_pairs, 
                           perpendicular_line_pairs):
    
    # Remove projective distortions on the image (affine rectification)
    image = remove_projective_distortion(image, parallel_line_pairs)

    # Remove affine distortions on the image (metric rectification)
    image = remove_affine_distortion(image, perpendicular_line_pairs)

    return image
    
# =============================================================================
