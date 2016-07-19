import numpy as np
import scipy
from copy import deepcopy
from math import sqrt
from numpy.linalg import inv
from point import Point


# =============================================================================
# ================================== COMMON  ==================================
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

def is_line_pair_list(list_):
    for pair in list_:
        if (type(pair) != tuple or type(pair[0]) != Point or
                type(pair[1]) != Point):
            return False

    return True

# =============================================================================



# =============================================================================
# ========================= STRATIFIED RECTIFICATION ==========================
# =============================================================================

def solve_k_matrix_system(orthogonal_line_pairs):
    pair1 = orthogonal_line_pairs[0]
    pair2 = orthogonal_line_pairs[1]

    (l, m) = pair1
    (r, s) = pair2

    (l1, l2, l3) = [l.x, l.y, l.z]
    (m1, m2, m3) = [m.x, m.y, m.z]
    (r1, r2, r3) = [r.x, r.y, r.z]
    (s1, s2, s3) = [s.x, s.y, s.z]

    b = (r1*s1*l2*m2) - (r2*s2*l1*m1)
    b /= l1*m1*(r1*s2 + r2*s1) - r1*s1*(l1*m2 + l2*m1)

    c = -(l1*m2 + l2*m1)*b - l2*m2
    c /= l1*m1

    return b, c

def remove_projective_distortion(image, parallel_line_pairs):
    # Check input format
    if (len(parallel_line_pairs) != 2 or
            not is_line_pair_list(parallel_line_pairs)):
        raise Exception("Incorrect set of parallel line pairs")

    # Compute the vanishing points
    vanishing_points = list()
    for pair in parallel_line_pairs:
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
    return transform_image(H_p, image), H_p


def remove_affine_distortion(image, orthogonal_line_pairs):
    # Check input format
    if (len(orthogonal_line_pairs) != 2 or
            not is_line_pair_list(orthogonal_line_pairs)):
        raise Exception("Incorrect set of orthogonal line pairs")

    # Normalize all the lines
    for (l1, l2) in orthogonal_line_pairs:
        l1.normalize()
        l2.normalize()

    # Compute the elements which specify the conic dual to the circular points
    (b, c) = solve_k_matrix_system(orthogonal_line_pairs)
    #print(c)
    #print(b)
    a = sqrt(c - b*b)

    # Compute the matrix H_a_inv. This is the transformation that caused the
    # affine distortion of the matrix, as opposed to its inverse (H_a, not
    # computed), which which would remove the distortion.
    H_a_inv = np.array([[a, b, 1], [0, 1, 0], [0, 0, 1]])

    # Compute the transformed image and return it. There is no need to compute
    # H_a, since transform_image would utilize its inverse to avoid having
    # holes in the rectified image
    return transform_image(H_a_inv, image, invert=False), inv(H_a_inv)


def stratified_metric_rect(image, parallel_line_pairs,
                           orthogonal_line_pairs):
    # Remove projective distortions on the image (affine rectification)
    (image, H_p) = remove_projective_distortion(image, parallel_line_pairs)

    # Correct coordinates from the orthogonal lines
    orthogonal_line_pairs = deepcopy(orthogonal_line_pairs)
    H_p_line = inv(H_p).transpose()
    for (l1, l2) in orthogonal_line_pairs:
        l1.transform(H_p_line)
        l2.transform(H_p_line)

    # Remove affine distortions on the image (metric rectification)
    (image, H_a) = remove_affine_distortion(image, orthogonal_line_pairs)

    # TODO: Maybe return the complete transformation as well for some possible
    # future use
    return image

# =============================================================================



# =============================================================================
# ======================= DIRECT METRIC RECTIFICATION =========================
# =============================================================================

# Compute the null space of the matrix A, with eps being the maximum acceptable
# error
def null_space(A, eps=1e-15):
   u, s, vh = scipy.linalg.svd(A)
   null_mask = (s <= eps)
   null_space = scipy.compress(null_mask, vh, axis=0)
   return scipy.transpose(null_space)

# Create the constraint on the conic dual of the circular points that
# corresponds to the two lines l and m
def create_equation(l, m):
    (l1, l2, l3) = [l.x, l.y, l.z]
    (m1, m2, m3) = [m.x, m.y, m.z]

    return [l1*m1, (l1*m2 + l2*m1)/2, l2*m2, (l1*m3 + l3*m1)/2,
            (l2*m3 + l3*m2)/2, l3*m3]

# Create the 6x6 matrix corresponding to the system of equations needed to find
# the image of the conic dual of the circular points. The first 5 rows
# correspond to the constraints derived from the orthogonal lines, and the 6th
# row is the length 6 zero vector, so that the matrix is square
def create_matrix(orthogonal_line_pairs):
    # Check input format
    if (len(orthogonal_line_pairs) != 5 or
            not is_line_pair_list(orthogonal_line_pairs)):
        raise Exception("Incorrect number of orthogonal line pairs")

    # Create the coefficient matrix A of the system Ac = 0, where c is a conic
    # as a 6-tuple and 0 is the length 6 zero vector
    equations = list()
    for (l, m) in orthogonal_line_pairs:
        equations.append(create_equation(l, m))
    equations.append([0]*6)

    return np.matrix(equations)

# Compute the projective transformation which occurred on the conic dual of the
# circular points. The input C corresponds to its distorted image
def compute_transformation(C):
    matrix = C.getA()

    c = matrix[0][0]
    b = matrix[0][1]
    d = matrix[1][1]
    a = sqrt(c - b*b)
    val1 = matrix[0][2]
    val2 = matrix[1][2]

    v1 = (val1 - b*val2) / (c - b*b)
    v2 = val2 - b*v1

    # H_p is the projective part of the transformation, similar to the H_p in
    # the stratified metric rectification
    H_p = np.matrix([[1, 0, 0], [0, 1, 0], [v1, v2, 1]])
    # H_a is the affine part of the transformation, similar to the H_a in the
    # stratified metric transformation
    H_a = np.matrix([[a, b, 0], [0, 1, 0], [0, 0, 1]])
    # The complete transformation (up to a similarity) is H_p*H_a
    H = H_p*H_a

    return H

#
def direct_metric_rect(image, orthogonal_line_pairs):
    # Compute the system of equations needed to identify C^*_inf. This is a 6x6
    # matrix, with the last row being the zero vector
    matrix = create_matrix(orthogonal_line_pairs)

    # Compute the conic C, as a 6 vector (a, b, c, d, e, f) which is the null
    # space of the system of equations
    C = np.array(null_space(matrix))

    # Create the conic matrix C, which is the image of C^*_inf
    (a, b, c, d, e, f) = map(lambda x: x[0], C)
    C = np.matrix([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
    # Normalize the KK^T elements in C to the format [[c, b], [b, 1]]
    C *= 1/c

    H = []
    try:
        # Compute the projective transformation that occurred on C^*_inf. This
        # is not the rectifying transformation, but the one corresponding to
        # the distortion
        H = compute_transformation(C)
    except ValueError:
        raise Exception("Bad line choices")

    # Return the rectified image
    return transform_image(np.array(H), image, invert=False)

# =============================================================================

# =============================================================================
# ================== ALGEBRIC CROOS RATIO RECTIFICATION =======================
# =============================================================================

def remove_projective_distortion_with_ratio(image, infinity_line):
    infinity_line.normalize()

    # Compute the projective transformation H_p, which returns the infinity
    # line to its correct position, removing projective distortion on the image
    a = infinity_line.x
    b = infinity_line.y
    c = infinity_line.z
    H_p = np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])

    # Compute the transformed image and return it
    return transform_image(H_p, image), H_p

def crossRatio_rect(image, infinity_line, orthogonal_line_pairs):
    # Remove projective distortions on the image (affine rectification)
    (image, H_p) = remove_projective_distortion_with_ratio(image, infinity_line)

    # Correct coordinates from the orthogonal lines
    orthogonal_line_pairs = deepcopy(orthogonal_line_pairs)
    H_p_line = inv(H_p).transpose()
    for (l1, l2) in orthogonal_line_pairs:
        l1.transform(H_p_line)
        l2.transform(H_p_line)

    # Remove affine distortions on the image (metric rectification)
    (image, H_a) = remove_affine_distortion(image, orthogonal_line_pairs)

    # TODO: Maybe return the complete transformation as well for some possible
    # future use
    return image
