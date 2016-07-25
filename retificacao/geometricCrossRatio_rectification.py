import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import sys
from line_builder import Line_Builder
from point import Point
from rectification import remove_projective_distortion
from rectification import crossRatio_rect
from scipy import misc
import time

# For image aquisition
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
<<<<<<< HEAD:retificacao/main-geometricCrossRatio.py
from userInputRatio import UserInputRatio

#def openImage():
'''
def calcPFbyCrossRatio(A, B, C):

    [xA, yA, zA] = A.to_nparray()
    [xB, yB, zB] = B.to_nparray()
    [xC, yC, zC] = C.to_nparray()

    AB = A.cross(B)
    BC = B.cross(C)

    AB.normalize()
    BC.normalize()

    [xAB, yAB, zAB] = AB.to_nparray()
    [xBC, yBC, zBC] = BC.to_nparray()

    alpha = xAB*yC - yAB*xC - xBC*yA + yBC*xA
    beta  = zAB*xC - xAB*zC - zBC*xA + xBC*zA
    gamma = yAB*zC - zAB*yC - yBC*zA + zBC*yA

    yF = (-xAB*alpha + zAB*gamma)/(xAB*beta - yAB*alpha)
    xF = -(yAB*yF + zAB)/xAB

    F = Point(-xF, -yF, 1)

    return F
'''
=======

from userInputRatio import UserInputRatio
>>>>>>> e554ca9e00656359d7389297c5a59698f6e1d0a6:retificacao/geometricCrossRatio_rectification.py

def calcPFbyCrossRatio(A_, B_, C_, a, b):
    A = A_
    B = A + Point(a,0,0)
    C = B + Point(b,0,0)

    BB_ = B.cross(B_)
    BB_.normalize()
    CC_ = C.cross(C_)
    CC_.normalize()
    O = BB_.cross(CC_)
    O.normalize()
    [xO, yO, zO] = O.to_nparray()

    l = A.cross(C)
    l.normalize()
    [xl, yl, zl] = l.to_nparray()
    zl_ = -(xl*xO + yl*yO)/zO
    l_ = Point(xl, yl, zl_)
    l_.normalize()
    A_C_ = A_.cross(C_)
    A_C_.normalize()
    PF = A_C_.cross(l_)
    PF.normalize()

    return PF

## Close window and change progress in code
def press(event):
    #print('press', event.key)
    if event.key == 'enter':
        plt.close()

# =============================================================================
# ============================== LOAD THE IMAGE ===============================
# =============================================================================

filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
                                        ("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
                                        ("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

image = misc.imread(filename, mode = 'RGB')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Click to build line segments')

(row_num, col_num, _) = image.shape

# =============================================================================



# =============================================================================
# ==================== CREATE LISTENERS FOR POINT CAPTURE =====================
# =============================================================================

line_builder = Line_Builder(fig, ax, 8, col_num, row_num, crossRatio=True)

# =============================================================================


fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image)
plt.show()


# =============================================================================
# ========================= COMPUTE THE INFINITY LINE =========================
# =============================================================================

points = line_builder.get_points()
#(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16) = line_builder.get_points()

lines = line_builder.get_lines()
#(line1, line2, line3, line4, line5, line6, line7, line8) = line_builder.get_lines()

if len(lines) < 8:
    raise Exception("Not enough input lines")

#points

A1 = points[0]
B1 = points[1]
C1 = points[3]

A2 = points[4]
B2 = points[5]
C2 = points[7]

<<<<<<< HEAD:retificacao/main-geometricCrossRatio.py
a1 = UserInputRatio("a1").TempVar
a1 = float(a1)

b1 = UserInputRatio("b1").TempVar
b1 = float(b1)

a2 = UserInputRatio("a2").TempVar
a2 = float(a2)

b2 = UserInputRatio("b2").TempVar
b2 = float(b2)

=======
a1 = float(UserInputRatio("a1").TempVar)
b1 = float(UserInputRatio("b1").TempVar)
a2 = float(UserInputRatio("a2").TempVar)
b2 = float(UserInputRatio("b2").TempVar)
>>>>>>> e554ca9e00656359d7389297c5a59698f6e1d0a6:retificacao/geometricCrossRatio_rectification.py

PF1 = calcPFbyCrossRatio(A1, B1, C1, a1, b1)
PF2 = calcPFbyCrossRatio(A2, B2, C2, a2, b2)

# Compute points in the Infinity Line
#PF1 = lines[0].cross(lines[1])
#PF2 = lines[2].cross(lines[3])

# Compute the Infinity Line
horizon = PF1.cross(PF2)
horizon.normalize()

# =============================================================================



# =============================================================================
# ============================= UPDATE THE IMAGE ==============================
# =============================================================================

p1_px = points[0].get_pixel_coord(col_num, row_num)
p2_px = points[1].get_pixel_coord(col_num, row_num)
p3_px = points[2].get_pixel_coord(col_num, row_num)
p4_px = points[3].get_pixel_coord(col_num, row_num)
p5_px = points[4].get_pixel_coord(col_num, row_num)
p6_px = points[5].get_pixel_coord(col_num, row_num)
p7_px = points[6].get_pixel_coord(col_num, row_num)
p8_px = points[7].get_pixel_coord(col_num, row_num)

pf1_px = PF1.get_pixel_coord(col_num, row_num)
pf2_px = PF2.get_pixel_coord(col_num, row_num)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(p1_px[1] , p1_px[0], c='r')
ax.scatter(p2_px[1] , p2_px[0], c='r')
ax.plot( [p1_px[1], p2_px[1]], [p1_px[0], p2_px[0]], color="r", linewidth=2.0)
ax.scatter(p3_px[1] , p3_px[0], c='r')
ax.scatter(p4_px[1] , p4_px[0], c='r')
ax.plot( [p3_px[1], p4_px[1]], [p3_px[0], p4_px[0]], color="r", linewidth=2.0)

ax.plot( [p2_px[1], pf1_px[1]], [p2_px[0], pf1_px[0]], "r--", linewidth=2.0)
ax.plot( [p4_px[1], pf1_px[1]], [p4_px[0], pf1_px[0]], "r--", linewidth=2.0)

ax.scatter(p5_px[1] , p5_px[0], c='g')
ax.scatter(p6_px[1] , p6_px[0], c='g')
ax.plot( [p5_px[1], p6_px[1]], [p5_px[0], p6_px[0]], color="g", linewidth=2.0)
ax.scatter(p7_px[1] , p7_px[0], c='g')
ax.scatter(p8_px[1] , p8_px[0], c='g')
ax.plot( [p7_px[1], p8_px[1]], [p7_px[0], p8_px[0]], color="g", linewidth=2.0)

ax.plot( [p6_px[1], pf2_px[1]], [p6_px[0], pf2_px[0]], "g--", linewidth=2.0)
ax.plot( [p8_px[1], pf2_px[1]], [p8_px[0], pf2_px[0]], "g--", linewidth=2.0)


# Draw the horizon (Infinity Line)
ax.scatter(pf1_px[1] , pf1_px[0], c='b')
ax.scatter(pf2_px[1] , pf2_px[0], c='b')
ax.plot( [pf1_px[1], pf2_px[1]], [pf1_px[0], pf2_px[0]], color="b")

fig.canvas.set_window_title('Original Image with Infinity Line')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image)
plt.show()

# =============================================================================


# =============================================================================
# ===================== STRATIFIED METRIC RECTIFICATION =======================
# =============================================================================

line_pairs = list()
i = 0
while i < len(lines):
    line_pairs.append( (lines[i], lines[i+1]) )
    i += 2

parallel_line_pairs = line_pairs[:2]
perpendicular_line_pairs = line_pairs[2:]

#tic = time.clock()
image_ = crossRatio_rect(image, horizon, perpendicular_line_pairs)
#stratified_metric_rect(image, parallel_line_pairs,
#                            perpendicular_line_pairs)
#toc = time.clock()
#print(toc - tic)

# =============================================================================

fig = plt.figure()
fig.canvas.set_window_title('Rectified Image (Stratified Metric Rectification)')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image_)
plt.show()
