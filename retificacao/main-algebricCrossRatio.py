import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import sys
from line_builder import Line_Builder
from point import Point
from point import Point2
from rectification import remove_projective_distortion_with_ratio
from rectification import crossRatio_rect
from scipy import misc
import time

# For image aquisition
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

import math

class UserInputRatio():
    def __init__(self):
        self.Master=Tk()
        self.Entry=Entry(self.Master)
        self.Master.wm_title("Ratio a:b")
        self.Entry.pack()

        self.Button=Button(self.Master,text="Ok",command=self.Return)
        self.Button.pack()            

        self.Master.mainloop()
    
    def Return(self):
        self.TempVar=self.Entry.get() 
        self.Entry.quit()


## Close window and change progress in code
def press(event):
    #print('press', event.key)
    if event.key == 'enter':
        plt.close()


def calcAlgebricPFbyCrossRatio(A_, B_, C_, a, b):
    A_distance = A_.euclideanDistance(B_)
    B_distance = B_.euclideanDistance(C_) 

  #  h12 = 0
  #  h22 = 1
  #  h21 = ((b*A_distance)-(a*B_distance))/(A_distance*b*(A_distance+B_distance))
  #  h11 = (a/A_distance) + (a*h21)
   

    h12 = 0
    h22 = A_.y
    h11 = A_distance/a  
    h21 = (A_distance/(a*(A_distance+B_distance))) - (1/(a+b))

    H = np.array([[h11,h12], [h21,h22]])

    PF = Point2(1,0)
    PF.transform(H)
    PF.normalize()

    (x,y) = PF.to_nparray()
  #  print("X: ", x)
  #  print("Y: ", y)
 
    vetorDif1 = (B_-A_)
   # print("VetorDif: ",vetorDif1)
    vetorDif1.r3Normalize()
    vetorDif1 = vetorDif1*x
   # print("Vetor dif & x: ", vetorDif1)
    PF = vetorDif1 + A_
    PF.normalize()
   # print("PF: ",PF)
    return PF

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

A1 = points[0]
B1 = points[1]
C1 = points[3]

A2 = points[4]
B2 = points[5]
C2 = points[7]

PF1 = calcAlgebricPFbyCrossRatio(A1, B1, C1, 1, 1)
PF2 = calcAlgebricPFbyCrossRatio(A2, B2, C2, 1, 1)


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

