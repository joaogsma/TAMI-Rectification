import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import sys
from line_builder import Line_Builder
from point import Point
from rectification import remove_projective_distortion
from rectification import direct_metric_rect
from scipy import misc
import time

# For image aquisition
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk


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


## Calculate vanish points
def get_H_2_2(a,b,c,alinha,blinha):
	h12 = 0
	h22 = 1
	h21 = ((b*alinha)-(a*blinha))/(alinha*b*(alinha+blinha))
	h11 = (a/alinha) + (a*h21)
	return h11,h12,h21,h22

# =============================================================================
# ========================== GET INPUT RATIO A:B ==============================
# =============================================================================

ratio = UserInputRatio().TempVar
ratio = float(ratio)


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

line_builder = Line_Builder(fig, ax, 2, col_num, row_num)

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

if len(lines) < 2:
    raise Exception("Not enough input lines")

## Teste Euclidean Distance
print ("First point X = ", points[0].x)
print ("First point Y =  ", points[0].y)
print ("Second point X = ", points[1].x)
print ("Second point Y = ", points[1].y)
print ("Euclidean Distance = ", points[0].euclideanDistance(points[1]))
print ("Euclidean Distance = ", points[0].euclideanDistance(points[0]))
print ("Euclidean Distance = ", points[1].euclideanDistance(points[1]))
print ("Euclidean Distance = ", points[1].euclideanDistance(points[0]))


# =============================================================================
# ======================== COMPUTE TWO VANISH POINT ===========================
# =============================================================================

###### First Vanish Point ######
# a/b = ratio    
a1 = ratio
b1 = 1

## Need to get input from c by user
c1 = Point(points[1].x,points[1].y+2,points[1].z)
alinha1 = points[0].euclideanDistance(points[1])
blinha1 = points[1].euclideanDistance(c1)

H1 = get_H_2_2(a1,b1,c1,alinha1,blinha1)
print (H1)

###### Second Vanish Point ######
# a/b = ratio same ratio as the first point
a2 = ratio
b2 = 1

## Need to get input from c by user
c2 = Point(points[3].x,points[3].y+2,points[3].z)
alinha2 = points[2].euclideanDistance(points[3])
blinha2 = points[3].euclideanDistance(c2)

H2 = get_H_2_2(a2,b2,c2,alinha2,blinha2)
print (H2)


## Nao sei se eh assim essa parte aqui de baixo
# Vanish Point one
PF1 = H1*Point(1,0,0)

# Vanish Point two
PF2 = H1*Point(1,0,0)

# Compute the Infinity Line
horizon = PF1.cross(PF2)
horizon.normalize()