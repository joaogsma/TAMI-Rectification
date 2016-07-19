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


#def openImage():



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

line_builder = Line_Builder(fig, ax, 10, col_num, row_num)

# =============================================================================


fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image)
plt.show()


# =============================================================================
# ========================= COMPUTE THE INFINITY LINE =========================
# =============================================================================

points = line_builder.get_points()

lines = line_builder.get_lines()

if len(lines) < 10:
    raise Exception("Not enough input lines")

# =============================================================================



# =============================================================================
# ======================= DIRECT METRIC RECTIFICATION =========================
# =============================================================================

for line in lines:
    line.normalize()

orthogonal_line_pairs = list()
i = 0
while i < len(lines):
    orthogonal_line_pairs.append( (lines[i], lines[i+1]) )
    i += 2 

#tic = time.clock()
image_ = direct_metric_rect(image, orthogonal_line_pairs)
#toc = time.clock()
#print(toc - tic)

# =============================================================================

fig = plt.figure()
fig.canvas.set_window_title('Rectified Image (Stratified Metric Rectification)')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image_)
plt.show()