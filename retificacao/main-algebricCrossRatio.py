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
from userInputRatio import UserInputRatio
import time

from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk


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

a = UserInputRatio("a").TempVar
a = float(a)

b = UserInputRatio("b").TempVar
b = float(b)

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

line_builder = Line_Builder(fig, ax, 6, col_num, row_num)

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

if len(lines) < 6:
    raise Exception("Not enough input lines")


## Teste Euclidean Distance
# print ("First point X = ", points[0].x)
# print ("First point Y =  ", points[0].y)
# print ("Second point X = ", points[1].x)
# print ("Second point Y = ", points[1].y)
# print ("Euclidean Distance = ", points[0].euclideanDistance(points[1]))
# print ("Euclidean Distance = ", points[0].euclideanDistance(points[0]))
# print ("Euclidean Distance = ", points[1].euclideanDistance(points[1]))
# print ("Euclidean Distance = ", points[1].euclideanDistance(points[0]))


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

h11,h12,h21,h22 = get_H_2_2(a1,b1,c1,alinha1,blinha1)
H1 = np.array([[h11,h12], [h21,h22]])
print(H1)

###### Second Vanish Point ######
# a/b = ratio same ratio as the first point
a2 = ratio
b2 = 1

## Need to get input from c by user
c2 = Point(points[3].x,points[3].y+2,points[3].z)
alinha2 = points[2].euclideanDistance(points[3])
blinha2 = points[3].euclideanDistance(c2)

h11,h12,h21,h22 = get_H_2_2(a2,b2,c2,alinha2,blinha2)
H2 = np.array([[h11,h12], [h21,h22]])
print (H2)


# # Nao sei se eh assim essa parte aqui de baixo
# # Vanish Point one
PF1 = Point2(1,0)
PF1.transform(H1)
print("PF1: ", PF1)
PF1.normalize()
print("PF1: ", PF1)
# # Vanish Point two
PF2 = Point2(1,0)
PF2.transform(H2)
print("PF2:", PF2)
PF2.normalize()


# # Compute the Infinity Line
# horizon = PF1.cross(PF2)
# horizon.normalize()

(coordx,coordy) = PF1.to_nparray()
vetorDif1 = blinha1-alinha1
vetorDif1 = Point(vetorDif1,1,1)
vetorDif1.r3Normalize()
vetorDif1 = vetorDif1*coordx
PF1 = vetorDif1 + Point(alinha1,1,1)
print(PF1)


(coordx,coordy) = PF2.to_nparray()
vetorDif2 = blinha2-alinha2
vetorDif2 = Point(vetorDif2,1,1)
vetorDif2.r3Normalize()
vetorDif2 = vetorDif2*coordx
PF2 = vetorDif2 + Point(alinha2,1,1)
print(PF2)

# Compute the Infinity Line
horizon = PF1.cross(PF2)
horizon.normalize()


line_pairs = list()
i = 2
while i < len(lines):
    line_pairs.append( (lines[i], lines[i+1]) )
    i += 2

perpendicular_line_pairs = line_pairs[2:]



#print(perpendicular_line_pairs)

#tic = time.clock()
algebric_crossRatio_rect(image, horizon, perpendicular_line_pairs)

fig = plt.figure()
fig.canvas.set_window_title('Rectified Image (Stratified Metric Rectification)')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image_)
plt.show()

# Lembrei
# Vc vai multiplicar o infinito de P1, que eh (1,0) por H
# E normalizar
# Aí vc vai ter a imagem dele normalizada
# Que eh no formato (x, 1)
# Então x eh a distância de alinha pra ele
# Vc pega então o vetor blinha - alinha
# Normaliza ele pra ter comprimento 1
# (normalização de vetores de R3, não eh a normalização de P2)
# Multiplica por x e soma ao ponto alinha
# Aí vc tem o ponto de fuga dessa reta
# Eh so repetir pra outra
# Agora já tá aí
# Eh esse msm o procedimento
# Outra coisa
# Tem a forma geométrica tbm ora fazer


# =============================================================================
