import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from copy import deepcopy
from line_builder import LineBuilder
from point import Point
from scipy import misc

inverse = np.linalg.inv

def getPixel(img, x, y):
	return img[x][y]

def putPixel(img, x, y, r, g, b):
	img[x][y] = [r, g, b]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')

f = misc.imread(sys.argv[1], mode = 'RGB')

(ySize, xSize, _) = np.shape(f)

#print("(xSize, ySize) = (%d, %d)" % (xSize, ySize))


line1, = ax.plot([], [], color="r")  # empty line
line2, = ax.plot([], [], color="r")  # empty line
line3, = ax.plot([], [], color="g")  # empty line
line4, = ax.plot([], [], color="g")  # empty line
lb = LineBuilder(ax, line1, line2, line3, line4)

plt.imshow(f)
plt.show()
'''##### Produto vetorial para obter as Retas de Fuga #######################'''

P1 = Point(lb.xs1[0], lb.ys1[0], 1.0 )
P2 = Point(lb.xs1[1], lb.ys1[1], 1.0 )
P3 = Point(lb.xs2[0], lb.ys2[0], 1.0 )
P4 = Point(lb.xs2[1], lb.ys2[1], 1.0 )
P5 = Point(lb.xs3[0], lb.ys3[0], 1.0 )
P6 = Point(lb.xs3[1], lb.ys3[1], 1.0 )
P7 = Point(lb.xs4[0], lb.ys4[0], 1.0 )
P8 = Point(lb.xs4[1], lb.ys4[1], 1.0 )

P1.to_img_coord(xSize, ySize)
P2.to_img_coord(xSize, ySize)
P3.to_img_coord(xSize, ySize)
P4.to_img_coord(xSize, ySize)
P5.to_img_coord(xSize, ySize)
P6.to_img_coord(xSize, ySize)
P7.to_img_coord(xSize, ySize)
P8.to_img_coord(xSize, ySize)

reta1 = P1.cross(P2)
reta2 = P3.cross(P4)
reta3 = P5.cross(P6)
reta4 = P7.cross(P8)

'''##### Produto vetorial para obter os Pontos de Fuga ######################'''
PF1 = reta1.cross(reta2)
PF2 = reta3.cross(reta4)

horizonte = PF1.cross(PF2)
horizonte.normalize()

# exibindo novamente

p1_px = P1.get_pixel_coord(xSize, ySize)
p2_px = P2.get_pixel_coord(xSize, ySize)
p3_px = P3.get_pixel_coord(xSize, ySize)
p4_px = P4.get_pixel_coord(xSize, ySize)
p5_px = P5.get_pixel_coord(xSize, ySize)
p6_px = P6.get_pixel_coord(xSize, ySize)
p7_px = P7.get_pixel_coord(xSize, ySize)
p8_px = P8.get_pixel_coord(xSize, ySize)

pf1_px = PF1.get_pixel_coord(xSize, ySize)
pf2_px = PF2.get_pixel_coord(xSize, ySize)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(p1_px[0] , p1_px[1], c='r')
ax.scatter(p2_px[0] , p2_px[1], c='r')
ax.plot( [p1_px[0], p2_px[0]], [p1_px[1], p2_px[1]], color="r", linewidth=2.0)
ax.scatter(p3_px[0] , p3_px[1], c='r')
ax.scatter(p4_px[0] , p4_px[1], c='r')
ax.plot( [p3_px[0], p4_px[0]], [p3_px[1], p4_px[1]], color="r", linewidth=2.0)

ax.plot( [p2_px[0], pf1_px[0]], [p2_px[1], pf1_px[1]], "r--", linewidth=2.0)
ax.plot( [p4_px[0], pf1_px[0]], [p4_px[1], pf1_px[1]], "r--", linewidth=2.0)

ax.scatter(p5_px[0] , p5_px[1], c='g')
ax.scatter(p6_px[0] , p6_px[1], c='g')
ax.plot( [p5_px[0], p6_px[0]], [p5_px[1], p6_px[1]], color="g", linewidth=2.0)
ax.scatter(p7_px[0] , p7_px[1], c='g')
ax.scatter(p8_px[0] , p8_px[1], c='g')
ax.plot( [p7_px[0], p8_px[0]], [p7_px[1], p8_px[1]], color="g", linewidth=2.0)

ax.plot( [p6_px[0], pf2_px[0]], [p6_px[1], pf2_px[1]], "g--", linewidth=2.0)
ax.plot( [p8_px[0], pf2_px[0]], [p8_px[1], pf2_px[1]], "g--", linewidth=2.0)


''' HORIZONTE'''
ax.scatter(pf1_px[0] , pf1_px[1], c='b')
ax.scatter(pf2_px[0] , pf2_px[1], c='b')
ax.plot( [pf1_px[0], pf2_px[0]], [pf1_px[1], pf2_px[1]], color="b")
plt.imshow(f)
plt.show()

a = horizonte.x
b = horizonte.y
c = horizonte.z

''' Retificação Afim '''
H = np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])

f_ret = np.array([[ [np.uint8(0)]*3 for x in range(xSize)] for y in range(ySize)])

for col in range(0, xSize):
    for row in range(0, ySize):
        #print()
        #print(str(col) + "/" + str(xSize) + "  " + str(row) + "/" + str(ySize))
        pixelValue = f[row][col]
        p = Point(col, row, 1)
        p.to_img_coord(xSize, ySize)
        p.transform(H)
        p.normalize()
        (col_px, row_px)  = p.get_pixel_coord(xSize, ySize)
        
        if ((0 < col_px < xSize) and (0 < row_px < ySize)):
            #print(str(col_px) + "/" + str(xSize) + "  " + str(row_px) + "/" + str(ySize))
            f_ret[row_px][col_px][0] = pixelValue[0]
            f_ret[row_px][col_px][1] = pixelValue[1]
            f_ret[row_px][col_px][2] = pixelValue[2]


plt.imshow(f_ret)
plt.show()
