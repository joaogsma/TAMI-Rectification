import numpy as np
import matplotlib.pyplot as plt
import sys
from line_builder import LineBuilder
from point import Point
from scipy import misc


def getPixel(img, x, y):
	return img[x][y]

def putPixel(img, x, y, r, g, b):
	img[x][y] = [r, g, b]

# TODO
#def transpose(H):
#    if len(H) != 3:
#            raise Exception("Incorrect matrix size")
#
#    for line in H:
#        if len(line) != 3: 
#            raise Exception("Incorrect matrix size")



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')

f = misc.imread(sys.argv[1], mode = 'L')

(xSize, ySize,) = np.shape(f)
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

print("P1 = " + str(P1))
print("P2 = " + str(P2))

reta1 = P1.cross(P2)
reta2 = P3.cross(P4)
reta3 = P5.cross(P6)
reta4 = P7.cross(P8)

print("reta P1-P2: " + str(reta1))

'''##### Produto vetorial para obter os Pontos de Fuga ######################'''
PF1 = reta1.cross(reta2)
PF2 = reta3.cross(reta4)

horizonte = PF1.cross(PF2)

reta1.normalize()
PF1.normalize()
PF2.normalize()
horizonte.normalize()

print("reta1 = ", reta1)
print("PF1 = ", PF1)
print("PF2 = ", PF2)
print("horizonte = ", horizonte)


# exibindo novamente

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(P1.x , P1.y, c='r')
ax.scatter(P2.x , P2.y, c='r')
ax.plot( [P1.x, P2.x], [P1.y, P2.y], color="r", linewidth=2.0)
ax.scatter(P3.x , P3.y, c='r')
ax.scatter(P4.x , P4.y, c='r')
ax.plot( [P3.x, P4.x], [P3.y, P4.y], color="r", linewidth=2.0)

ax.plot( [P2.x, PF1.x], [P2.y, PF1.y], "r--", linewidth=2.0)
ax.plot( [P4.x, PF1.x], [P4.y, PF1.y], "r--", linewidth=2.0)

ax.scatter(P5.x , P5.y, c='g')
ax.scatter(P6.x , P6.y, c='g')
ax.plot( [P5.x, P6.x], [P5.y, P6.y], color="g")
ax.scatter(P7.x , P7.y, c='g')
ax.scatter(P8.x , P8.y, c='g')
ax.plot( [P7.x, P8.x], [P7.y, P8.y], color="g")

ax.plot([P6.x, PF2.x], [P6.y, PF2.y], "g--", linewidth=2.0)
ax.plot([P8.x, PF2.x], [P8.y, PF2.y], "g--", linewidth=2.0)


''' HORIZONTE'''
ax.scatter(PF1.x , PF1.y, c='b')
ax.scatter(PF2.x , PF2.y, c='b')
ax.plot( [PF1.x, PF2.x], [PF1.y, PF2.y], color="b")
plt.imshow(f)
plt.show()

a = horizonte.x
b = horizonte.y
c = horizonte.z

''' Retificação Afim '''
H = np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])

print("H: " + str(H))

f_ret = [[0 for x in range(xSize)] for y in range(ySize)]

for x in range(0, xSize):
    for y in range(0, ySize):
        pixelValue = f[x][y]
        p = Point(x, y, 1)
        p.transform(H)
        p.normalize()
        xr = int(p.x)
        yr = int(p.y)
        if ((0 < xr < xSize) and (0 < yr < ySize)):
            #f_ret[xr][yr] = f_ret[xr][yr] + pixelValue
            f_ret[xr][yr] = pixelValue
            #print("(x, y) = (%d, %d)" %(x, y))


PF1.transform(H)
PF2.transform(H)
L_inf = PF1.cross(PF2)
L_inf.normalize()

print("Novo horizonte: " + str(L_inf))

P1.transform(H)
P2.transform(H)
P3.transform(H)
P4.transform(H)
P5.transform(H)
P6.transform(H)
P7.transform(H)
P8.transform(H)

reta1 = P1.cross(P2)
reta2 = P3.cross(P4)
reta3 = P5.cross(P6)
reta4 = P7.cross(P8)

reta1.normalize()
reta2.normalize()
reta3.normalize()
reta4.normalize()

print("Reta1: " + str(reta1))
print("Reta2: " + str(reta2))
print("Reta3: " + str(reta3))
print("Reta4: " + str(reta4))

print("Reta1 X Reta2: " + str(reta1.cross(reta2)))
print("Reta3 X Reta4: " + str(reta3.cross(reta4)))

plt.imshow(f_ret)
plt.show()
