import sys
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')


class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, other):
        if(type(other) == Point):
            return (self.x * other.x + self.y * other.y + self.z * other.z)
        else:
            return Point(self.x * other, self.y * other, self.z * other)
    def __rmul__(self, other):
        if(type(other) == Point):
            return (self.x * other.x + self.y * other.y + self.z * other.z)
        else:
            return Point(self.x * other, self.y * other, self.z * other)
    def __truediv__(self, other):
        if(type(other) != Point):
            return Point(self.x / other, self.y / other, self.z / other)
        else:
            return (self.x / other.x + self.y / other.y + self.z / other.z)
    def __pow__(self, other):
        return Point(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)
    def __rpow__(self, other):
        return Point(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)
    def __str__(self):
        return "(%f, %f, %f)" % (self.x, self.y, self.z)
    def to_nparray(self):
        return np.array([self.x, self.y, self.z])


class LineBuilder:
    def __init__(self, line1, line2, line3, line4):
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3
        self.line4 = line4

        '''
        self.xs = list(line1.get_xdata())
        self.ys = list(line1.get_ydata())'''

        self.xs1 = list(line1.get_xdata())
        self.ys1 = list(line1.get_ydata())

        self.xs2 = list(line2.get_xdata())
        self.ys2 = list(line2.get_ydata())

        self.xs3 = list(line3.get_xdata())
        self.ys3 = list(line3.get_ydata())

        self.xs4 = list(line4.get_xdata())
        self.ys4 = list(line4.get_ydata())

        self.cid = line1.figure.canvas.mpl_connect('button_press_event', self)
        self.count = 0

    '''def auxCall(self):
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        ax.scatter(self.xs, self.ys, c='r')
        self.count = self.count + 1'''
    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line1.axes: return
        if (self.count <= 1):
            self.xs1.append(event.xdata)
            self.ys1.append(event.ydata)
            #self.auxCall()
            self.line1.set_data(self.xs1, self.ys1)
            ax.scatter(self.xs1, self.ys1, c='r')
            self.line1.figure.canvas.draw()
            self.count = self.count + 1
            '''if (self.count == 2):
                self.xs2 = []#list(line2.get_xdata())
                self.ys2 = []#list(line2.get_ydata())'''
        elif (2 <= self.count <= 3):
            self.xs2.append(event.xdata)
            self.ys2.append(event.ydata)
            #self.auxCall()
            self.line2.set_data(self.xs2, self.ys2)
            ax.scatter(self.xs2, self.ys2, c='r')
            self.line2.figure.canvas.draw()
            self.count = self.count + 1
            '''if (self.count == 3):
                self.xs = list(line3.get_xdata())
                self.ys = list(line3.get_ydata())'''
        elif (4 <= self.count <= 5):
            self.xs3.append(event.xdata)
            self.ys3.append(event.ydata)
            #self.auxCall()
            self.line3.set_data(self.xs3, self.ys3)
            ax.scatter(self.xs3, self.ys3, c='g')
            self.line3.figure.canvas.draw()
            self.count = self.count + 1
            '''if (self.count == 5):
                self.xs = list(line4.get_xdata())
                self.ys = list(line4.get_ydata())'''
        elif (6 <= self.count <= 7):
            self.xs4.append(event.xdata)
            self.ys4.append(event.ydata)
            #self.auxCall()
            self.line4.set_data(self.xs4, self.ys4)
            ax.scatter(self.xs4, self.ys4, c='g')
            self.line4.figure.canvas.draw()
            self.count = self.count + 1
            if (self.count == 7):
                return
        '''else:
            print("(xs1, ys1)", xs1[0], ys1[0])
            ax = fig.add_subplot(110)
            ax.set_title('(xs1(0), ys1(0)) = ', xs1[0], ys1[0])'''


def getPixel(img, x, y):
	return img[x][y]

def putPixel(img, x, y, r, g, b):
	img[x][y] = [r, g, b]

f = misc.imread(sys.argv[1], mode = 'L')

(xSize, ySize,) = np.shape(f)
#print("(xSize, ySize) = (%d, %d)" % (xSize, ySize))


line1, = ax.plot([], [], color="r")  # empty line
line2, = ax.plot([], [], color="r")  # empty line
line3, = ax.plot([], [], color="g")  # empty line
line4, = ax.plot([], [], color="g")  # empty line
lb = LineBuilder(line1, line2, line3, line4)

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

reta1 = P1**P2
reta2 = P3**P4
reta3 = P5**P6
reta4 = P7**P8

'''##### Produto vetorial para obter os Pontos de Fuga ######################'''
PF1 = reta1**reta2
PF2 = reta3**reta4

horizonte = PF1**PF2

reta1 = reta1/reta1.z
PF1 = PF1/PF1.z
PF2 = PF2/PF2.z
horizonte = horizonte/horizonte.z

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

print("type f: ", type(f))
print("type H: ", type(H))

def transf(H, P):
	[P0, P1, P2] = P
	return np.array([H[0][0]*P0 + H[0][1]*P1 + H[0][2]*P2, H[1][0]*P0 + H[1][1]*P1 + H[1][2]*P2, H[2][0]*P0 + H[2][1]*P1 + H[2][2]*P2])

f_ret = [[0 for x in range(xSize)] for y in range(ySize)]

for x in range(0, xSize):
    for y in range(0, ySize):
        pixelValue = f[x][y]
        p = np.array([x, y, 1])
        #print("H = ", H)
        #print("p = ", p)

        p_ret = transf(H, p)
        #print("p_ret = ", p_ret)
        p_ret = Point(p_ret[0], p_ret[1], p_ret[2])
        p_ret = p_ret/p_ret.z #normalizando
        #print("p_ret = ", p_ret)
        #print("Type(p_ret) = ", type(p_ret))
        xr = int(p_ret.x)
        yr = int(p_ret.y)
        if ((0 < xr < xSize) and (0 < yr < ySize)):
            f_ret[xr][yr] = f_ret[xr][yr] + pixelValue
            print("(x, y) = (%d, %d)" %(x, y))


plt.imshow(f_ret)
plt.show()

'''
print("line", len(lb.xs1))
print("")
print("")
print("lb.xs1 = ", lb.xs1)
print("lb.xs4 = ", lb.xs4)'''
