import numpy as np
import math

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
        if (type(other) != int and type(other) != float and
            type(other) != np.int_ and type(other) != np.float_):
            raise Exception("Undefined Point division")

        if other == 0:
            raise Exception("Division by zero")

        return Point(self.x / other, self.y / other, self.z / other)
    
    def __str__(self):
        return "(%f, %f, %f)" % (self.x, self.y, self.z)

    def euclideanDistance(self,other):
        diff = self-other
        return math.sqrt(diff*diff)
    
    def cross(self, other):
        return Point(self.y*other.z - self.z*other.y, 
                     self.z*other.x - self.x*other.z, 
                     self.x*other.y - self.y*other.x)

    def get_pixel_coord(self, xSize, ySize):
        self.normalize()
        x = int(round(self.x + (xSize / 2)))
        y = int(round((ySize / 2) - self.y))

        return (y, x)


    def to_img_coord(self, xSize, ySize):
        self.x = self.x - (xSize / 2)
        self.y = (ySize / 2) - self.y

    def normalize(self):
        self.x /= self.z
        self.y /= self.z
        self.z = 1

    def transform(self, H):
        if len(H) != 3:
            raise Exception("Incorrect matrix size")

        for line in H:
            if len(line) != 3: 
                raise Exception("Incorrect matrix size")

        new_point = np.array([ H[0][0]*self.x + H[0][1]*self.y + H[0][2]*self.z, 
                               H[1][0]*self.x + H[1][1]*self.y + H[1][2]*self.z, 
                               H[2][0]*self.x + H[2][1]*self.y + H[2][2]*self.z ])

        self.x = new_point[0]
        self.y = new_point[1]
        self.z = new_point[2]

    def to_nparray(self):
        return np.array([self.x, self.y, self.z])
