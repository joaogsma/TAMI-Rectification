import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from copy import deepcopy
from line_builder import LineBuilder
from point import Point
from scipy import misc

inverse = np.linalg.inv

# =============================================================================
# ============================== LOAD THE IMAGE ===============================
# =============================================================================
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')

f = misc.imread(sys.argv[1], mode = 'RGB')

(row_num, col_num, _) = np.shape(f)
# -----------------------------------------------------------------------------



# =============================================================================
# ==================== CREATE LISTENERS FOR POINT CAPTURE =====================
# =============================================================================
line1, = ax.plot([], [], color="r")  # empty line
line2, = ax.plot([], [], color="r")  # empty line
line3, = ax.plot([], [], color="g")  # empty line
line4, = ax.plot([], [], color="g")  # empty line
lb = LineBuilder(ax, line1, line2, line3, line4)
# -----------------------------------------------------------------------------



plt.imshow(f)
plt.show()



# =============================================================================
# ========================= COMPUTE THE INFINITY LINE =========================
# =============================================================================
p1 = Point(lb.xs1[0], lb.ys1[0], 1.0 )
p2 = Point(lb.xs1[1], lb.ys1[1], 1.0 )
p3 = Point(lb.xs2[0], lb.ys2[0], 1.0 )
p4 = Point(lb.xs2[1], lb.ys2[1], 1.0 )
p5 = Point(lb.xs3[0], lb.ys3[0], 1.0 )
p6 = Point(lb.xs3[1], lb.ys3[1], 1.0 )
p7 = Point(lb.xs4[0], lb.ys4[0], 1.0 )
p8 = Point(lb.xs4[1], lb.ys4[1], 1.0 )

p1.to_img_coord(col_num, row_num)
p2.to_img_coord(col_num, row_num)
p3.to_img_coord(col_num, row_num)
p4.to_img_coord(col_num, row_num)
p5.to_img_coord(col_num, row_num)
p6.to_img_coord(col_num, row_num)
p7.to_img_coord(col_num, row_num)
p8.to_img_coord(col_num, row_num)

reta1 = p1.cross(p2)
reta2 = p3.cross(p4)
reta3 = p5.cross(p6)
reta4 = p7.cross(p8)

# Compute points in the Infinity Line
PF1 = reta1.cross(reta2)
PF2 = reta3.cross(reta4)

# Compute the Infinity Line
horizon = PF1.cross(PF2)
horizon.normalize()
# -----------------------------------------------------------------------------



# =============================================================================
# ============================= UPDATE THE IMAGE ==============================
# =============================================================================
p1_px = p1.get_pixel_coord(col_num, row_num)
p2_px = p2.get_pixel_coord(col_num, row_num)
p3_px = p3.get_pixel_coord(col_num, row_num)
p4_px = p4.get_pixel_coord(col_num, row_num)
p5_px = p5.get_pixel_coord(col_num, row_num)
p6_px = p6.get_pixel_coord(col_num, row_num)
p7_px = p7.get_pixel_coord(col_num, row_num)
p8_px = p8.get_pixel_coord(col_num, row_num)

pf1_px = PF1.get_pixel_coord(col_num, row_num)
pf2_px = PF2.get_pixel_coord(col_num, row_num)

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


# Draw the horizon (Infinity Line)
ax.scatter(pf1_px[0] , pf1_px[1], c='b')
ax.scatter(pf2_px[0] , pf2_px[1], c='b')
ax.plot( [pf1_px[0], pf2_px[0]], [pf1_px[1], pf2_px[1]], color="b")
plt.imshow(f)
plt.show()
# -----------------------------------------------------------------------------



# =============================================================================
# =========================== AFFINE RECTIFICATION ============================
# =============================================================================
a = horizon.x
b = horizon.y
c = horizon.z
H = np.array([[1, 0, 0], [0, 1, 0], [a, b, c]])

f_ret = np.array([[ [np.uint8(0)]*3 for col in range(col_num)] for row in range(row_num)])

for col in range(0, col_num):
    for row in range(0, row_num):
        pixelValue = f[row][col]
        p = Point(col, row, 1)
        p.to_img_coord(col_num, row_num)
        p.transform(H)
        p.normalize()
        (col_px, row_px)  = p.get_pixel_coord(col_num, row_num)
        
        if ((0 < col_px < col_num) and (0 < row_px < row_num)):
            f_ret[row_px][col_px][0] = pixelValue[0]
            f_ret[row_px][col_px][1] = pixelValue[1]
            f_ret[row_px][col_px][2] = pixelValue[2]
# -----------------------------------------------------------------------------



plt.imshow(f_ret)
plt.show()
