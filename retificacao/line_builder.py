import matplotlib.pyplot as plt
from point import Point



class Line_Builder:
    def __init__(self, fig, ax, n_lines, col_num, row_num, crossRatio=False):
        self.ax = ax
        self.col_num = col_num
        self.row_num = row_num
        self.n_lines = n_lines
        self.crossRatio = crossRatio

        self.lines = list()
        while len(self.lines) < n_lines:
            draw_color = ("r" if len(self.lines) < 2 else
                          ("g" if len(self.lines) < 4 else
                            ("b" if len(self.lines) < 6 else "c")))
            l, = ax.plot([], [], color=draw_color)  # empty line
            self.lines.append(l)

        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.countLines = 0
        self.countPoints = 0
        self.x_list = list()
        self.y_list = list()

    def get_points(self):
        if len(self.x_list) != len(self.y_list):
            raise Exception("Error during line building")

        points = list()

        i = 0
        while i < len(self.x_list):
            p = Point(self.x_list[i], self.y_list[i], 1)
            i += 1
            p.to_img_coord(self.col_num, self.row_num)
            points.append(p)

        return points

    def get_lines(self):
        if (len(self.x_list) != len(self.y_list) or
                len(self.x_list) != 2*len(self.lines)):
            raise Exception("Error during line building")

        lines_ = list()

        i = 0
        while i < len(self.x_list):
            p = Point(self.x_list[i], self.y_list[i], 1)
            q = Point(self.x_list[i+1], self.y_list[i+1], 1)
            i += 2
            p.to_img_coord(self.col_num, self.row_num)
            q.to_img_coord(self.col_num, self.row_num)
            lines_.append(p.cross(q))

        return lines_

    def calcColinearPoint(self, P0, PF, x):
        P0.normalize()
        PF.normalize()
        x0 = P0.x
        y0 = P0.y
        xf = PF.x
        yf = PF.y
        print("P0 = ", P0)
        print("PF = ", PF)
        a = (yf - y0)/(xf - x0)
        b = y0 - a*x0
        y = a*x + b
        P = Point(x,y,1)
        return P


    def __call__(self, event):
        if event.inaxes!=self.lines[0].axes:
            return

        if self.countLines >= 2*self.n_lines:
            return

        draw_color = ("r" if self.countLines < 4 else
                      ("g" if self.countLines < 8 else
                        ("b" if self.countLines < 12 else "c")))

        x_ = event.xdata
        y_ = event.ydata

        if (self.crossRatio and ((self.countPoints == 1) or (self.countPoints == 5))):

            self.x_list.append(x_)
            self.x_list.append(x_)
            self.y_list.append(y_)
            self.y_list.append(y_)
            self.countPoints += 2
            self.countLines += 1
            print("countPoints = ", self.countPoints)
        elif (self.crossRatio and ((self.countPoints == 3) or (self.countPoints == 7))):
            print("AQUI")
            i = len(self.x_list) - 1
            PF = Point(self.x_list[i], self.y_list[i], 1)
            P0 = Point(self.x_list[i-2], self.y_list[i-2], 1)
            P = self.calcColinearPoint(P0, PF, x_)

            self.x_list.append(P.x)
            self.y_list.append(P.y)
            self.countPoints += 1
        else:
            self.x_list.append(x_)
            self.y_list.append(y_)
            self.countPoints += 1

        line = self.lines[int(self.countLines/2)]

        if (self.crossRatio and (self.countLines == 3) or (self.countLines == 7)):
            xs = self.x_list[len(self.x_list) - 4 + (len(self.x_list) % 4):]
            ys = self.y_list[len(self.x_list) - 4 + (len(self.x_list) % 4):]
        else:
            xs = self.x_list[len(self.x_list) - 2 + (len(self.x_list) % 2):]
            ys = self.y_list[len(self.x_list) - 2 + (len(self.x_list) % 2):]

        line.set_data(xs, ys)
        self.ax.scatter(xs, ys, c=draw_color)
        line.figure.canvas.draw()
        self.countLines += 1
