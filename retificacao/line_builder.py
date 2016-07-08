import matplotlib.pyplot as plt
from point import Point

class Line_Builder:
    def __init__(self, fig, ax, n_lines, col_num, row_num):
        self.ax = ax
        self.col_num = col_num
        self.row_num = row_num
        self.n_lines = n_lines

        self.lines = list()
        while len(self.lines) < n_lines:
            draw_color = ("r" if len(self.lines)<2 else 
                     ("g" if len(self.lines)<4 else "b"))
            l, = ax.plot([], [], color=draw_color)  # empty line
            self.lines.append(l)

        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.count = 0

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



    def __call__(self, event):
        if event.inaxes!=self.lines[0].axes:
            return

        if self.count >= 2*self.n_lines:
            return
        
        draw_color = ("r" if self.count<4 else 
                     ("g" if self.count<8 else "b"))

        self.x_list.append(event.xdata)
        self.y_list.append(event.ydata)
        
        line = self.lines[int(self.count/2)]

        xs = self.x_list[len(self.x_list) - 2 + (len(self.x_list) % 2):]
        ys = self.y_list[len(self.x_list) - 2 + (len(self.x_list) % 2):]

        line.set_data(xs, ys)
        self.ax.scatter(xs, ys, c=draw_color)
        line.figure.canvas.draw()
        self.count += 1
