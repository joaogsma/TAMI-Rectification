import matplotlib.pyplot as plt

class LineBuilder:
    def __init__(self, ax, line1, line2, line3, line4):
        self.ax = ax
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3
        self.line4 = line4

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

    def __call__(self, event):
        if event.inaxes!=self.line1.axes: return
        if (self.count <= 1):
            self.xs1.append(event.xdata)
            self.ys1.append(event.ydata)
            self.line1.set_data(self.xs1, self.ys1)
            self.ax.scatter(self.xs1, self.ys1, c='r')
            self.line1.figure.canvas.draw()
            self.count = self.count + 1
        elif (2 <= self.count <= 3):
            self.xs2.append(event.xdata)
            self.ys2.append(event.ydata)
            self.line2.set_data(self.xs2, self.ys2)
            self.ax.scatter(self.xs2, self.ys2, c='r')
            self.line2.figure.canvas.draw()
            self.count = self.count + 1
        elif (4 <= self.count <= 5):
            self.xs3.append(event.xdata)
            self.ys3.append(event.ydata)
            self.line3.set_data(self.xs3, self.ys3)
            self.ax.scatter(self.xs3, self.ys3, c='g')
            self.line3.figure.canvas.draw()
            self.count = self.count + 1
        elif (6 <= self.count <= 7):
            self.xs4.append(event.xdata)
            self.ys4.append(event.ydata)
            self.line4.set_data(self.xs4, self.ys4)
            self.ax.scatter(self.xs4, self.ys4, c='g')
            self.line4.figure.canvas.draw()
            self.count = self.count + 1
