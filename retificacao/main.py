from tkinter import *
from subprocess import call


class Application(Frame):
    def directMetricRectification(self):
        #print ("hi there, everyone!")
        call(["python3.5", "directMetric_rectification.py"])

    def stratified_metric_rectification(self):
        call(["python3.5", "stratified_metric_rectification.py"])

    def algebricCrossRatio_rectification(self):
        call(["python3.5", "algebricCrossRatio_rectification.py"])

    def geometricCrossRatio_rectification(self):
        call(["python3.5", "geometricCrossRatio_rectification.py"])

    def createWidgets(self):

        self.directMetricRectification_button = Button(self)
        self.directMetricRectification_button["text"] = "DirectMetric Rectification"
        self.directMetricRectification_button["command"] = self.directMetricRectification
        self.directMetricRectification_button.pack(side="top",fill='both', expand=True)

        self.stratifiedMetricRectification_button = Button(self)
        self.stratifiedMetricRectification_button["text"] = "StratifiedMetric Rectification"
        self.stratifiedMetricRectification_button["command"] = self.stratified_metric_rectification
        self.stratifiedMetricRectification_button.pack(side="top",fill='both', expand=True)

        self.algebricCrossRatioRectification_button = Button(self)
        self.algebricCrossRatioRectification_button["text"] = "AlgebricCrossRatio Rectification"
        self.algebricCrossRatioRectification_button["command"] = self.algebricCrossRatio_rectification
        self.algebricCrossRatioRectification_button.pack(side="top",fill='both', expand=True)

        self.geometricCrossRatioRectification_button = Button(self)
        self.geometricCrossRatioRectification_button["text"] = "GeometricCrossRatio Rectification"
        self.geometricCrossRatioRectification_button["command"] = self.geometricCrossRatio_rectification
        self.geometricCrossRatioRectification_button.pack(side="top",fill='both', expand=True)

        self.Quit = Button(self)
        self.Quit["text"] = "Quit"
        self.Quit["fg"]   = "red"
        self.Quit["command"] =  self.quit
        self.Quit.pack(side="top",fill='both', expand=True)

        w = Label(root, text=" ")
        w.pack( padx = 100, pady = 100)

        w = Label(root, text="Developers:")
        w.pack(side="left", padx = 2, pady = 2)

        w = Label(root, text=" ")
        w.pack(side="left",pady = 10)

        w = Label(root, text="Heitor Rapela Medeiros")
        w["fg"]   = "blue"
        w.pack(side="left",pady = 10)

        w = Label(root, text="  | ")
        #w["fg"]   = "blue"
        w.pack(side="left",pady = 10)

        w = Label(root, text=" ")
        w.pack(side="left",pady = 10)

        w = Label(root, text="João Gabriel Abreu")
        w["fg"]   = "blue"
        w.pack(side="left",pady = 10)

        w = Label(root, text="  | ")
        #w["fg"]   = "blue"
        w.pack(side="left",pady = 10)

        w = Label(root, text=" ")
        w.pack(side="left",pady = 10)

        w = Label(root, text="Luiz Gustavo da Rocha Charamba")
        w["fg"]   = "blue"
        w.pack(side="left",pady = 10)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()


root = Tk()
root.title("Retificação - Tópicos Avançados em Mídia e Interação: TAMI 2016.1")
root.geometry('{}x{}'.format(640, 480))

w = Label(root, text="Rectification Methods: ")
w.pack(side="top",padx = 50, pady = 20)
#w.pack( padx = 30, pady = 30)

app = Application(master=root)
app.mainloop()
root.destroy()

