<<<<<<< HEAD
# For image aquisition
from tkinter import *



=======
from tkinter import *


>>>>>>> e554ca9e00656359d7389297c5a59698f6e1d0a6
class UserInputRatio():
    def __init__(self, text):
        self.Master=Tk()
        self.Entry=Entry(self.Master)
        self.Master.wm_title(text)
        self.Entry.pack()

<<<<<<< HEAD
        self.Button=Button(self.Master,text=text,command=self.Return)
        self.Button.pack()

        self.Master.mainloop()

    def Return(self):
        self.TempVar=self.Entry.get()
        self.Entry.quit()
=======
        self.Button=Button(self.Master,text=text ,command=self.Return)
        self.Button.pack()            

        self.Master.mainloop()
    
    def Return(self):
        self.TempVar=self.Entry.get() 
        self.Entry.quit()

    #def press(event):
    #print('press', event.key)
    #    if event.key == 'enter':
     #       plt.close()
>>>>>>> e554ca9e00656359d7389297c5a59698f6e1d0a6
