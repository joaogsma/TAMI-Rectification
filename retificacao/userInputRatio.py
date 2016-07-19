from tkinter import *


class UserInputRatio():
    def __init__(self, text):
        self.Master=Tk()
        self.Entry=Entry(self.Master)
        self.Master.wm_title(text)
        self.Entry.pack()

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