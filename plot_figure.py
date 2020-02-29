import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from aucroc import aucroc

class plot_figure:

    def __init__(self, test_label, predict_result):
        self.test_label = test_label
        self.predict_result = predict_result

    def plot_fig(self, roc_name):
        self.root = tkinter.Tk()
        self.root.wm_title("Embedding in Tk")

        obj_roc = aucroc(self.test_label, self.predict_result)
        fig, aucs = obj_roc.plot_auroc(roc_name)

        canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self.root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        button = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        button.pack(side=tkinter.BOTTOM)

        tkinter.mainloop()


    def _quit(self):
        self.root.quit()  # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate