#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.22
#  in conjunction with Tcl version 8.6
#    May 16, 2019 10:32:22 PM +07  platform: Windows NT

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import UI_support
import tkinter.filedialog as fdialog
from tkinter import *
import warnings
warnings.filterwarnings('ignore')
from load_data import load_dataset
from preprocessing import praproses
from train_test_label import fold
from BOWTFIDF import BagOfWords
from d2v import d2v
from logreg import logreg
from svm import SVM
from mnb import MNB
import io
from contextlib import redirect_stdout
from threading import Thread
import numpy as np
from plot_figure import plot_figure
from Save_CSV import save_to_csv

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = MainFrame (root)
    UI_support.init(root, top)
    root.mainloop()

w = None
def create_MainFrame(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = MainFrame (w)
    UI_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_MainFrame():
    global w
    w.destroy()
    w = None

class MainFrame:

    def fileopen(self):
        global v1, clean_review, train, test
        filename = fdialog.askopenfilename(filetypes=(("TSV Files", "*.tsv"), ("All Files", "*.*")))
        name = filename
        path = StringVar()
        path.set(name)
        self.Path_Field.configure(textvariable=path)
        try:
            file = io.StringIO()
            with redirect_stdout(file):
                obj_load = load_dataset(name)
                df = obj_load.load()
                obj_review = praproses(df['review'])
                df['review'] = obj_review.cleaning()
            output = file.getvalue()

            obj_split = fold(df)
            train, test = obj_split.train_test()

            self.Scrolledtext1.configure(state='normal')
            self.Scrolledtext1.insert(END, output)
            self.Scrolledtext1.configure(state='disabled')

            v1 = IntVar()
            self.BOW.configure(state='normal', variable=v1, value=1, command=self.vsm)
            self.D2V.configure(state='normal', variable=v1, value=2, command=self.vsm)
        except:
            self.Scrolledtext1.configure(state='normal')
            self.Scrolledtext1.insert(END, "Tidak bisa membaca file yang dimuat.\n")
            self.Scrolledtext1.configure(state='disabled')

    def vsm(self):
        self.MulaiVSM.configure(state='normal', command=lambda:self.start_submit_thread(self.vsm_start))

    def vsm_start(self):
        global bow_tfidf_train, bow_tfidf_test, d2v_train, d2v_test
        self.MulaiVSM.configure(state='disabled')
        self.Hitung.configure(state='disabled')
        self.MulaiKlasifikasi.configure(state='disabled')
        self.MNB.configure(state='disabled')
        self.SVM.configure(state='disabled')
        self.LogReg.configure(state='disabled')
        self.Scrolledtext1.configure(state='normal')
        file = io.StringIO()
        with redirect_stdout(file):
            if v1.get() == 1:
                obj_bow = BagOfWords(train, test)
                bow_tfidf_train, bow_tfidf_test = obj_bow.bow_tfidf()
            elif v1.get() == 2:
                d2v_train = []
                d2v_test = []
                k = 1
                for i in range(len(train)):
                    obj_d2v = d2v(train[i], test[i])
                    d2vtrain, d2vtest = obj_d2v.d2v(k)
                    k = k + 1
                    d2v_train.append(d2vtrain)
                    d2v_test.append(d2vtest)
        output = file.getvalue()
        self.Scrolledtext1.insert(END, output)
        self.Scrolledtext1.configure(state='disabled')
        self.classification_opt()
        self.MulaiVSM.configure(state='normal')

    def classification_opt(self):
        global v2
        v2 = IntVar()
        self.MNB.configure(state='normal', variable=v2, value=1, command=self.classification)
        self.SVM.configure(state='normal', variable=v2, value=2, command=self.classification)
        self.LogReg.configure(state='normal', variable=v2, value=3, command=self.classification)

    def classification(self):
        self.MulaiKlasifikasi.configure(state='normal',
                                        command=lambda: self.start_submit_thread(self.classification_start))

    def classification_start(self):
        try:
            self.Hitung.configure(state='disabled')
            self.MulaiKlasifikasi.configure(state='disabled')
            self.Scrolledtext1.configure(state='normal')
            file = io.StringIO()
            with redirect_stdout(file):
                if v2.get() == 3:
                    if v1.get() == 1:
                        obj_LR = logreg(bow_tfidf_train, train, bow_tfidf_test, test)
                        roc_name = "ROC BagOfWords Logistic Regression"
                        csv_name = "BagOfWords Logistic Regression"
                    elif v1.get() == 2:
                        obj_LR = logreg(d2v_train, train, d2v_test, test)
                        roc_name = "ROC Doc2Vec Logistic Regression"
                        csv_name = "Doc2Vec Logistic Regression"
                    pred_res = obj_LR.logreg()
                elif v2.get() == 2:
                    if v1.get() == 1:
                        obj_SVM = SVM(bow_tfidf_train, train, bow_tfidf_test, test)
                        roc_name = "ROC BagOfWords Support Vector Machine"
                        csv_name = "BagOfWords Support Vector Machine"
                    elif v1.get() == 2:
                        obj_SVM = SVM(d2v_train, train, d2v_test, test)
                        roc_name = "ROC Doc2Vec Support Vector Machine"
                        csv_name = "Doc2Vec Support Vector Machine"
                    pred_res = obj_SVM.svm()
                elif v2.get() == 1:
                    if v1.get() == 1:
                        obj_MNB = MNB(bow_tfidf_train, train, bow_tfidf_test, test)
                        roc_name = "ROC BagOfWords Multinomial Naive Bayes"
                        csv_name = "BagOfWords Multinomial Naive Bayes"
                    elif v1.get() == 2:
                        obj_MNB = MNB(d2v_train, train, d2v_test, test)
                        roc_name = "ROC Doc2Vec Multinomial Naive Bayes"
                        csv_name = "Doc2Vec Multinomial Naive Bayes"
                    pred_res = obj_MNB.mnb()
                obj_csv = save_to_csv(test)
                obj_csv.save(pred_res, csv_name)
            output = file.getvalue()
            self.Scrolledtext1.insert(END, output)
            self.Scrolledtext1.configure(state='disabled')
            self.Hitung.configure(state='normal',
                                  command=lambda: self.plot(pred_res, roc_name))
            self.MulaiKlasifikasi.configure(state='normal')
        except NameError:
            self.MulaiKlasifikasi.configure(state='normal')
            self.Scrolledtext1.configure(state='normal')
            self.Scrolledtext1.insert(END, "Lakukan pembentukan VSM terlebih dahulu!\n")
            self.Scrolledtext1.configure(state='disabled')
        except ValueError:
            self.MulaiKlasifikasi.configure(state='normal')
            self.Scrolledtext1.configure(state='normal')
            self.Scrolledtext1.insert(END, "Masukkan nilai learning rate dan jumlah iterasi. Bilangan yang diinput hanya berupa angka!\n")
            self.Scrolledtext1.configure(state='disabled')
        except AssertionError:
            self.MulaiKlasifikasi.configure(state='normal')
            self.Scrolledtext1.configure(state='normal')
            self.Scrolledtext1.insert(END, "Range yang diterima: 0 < learning rate <= 1 dan jumlah iterasi > 0!\n")
            self.Scrolledtext1.configure(state='disabled')
        except AttributeError:
            self.MulaiKlasifikasi.configure(state='normal')
            self.Scrolledtext1.configure(state='normal')
            self.Scrolledtext1.insert(END, "Fitur tidak bisa bernilai negatif!\n")
            self.Scrolledtext1.configure(state='disabled')

    def plot(self, pred_res, roc_name):
        obj_plot_fig = plot_figure(test, pred_res)
        obj_plot_fig.plot_fig(roc_name)

    def start_submit_thread(self, event):
        global submit_thread
        submit_thread = Thread(target=event)
        submit_thread.daemon = True
        self.TProgressbar1.start()
        submit_thread.start()
        root.after(25, self.check_submit_thread)

    def check_submit_thread(self):
        if submit_thread.is_alive():
            root.after(25, self.check_submit_thread)
        else:
            self.TProgressbar1.stop()

    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font9 = "-family {Roboto} -size 9"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font=font9)
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("757x480+300+108")
        top.title("VSM")
        top.configure(background="#ededed")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.MuatDokumen = tk.Button(top)
        self.MuatDokumen.place(relx=0.042, rely=0.086, height=23, width=114)
        self.MuatDokumen.configure(activebackground="#ececec")
        self.MuatDokumen.configure(activeforeground="#000000")
        self.MuatDokumen.configure(background="#2196f3")
        self.MuatDokumen.configure(command=lambda:self.start_submit_thread(self.fileopen))
        self.MuatDokumen.configure(disabledforeground="#a3a3a3")
        self.MuatDokumen.configure(font="-family {Roboto Medium} -size 9")
        self.MuatDokumen.configure(foreground="#ffffff")
        self.MuatDokumen.configure(highlightbackground="#d9d9d9")
        self.MuatDokumen.configure(highlightcolor="black")
        self.MuatDokumen.configure(pady="0")
        self.MuatDokumen.configure(text='''Muat Dokumen''')

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.042, rely=0.193, relheight=0.680, relwidth=0.29)

        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#ffffff")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")
        self.Frame1.configure(width=205)

        self.Label1 = tk.Label(self.Frame1)
        self.Label1.place(relx=0.024, rely=0.0, height=21, width=111)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(activeforeground="black")
        self.Label1.configure(background="#ffffff")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Roboto Medium} -size 9")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Vector Space Model''')

        self.Frame2 = tk.Frame(self.Frame1)
        self.Frame2.place(relx=0.050, rely=0.068, relheight=0.32
                , relwidth=0.902)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#ededed")
        self.Frame2.configure(highlightbackground="#d9d9d9")
        self.Frame2.configure(highlightcolor="black")
        self.Frame2.configure(width=185)

        self.BOW = tk.Radiobutton(self.Frame2)
        self.BOW.place(relx=0.027, rely=0.048, relheight=0.238, relwidth=0.546)
        self.BOW.configure(activebackground="#ececec")
        self.BOW.configure(activeforeground="#000000")
        self.BOW.configure(background="#ededed")
        self.BOW.configure(disabledforeground="#a3a3a3")
        self.BOW.configure(foreground="#000000")
        self.BOW.configure(highlightbackground="#d9d9d9")
        self.BOW.configure(highlightcolor="black")
        self.BOW.configure(justify='left')
        self.BOW.configure(state='disabled')
        self.BOW.configure(text='''Bag Of Words''')

        self.D2V = tk.Radiobutton(self.Frame2)
        self.D2V.place(relx=0.020, rely=0.35, relheight=0.209, relwidth=0.411)
        self.D2V.configure(activebackground="#ececec")
        self.D2V.configure(activeforeground="#000000")
        self.D2V.configure(background="#ededed")
        self.D2V.configure(disabledforeground="#a3a3a3")
        self.D2V.configure(font="-family {Roboto} -size 9")
        self.D2V.configure(foreground="#000000")
        self.D2V.configure(highlightbackground="#d9d9d9")
        self.D2V.configure(highlightcolor="black")
        self.D2V.configure(justify='left')
        self.D2V.configure(state='disabled')
        self.D2V.configure(text='''Doc2Vec''')

        self.MulaiVSM = tk.Button(self.Frame2)
        self.MulaiVSM.place(relx=0.050, rely=0.65, height=23, width=149)
        self.MulaiVSM.configure(activebackground="#ececec")
        self.MulaiVSM.configure(activeforeground="#000000")
        self.MulaiVSM.configure(background="#2196f3")
        self.MulaiVSM.configure(disabledforeground="#a3a3a3")
        self.MulaiVSM.configure(font="-family {Roboto Medium} -size 9")
        self.MulaiVSM.configure(foreground="#ffffff")
        self.MulaiVSM.configure(highlightbackground="#d9d9d9")
        self.MulaiVSM.configure(highlightcolor="black")
        self.MulaiVSM.configure(pady="0")
        self.MulaiVSM.configure(state='disabled')
        self.MulaiVSM.configure(text='''Mulai Pembentukan VSM''')

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.024, rely=0.4, height=20, width=106)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#ffffff")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Roboto Medium} -size 9")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Metode Klasifikasi''')

        self.Frame3 = tk.Frame(self.Frame1)
        self.Frame3.place(relx=0.049, rely=0.47, relheight=0.5
                , relwidth=0.902)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#ededed")
        self.Frame3.configure(highlightbackground="#d9d9d9")
        self.Frame3.configure(highlightcolor="black")
        self.Frame3.configure(width=185)

        self.MNB = tk.Radiobutton(self.Frame3)
        self.MNB.place(relx=0.027, rely=0.04, relheight=0.192, relwidth=0.876)
        self.MNB.configure(activebackground="#ececec")
        self.MNB.configure(activeforeground="#000000")
        self.MNB.configure(background="#ededed")
        self.MNB.configure(disabledforeground="#a3a3a3")
        self.MNB.configure(font="-family {Roboto} -size 9")
        self.MNB.configure(foreground="#000000")
        self.MNB.configure(highlightbackground="#d9d9d9")
        self.MNB.configure(highlightcolor="black")
        self.MNB.configure(justify='left')
        self.MNB.configure(state='disabled')
        self.MNB.configure(text='''Multinomial Naive Bayes''')

        self.LogReg = tk.Radiobutton(self.Frame3)
        self.LogReg.place(relx=0.027, rely=0.20, relheight=0.192, relwidth=0.714)
        self.LogReg.configure(activebackground="#ececec")
        self.LogReg.configure(activeforeground="#000000")
        self.LogReg.configure(background="#ededed")
        self.LogReg.configure(disabledforeground="#a3a3a3")
        self.LogReg.configure(font="-family {Roboto} -size 9")
        self.LogReg.configure(foreground="#000000")
        self.LogReg.configure(highlightbackground="#d9d9d9")
        self.LogReg.configure(highlightcolor="black")
        self.LogReg.configure(justify='left')
        self.LogReg.configure(state='disabled')
        self.LogReg.configure(text='''Logistic Regression''')

        self.SVM = tk.Radiobutton(self.Frame3)
        self.SVM.place(relx=0.027, rely=0.37, relheight=0.192, relwidth=0.854)
        self.SVM.configure(activebackground="#ececec")
        self.SVM.configure(activeforeground="#000000")
        self.SVM.configure(background="#ededed")
        self.SVM.configure(disabledforeground="#a3a3a3")
        self.SVM.configure(font="-family {Roboto} -size 9")
        self.SVM.configure(foreground="#000000")
        self.SVM.configure(highlightbackground="#d9d9d9")
        self.SVM.configure(highlightcolor="black")
        self.SVM.configure(justify='left')
        self.SVM.configure(state='disabled')
        self.SVM.configure(text='''Support Vector Machine''')

        self.MulaiKlasifikasi = tk.Button(top)
        self.MulaiKlasifikasi.place(relx=0.042, rely=0.89, height=23, width=100)
        self.MulaiKlasifikasi.configure(activebackground="#ececec")
        self.MulaiKlasifikasi.configure(activeforeground="#000000")
        self.MulaiKlasifikasi.configure(background="#2196f3")
        self.MulaiKlasifikasi.configure(disabledforeground="#a3a3a3")
        self.MulaiKlasifikasi.configure(font="-family {Roboto Medium} -size 9")
        self.MulaiKlasifikasi.configure(foreground="#ffffff")
        self.MulaiKlasifikasi.configure(highlightbackground="#d9d9d9")
        self.MulaiKlasifikasi.configure(highlightcolor="black")
        self.MulaiKlasifikasi.configure(pady="0")
        self.MulaiKlasifikasi.configure(state='disabled')
        self.MulaiKlasifikasi.configure(text='''Mulai Klasifikasi''')

        self.Hitung = tk.Button(top)
        self.Hitung.place(relx=0.202, rely=0.89, height=23, width=97)
        self.Hitung.configure(activebackground="#ececec")
        self.Hitung.configure(activeforeground="#000000")
        self.Hitung.configure(background="#2196f3")
        self.Hitung.configure(disabledforeground="#a3a3a3")
        self.Hitung.configure(font="-family {Roboto Medium} -size 9")
        self.Hitung.configure(foreground="#ffffff")
        self.Hitung.configure(highlightbackground="#d9d9d9")
        self.Hitung.configure(highlightcolor="black")
        self.Hitung.configure(pady="0")
        self.Hitung.configure(state='disabled')
        self.Hitung.configure(text='''Hitung AUROC''')

        self.Path_Field = tk.Entry(top)
        self.Path_Field.place(relx=0.24, rely=0.086,height=25, relwidth=0.713)
        self.Path_Field.configure(background="white")
        self.Path_Field.configure(state='readonly')
        self.Path_Field.configure(disabledforeground="#a3a3a3")
        self.Path_Field.configure(font="TkFixedFont")
        self.Path_Field.configure(foreground="#000000")
        self.Path_Field.configure(highlightbackground="#d9d9d9")
        self.Path_Field.configure(highlightcolor="black")
        self.Path_Field.configure(insertbackground="black")
        self.Path_Field.configure(selectbackground="#c4c4c4")
        self.Path_Field.configure(selectforeground="black")
        self.Path_Field.configure(takefocus="0")

        self.Scrolledtext1 = ScrolledText(top)
        self.Scrolledtext1.place(relx=0.354, rely=0.26, relheight=0.68, relwidth = 0.595)
        self.Scrolledtext1.configure(background="white")
        self.Scrolledtext1.configure(font="TkTextFont")
        self.Scrolledtext1.configure(foreground="black")
        self.Scrolledtext1.configure(highlightbackground="#d9d9d9")
        self.Scrolledtext1.configure(highlightcolor="black")
        self.Scrolledtext1.configure(insertbackground="black")
        self.Scrolledtext1.configure(insertborderwidth="3")
        self.Scrolledtext1.configure(state='disabled')
        self.Scrolledtext1.configure(selectbackground="#c4c4c4")
        self.Scrolledtext1.configure(selectforeground="black")
        self.Scrolledtext1.configure(takefocus="0")
        self.Scrolledtext1.configure(width=10)
        self.Scrolledtext1.configure(wrap="none")

        self.TProgressbar1 = ttk.Progressbar(top, length=420, mode="indeterminate")
        self.TProgressbar1.place(relx=0.354, rely=0.193, relwidth=0.594, relheight = 0.0, height = 20)


# The following code is added to facilitate the Scrolled widgets you specified.
class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''

    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)

        #self.configure(yscrollcommand=_autoscroll(vsb),
        #    xscrollcommand=_autoscroll(hsb))
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))

        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')

        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)

        # Copy geometry methods of master  (taken from ScrolledText.py)
        if py3:
            methods = tk.Pack.__dict__.keys() | tk.Grid.__dict__.keys() \
                  | tk.Place.__dict__.keys()
        else:
            methods = tk.Pack.__dict__.keys() + tk.Grid.__dict__.keys() \
                  + tk.Place.__dict__.keys()

        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

    def __str__(self):
        return str(self.master)

def _create_container(func):
    '''Creates a ttk Frame with a given master, and use this new frame to
    place the scrollbars and the widget.'''
    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
        container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
        return func(cls, container, **kw)
    return wrapped

class ScrolledText(AutoScroll, tk.Text):
    '''A standard Tkinter Text widget with scrollbars that will
    automatically show/hide as needed.'''
    @_create_container
    def __init__(self, master, **kw):
        tk.Text.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)

import platform
def _bound_to_mousewheel(event, widget):
    child = widget.winfo_children()[0]
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        child.bind_all('<MouseWheel>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-MouseWheel>', lambda e: _on_shiftmouse(e, child))
    else:
        child.bind_all('<Button-4>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Button-5>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-Button-4>', lambda e: _on_shiftmouse(e, child))
        child.bind_all('<Shift-Button-5>', lambda e: _on_shiftmouse(e, child))

def _unbound_to_mousewheel(event, widget):
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        widget.unbind_all('<MouseWheel>')
        widget.unbind_all('<Shift-MouseWheel>')
    else:
        widget.unbind_all('<Button-4>')
        widget.unbind_all('<Button-5>')
        widget.unbind_all('<Shift-Button-4>')
        widget.unbind_all('<Shift-Button-5>')

def _on_mousewheel(event, widget):
    if platform.system() == 'Windows':
        widget.yview_scroll(-1*int(event.delta/120),'units')
    elif platform.system() == 'Darwin':
        widget.yview_scroll(-1*int(event.delta),'units')
    else:
        if event.num == 4:
            widget.yview_scroll(-1, 'units')
        elif event.num == 5:
            widget.yview_scroll(1, 'units')

def _on_shiftmouse(event, widget):
    if platform.system() == 'Windows':
        widget.xview_scroll(-1*int(event.delta/120), 'units')
    elif platform.system() == 'Darwin':
        widget.xview_scroll(-1*int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.xview_scroll(-1, 'units')
        elif event.num == 5:
            widget.xview_scroll(1, 'units')

if __name__ == '__main__':
    vp_start_gui()




