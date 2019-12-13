#! /usr/bin/env python

import os, glob
import numpy as np
from keras.models import load_model
from generate_samples import read_from_directory
from Tkinter import *
from ttk import *
import tkFileDialog

class Dialog(Frame):

    def __init__(self, master):

        self.img_height, self.img_width = 256, 256

        self.frame = Frame.__init__(self, master)

        self.notebook = Notebook(master)

        self.tab1 = Frame(self.notebook, relief=RAISED, borderwidth=1)
        self.tab1.pack(fill=X)

        self.label = Label(self.tab1, text="Select your model to run against sanity test data set!")
        self.label.pack(pady=10)

        self.frame1a = Frame(self.tab1, relief=RAISED, borderwidth=1)
        self.frame1a.pack()

        self.btnFile = Button(self.frame1a, text="File", command=self.file_dialog)
        self.btnFile.pack(padx=10, pady=10, side=LEFT)

        self.textbox = Entry(self.frame1a)
        self.textbox.config(width=68)
        self.textbox.pack(padx=10, pady=10, side=LEFT)
        self.textbox.delete(0, END)
        self.textbox.insert(0, "*.hd5")
        self.textbox.bind("<Return>", self.textbox_carriage_return)

        self.button = Button(self.frame1a, text="OK", command=self.callback)
        self.button.pack(padx=10, pady=10, side=RIGHT)

        self.scroll1 = Scrollbar(self.tab1)
        self.scroll1.pack(side=RIGHT, fill=Y)

        self.list = Listbox(self.tab1)    # self, selectmode=EXTENDED
        self.list.config(height=38, width=96, yscrollcommand=self.scroll1.set)
        self.list.pack(expand=1)
        self.list.bind("<Double-Button-1>", self.double_clicked_item)
        self.scroll1.config(command=self.list.yview)

        self.tab2 = Frame(self.notebook)
        self.tab2.pack(fill=X)

        self.lblTopLabel = Label(self.tab2)
        self.lblTopLabel.grid(row=0, pady=10, columnspan=4)

        self.lblLabels = Label(self.tab2, text="Labels").grid(row=1)
        self.lblPreds = Label(self.tab2, text="Predictions").grid(row=2)

        self.txtLabels = Text(self.tab2)
        self.txtLabels.grid(row=1, column=1, columnspan=3, pady=16)
        self.txtLabels.config(width=80, height=18)

        self.txtPreds = Text(self.tab2)
        self.txtPreds.grid(row=2, column=1, columnspan=3)
        self.txtPreds.config(width=80, height=18)

        self.lblMean = Label(self.tab2, text="Mean")
        self.lblMean.grid(row=3, sticky=W)

        self.txtMean = Entry(self.tab2)
        self.txtMean.grid(row=3, column=1, sticky=W)

        self.lblStd = Label(self.tab2, text="Std Deviation")
        self.lblStd.grid(row=3, column=2, sticky=E)

        self.txtStd = Entry(self.tab2)
        self.txtStd.grid(row=3, column=3, sticky=E)

        self.notebook.add(self.tab1, text="Select Model")
        self.notebook.add(self.tab2, text="Validation Results")
        self.notebook.pack()

        self.current = None
        self.model_file = None

        self.update_list(self.textbox.get())

    def file_dialog(self):
        my_file = tkFileDialog.askopenfile(initialdir = "./",title = "Select file",filetypes = (("model files","*.hd5"),("all files","*.*")))
        self.run_model(os.path.abspath(my_file.name))
#        self.update_list(my_file)

    def double_clicked_item(self, event):
        self.model_file = self.list.get(event.widget.curselection())
        print(self.model_file)
        self.notebook.select(self.tab2)
        self.lblTopLabel.config(text="Running validaiton for model " + self.model_file)
        self.run_model(self.model_file)

    def callback(self):
        self.list.delete(0, END) # clear
        self.update_list(self.textbox.get())

    def update_list(self, pattern):
        x = glob.glob(pattern)
        self.list.delete(0, END)
        for item in x:
            self.list.insert(END, item.split("/")[-1])      # show only the file name

    def textbox_carriage_return(self, event):
        if event.widget == self.textbox:
            self.update_list(self.textbox.get())


    def run_model(self, file):
        model = load_model(file)
        print("Using model ..." + file)

        data, labels = read_from_directory("./sanity/*.png", self.img_height, self.img_width)

        print("LABELS:")
        print(labels)
        preds = model.predict(data)
        print("PREDICTING:")
        print(preds)

        labels = np.array(labels)

        self.txtLabels.delete(1.0, END)
        self.txtPreds.delete(1.0, END)
        self.txtLabels.insert(END, labels)
        self.txtPreds.insert(END, preds)

        diff = preds.flatten() - labels.flatten()
        percentDiff = (diff / labels.flatten()) * 100
        absPercentDiff = np.abs(percentDiff)

        # compute the mean and standard deviation of the absolute percentage difference
        mean = np.mean(absPercentDiff)
        std = np.std(absPercentDiff)

        self.txtMean.delete(0, END)
        self.txtStd.delete(0, END)
        self.txtMean.insert(END, mean)
        self.txtStd.insert(END, std)

        # finally, show some statistics on our model
        print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))

root = Tk()
root.geometry("800x800")
root.title("Super cool model run thingy")

app = Dialog(root)

root.mainloop()



