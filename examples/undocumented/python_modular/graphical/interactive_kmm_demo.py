#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2013 Cameron Lai, based on interactive_svm_demo by Christian
# Widmer which itself is based on PyQT Demo by Eli Bendersky
#
"""
Shogun KMM demo based on interactive SVM demo by Christian \
Widmer and Soeren Sonnenburg which itself is based on PyQT Demo by Eli Bendersky

Cameron Lai
License: GPLv3
"""
import numpy
import sys, os, csv
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
from matplotlib import mpl
from matplotlib.colorbar import make_axes, Colorbar
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from modshogun import *
from modshogun import KernelMeanMatching
from modshogun import *
from modshogun import Math
import util

class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.setWindowTitle('SHOGUN interactive demo')

        self.data = DataHolder()
        self.series_list_model = QStandardItemModel()

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        self.on_show()

    def load_file(self, filename=None):
        filename = QFileDialog.getOpenFileName(self,
            'Open a data file', '.', 'CSV files (*.csv);;All Files (*.*)')

        if filename:
            self.data.load_from_file(filename)
            self.fill_series_list(self.data.series_names())
            self.status_text.setText("Loaded " + filename)

    def on_show(self):
        self.axes.clear()
        self.axes.grid(True)
        self.axes.plot(self.data.x1_test, self.data.x2_test, 'ro')
        self.axes.plot(self.data.x1_train, self.data.x2_train, 'bo')
        self.axes.set_xlim((-5,5))
        self.axes.set_ylim((-5,5))
        self.canvas.draw()
        self.fill_series_list(self.data.get_stats())

    def on_about(self):
        msg = __doc__
        QMessageBox.about(self, "About the demo", msg.strip())

    def fill_series_list(self, names):
        self.series_list_model.clear()

        for name in names:
            item = QStandardItem(name)
            item.setCheckState(Qt.Unchecked)
            item.setCheckable(False)
            self.series_list_model.appendRow(item)

    def onclick(self, event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
        if event.button==1:
            label = 1.0
        else:
            label = -1.0
        self.data.add_example(event.xdata, event.ydata, label)
        self.on_show()


    def clear(self):
        self.data.clear()
        self.on_show()

    def enable_widgets(self):
        kernel_name = self.kernel_combo.currentText()
        if kernel_name == "LinearKernel":
            self.sigma.setDisabled(True)
            self.degree.setDisabled(True)
        elif kernel_name == "PolynomialKernel":
            self.sigma.setDisabled(True)
            self.degree.setEnabled(True)
        elif kernel_name == "GaussianKernel":
            self.sigma.setEnabled(True)
            self.degree.setDisabled(True)

    def train_kmm(self):

        width = float(self.sigma.text())
        degree = int(self.degree.text())

        self.axes.clear()
        self.axes.grid(True)
        self.axes.plot(self.data.x1_test, self.data.x2_test, 'ro')
        self.axes.plot(self.data.x1_train, self.data.x2_train, 'bo')

        # train kmm
        labels = self.data.get_labels()
        lab = BinaryLabels(labels)
        features = self.data.get_examples()
        train = RealFeatures(features)

        nTrain=len(self.data.x1_train);
        nTest=len(self.data.x1_test);
        trainI=numpy.array(range(nTrain), dtype=numpy.int32)
        testI=numpy.array(range(nTrain,nTest+nTrain),dtype=numpy.int32)


        kernel_name = self.kernel_combo.currentText()
        print "current kernel is %s" % (kernel_name)

        if kernel_name == "LinearKernel":
            gk = LinearKernel(train, train)
            gk.set_normalizer(IdentityKernelNormalizer())
        elif kernel_name == "PolynomialKernel":
            gk = PolyKernel(train, train, degree, True)
            gk.set_normalizer(IdentityKernelNormalizer())
        elif kernel_name == "GaussianKernel":
            gk = GaussianKernel(train, train, width)

        kmm = KernelMeanMatching(gk, trainI, testI)
        w = kmm.compute_weights()
        print 'Weights'
        print w

        self.axes.clear()
        self.axes.grid(True)
        self.axes.plot(self.data.x1_test, self.data.x2_test, 'ro')
        m_size=numpy.array(w*1000, dtype=numpy.int32)
        self.axes.scatter(self.data.x1_train, self.data.x2_train, s=m_size)
        self.axes.set_xlim((-5,5))
        self.axes.set_ylim((-5,5))
        self.canvas.draw()

    def create_main_frame(self):
        self.main_frame = QWidget()

        plot_frame = QWidget()

        self.dpi = 100
        self.fig = Figure((6.0, 6.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        cid = self.canvas.mpl_connect('button_press_event', self.onclick)
        self.axes = self.fig.add_subplot(111)
        self.cax = None
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        log_label = QLabel("Number of examples:")
        self.series_list_view = QListView()
        self.series_list_view.setModel(self.series_list_model)

        spin_label = QLabel('sigma')
        self.sigma = QLineEdit()
        self.sigma.setText("1.2")
        #self.sigma.setMinimum(1)
        spin_label2 = QLabel('degree')
        self.degree = QLineEdit()
        self.degree.setText("2")
        #self.sigma.setMinimum(1)

        spins_hbox = QHBoxLayout()
        spins_hbox.addWidget(spin_label)
        spins_hbox.addWidget(self.sigma)
        spins_hbox.addWidget(spin_label2)
        spins_hbox.addWidget(self.degree)
        spins_hbox.addStretch(1)

        self.show_button = QPushButton("&Perform KMM")
        self.connect(self.show_button, SIGNAL('clicked()'), self.train_kmm)

        self.clear_button = QPushButton("&Clear")
        self.connect(self.clear_button, SIGNAL('clicked()'), self.clear)


        self.kernel_combo = QComboBox()
        self.kernel_combo.insertItem(-1, "GaussianKernel")
        self.kernel_combo.insertItem(-1, "PolynomialKernel")
        self.kernel_combo.insertItem(-1, "LinearKernel")
        self.kernel_combo.maximumSize = QSize(300, 50)
        self.connect(self.kernel_combo, SIGNAL("currentIndexChanged(QString)"), self.enable_widgets)


        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.canvas)
        #left_vbox.addWidget(self.mpl_toolbar)

        right0_vbox = QVBoxLayout()
        right0_vbox.addWidget(log_label)
        right0_vbox.addWidget(self.series_list_view)
        #right0_vbox.addWidget(self.legend_cb)
        right0_vbox.addStretch(1)

        right2_vbox = QVBoxLayout()
        right2_label = QLabel("Settings")
        right2_vbox.addWidget(right2_label)
        right2_vbox.addWidget(self.show_button)
        right2_vbox.addWidget(self.kernel_combo)
        right2_vbox.addLayout(spins_hbox)
        right2_clearlabel = QLabel("Remove Data")
        right2_vbox.addWidget(right2_clearlabel)

        right2_vbox.addWidget(self.clear_button)


        right2_vbox.addStretch(1)

        right_vbox = QHBoxLayout()
        right_vbox.addLayout(right0_vbox)
        right_vbox.addLayout(right2_vbox)


        hbox = QVBoxLayout()
        hbox.addLayout(left_vbox)
        hbox.addLayout(right_vbox)
        self.main_frame.setLayout(hbox)

        self.setCentralWidget(self.main_frame)
        self.enable_widgets()


    def create_status_bar(self):
        self.status_text = QLabel("")
        self.statusBar().addWidget(self.status_text, 1)

    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        load_action = self.create_action("&Load file",
            shortcut="Ctrl+L", slot=self.load_file, tip="Load a file")
        quit_action = self.create_action("&Quit", slot=self.close,
            shortcut="Ctrl+Q", tip="Close the application")

        self.add_actions(self.file_menu,
            (load_action, None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About",
            shortcut='F1', slot=self.on_about,
            tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(  self, text, slot=None, shortcut=None,
                        icon=None, tip=None, checkable=False,
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action


class DataHolder(object):
    """ Just a thin wrapper over a dictionary that holds integer
        data series. Each series has a name and a list of numbers
        as its data. The length of all series is assumed to be
        the same.

        The series can be read from a CSV file, where each line
        is a separate series. In each series, the first item in
        the line is the name, and the rest are data numbers.
    """
    def __init__(self, filename=None):
        self.clear()
        self.load_from_file(filename)


    def clear(self):
        self.x1_test = []
        self.x2_test = []
        self.x1_train = []
        self.x2_train = []


    def get_stats(self):
        num_train = len(self.x1_train)
        num_test = len(self.x1_test)

        str_train = "num train examples: %i" % num_train
        str_test = "num test examples: %i" % num_test

        return (str_train, str_test)


    def get_labels(self):
        return numpy.array([1]*len(self.x1_test) + [-1]*len(self.x1_train), dtype=numpy.float64)


    def get_examples(self):
        num_pos = len(self.x1_test)
        num_neg = len(self.x1_train)
        examples = numpy.zeros((2,num_pos+num_neg))

        for i in xrange(num_pos):
            examples[0,i] = self.x1_test[i]
            examples[1,i] = self.x2_test[i]

        for i in xrange(num_neg):
            examples[0,i+num_pos] = self.x1_train[i]
            examples[1,i+num_pos] = self.x2_train[i]

        return examples


    def add_example(self, x1, x2, label):
        if label==1:
            self.x1_train.append(x1)
            self.x2_train.append(x2)
        else:
            self.x1_test.append(x1)
            self.x2_test.append(x2)

    def load_from_file(self, filename=None):
        self.data = {}
        self.names = []

        if filename:
            for line in csv.reader(open(filename, 'rb')):
                print line
                self.names.append(line[0])
                self.data[line[0]] = map(int, line[1:])
                self.datalen = len(line[1:])

    def series_names(self):
        """ Names of the data series
        """
        return self.names

    def series_len(self):
        """ Length of a data series
        """
        return self.datalen

    def series_count(self):
        return len(self.data)

    def get_series_data(self, name):
        return self.data[name]


def main():
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()

    #~ dh = DataHolder('qt_mpl_data.csv')
    #~ print dh.data
    #~ print dh.get_series_data('1991 Sales')
    #~ print dh.series_names()
    #~ print dh.series_count()
