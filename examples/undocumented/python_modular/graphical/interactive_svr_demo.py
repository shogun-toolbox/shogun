"""
Shogun demo, based on PyQT Demo by Eli Bendersky

Christian Widmer
Soeren Sonnenburg
License: GPLv3
"""
import numpy
import sys, os, csv
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
from matplotlib.colorbar import make_axes, Colorbar
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from modshogun import *

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
        self.axes.plot(self.data.x1, self.data.x2, 'bo')
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
        self.data.add_example(event.xdata, event.ydata)
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

    def train_svm(self):

        width = float(self.sigma.text())
        degree = int(self.degree.text())

        self.axes.clear()
        self.axes.grid(True)
        self.axes.plot(self.data.x1, self.data.x2, 'bo')

        # train svm
        labels = self.data.get_labels()
        print type(labels)
        lab = RegressionLabels(labels)
        features = self.data.get_examples()
        train = RealFeatures(features)

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

        cost = float(self.cost.text())
        tubeeps = float(self.tubeeps.text())

        print "cost", cost
        svm = LibSVR(cost, tubeeps, gk, lab)
        svm.train()
        svm.set_epsilon(1e-2)

        x=numpy.linspace(-5.0,5.0,100)
        y=svm.apply(RealFeatures(numpy.array([x]))).get_labels()

        self.axes.plot(x,y,'r-')

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


        cost_label = QLabel('C')
        #self.cost = QSpinBox()#QLineEdit()
        self.cost = QLineEdit()
        self.cost.setText("1.0")
        #self.cost.setMinimum(1)
        spin_label2 = QLabel('tube')
        self.tubeeps = QLineEdit()
        self.tubeeps.setText("0.1")
        spin_label3 = QLabel('sigma')
        self.sigma = QLineEdit()
        self.sigma.setText("1.2")
        #self.sigma.setMinimum(1)
        spin_label4 = QLabel('d')
        self.degree = QLineEdit()
        self.degree.setText("2")
        #self.sigma.setMinimum(1)

        spins_hbox = QHBoxLayout()
        spins_hbox.addWidget(cost_label)
        spins_hbox.addWidget(self.cost)
        spins_hbox.addWidget(spin_label2)
        spins_hbox.addWidget(self.tubeeps)
        spins_hbox.addWidget(spin_label3)
        spins_hbox.addWidget(self.sigma)
        spins_hbox.addWidget(spin_label4)
        spins_hbox.addWidget(self.degree)
        spins_hbox.addStretch(1)

        self.legend_cb = QCheckBox("Show Support Vectors")
        self.legend_cb.setChecked(False)

        self.show_button = QPushButton("&Train SVR")
        self.connect(self.show_button, SIGNAL('clicked()'), self.train_svm)

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
        self.x1 = []
        self.x2 = []


    def get_stats(self):

        num = len(self.x1)

        str_num = "num examples: %i" % num

        return (str_num, str_num)


    def get_labels(self):
        return numpy.array(self.x2, dtype=numpy.float64)


    def get_examples(self):
        num = len(self.x1)
        examples = numpy.zeros((1,num))

        for i in xrange(num):
            examples[0,i] = self.x1[i]
        return examples


    def add_example(self, x1, x2):
        self.x1.append(x1)
        self.x2.append(x2)

    def load_from_file(self, filename=None):
        self.data = {}
        self.names = []

        if filename:
            for line in csv.reader(open(filename, 'rb')):
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
