#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann, based on interactive_svm_demo by Christian
# Widmer which itself is based on PyQT Demo by Eli Bendersky
#

"""
Shogun Gaussian processes demo based on interactive SVM demo by Christian \
Widmer and Soeren Sonnenburg which itself is based on PyQT Demo by Eli Bendersky

NOT NEARLY FINISHED YET. Just checked in due to the quick need of a demo.

Heiko Strathmann
License: GPLv3
"""
import sys, os, csv
import scipy as SP
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from numpy import *

import matplotlib
from matplotlib import mpl
from matplotlib.colorbar import make_axes, Colorbar
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
import util

class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.setWindowTitle('SHOGUN interactive demo')

        self.series_list_model = QStandardItemModel()

        self.create_main_frame()
        self.create_status_bar()
        self.create_toy_data()
        
        self.on_show()

    def on_show(self):
        self.axes.clear()
        self.axes.plot(self.x, self.y, 'ro')
        self.axes.set_xlim((-5,5))
        self.axes.set_ylim((-5,5))
        self.axes.grid(True)
        self.canvas.draw()
    

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
        0
        #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
        #self.data.add_example(event.xdata, event.ydata)
        #self.on_show()

    def create_toy_data(self):
        #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
        xmin = -5
        xmax = 5
        x = SP.arange(xmin,xmax,(xmax-xmin)/100.0)

        C = 2       #offset

        b = 0

        y  = b*x + C + float(self.sine_amplitude.text())*SP.sin(float(self.sine_freq.text())*x)
        #    dy = b   +     1*SP.cos(x)
        y += float(self.noise_level.text())*random.randn(y.shape[0])

        self.y=y-y.mean()

        self.x= x[:,SP.newaxis]
        self.on_show()
    
    def learn_kernel_width(self):
        root=ModelSelectionParameters();
        c1=ModelSelectionParameters("inference_method", inf);
        root.append_child(c1);

        c2 = ModelSelectionParameters("scale");
        c1.append_child(c2);
        c2.build_values(0.01, 4.0, R_LINEAR);
        c3 = ModelSelectionParameters("likelihood_model", likelihood);
        c1.append_child(c3);

        c4=ModelSelectionParameters("sigma");
        c3.append_child(c4);
        c4.build_values(0.001, 4.0, R_LINEAR);
        c5 =ModelSelectionParameters("kernel", SECF);
        c1.append_child(c5);

        c6 =ModelSelectionParameters("width");
        c5.append_child(c6);
        c6.build_values(0.001, 4.0, R_LINEAR);
        
        crit = GradientCriterion();

        grad=GradientEvaluation(gp, feat_train, labels,
			    crit);

        grad.set_function(inf);

        gp.print_modsel_params();

        root.print_tree();

        grad_search=GradientModelSelection(
			    root, grad);

        grad.set_autolock(0);

        best_combination=grad_search.select_model(1);
    
        self.sigma.setText("1.0")
        self.plot_gp()

    def plot_gp(self):
        feat_train = RealFeatures(self.x.T)
        labels = RegressionLabels(self.y)
        #[x,y]=self.data.get_data()
        #feat_train=RealFeatures(x.T)
        #labels=RegressionLabels(y)
        n_dimensions = 1
        
        #new interface with likelihood parametres being decoupled from the covaraince function
        likelihood = GaussianLikelihood()
        covar_parms = SP.log([2])
        hyperparams = {'covar':covar_parms,'lik':SP.log([1])}
        
        # construct covariance function
        width=float(self.sigma.text())
        SECF = GaussianKernel(feat_train, feat_train,width)
        covar = SECF
        zmean = ZeroMean();
        inf = ExactInferenceMethod(SECF, feat_train, zmean, labels, likelihood);

        # location of unispaced predictions
        x_test = array([linspace(min(self.x),max(self.x), feat_train.get_num_vectors())])
        feat_test=RealFeatures(x_test)
        
        gp = GaussianProcessRegression(inf, feat_train, labels);
        gp.set_return_type(GaussianProcessRegression.GP_RETURN_COV);
        covariance = gp.apply_regression(feat_test);
        gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS);
        predictions = gp.apply_regression();
        
        #print "x_test"
        #print feat_test.get_feature_matrix()
        #print "mean predictions"
        #print predictions.get_labels()
        print "covariances"
        print covariance.get_labels()
        
        self.axes.clear()
        self.axes.grid(True)
        self.axes.set_xlim((-5,5))
        self.axes.set_ylim((-5,5))
        self.axes.hold(True)
        self.axes.plot(feat_test.get_feature_matrix()[0], predictions.get_labels(), 'b-x')
        self.axes.plot(feat_train.get_feature_matrix()[0], labels.get_labels(), 'ro')
        self.axes.plot(feat_test.get_feature_matrix()[0], predictions.get_labels()-3*sqrt(covariance.get_labels()))
        self.axes.plot(feat_test.get_feature_matrix()[0], predictions.get_labels()+3*sqrt(covariance.get_labels()))
        self.axes.hold(False)
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
        
        self.sine_freq = QLineEdit()
        self.sine_freq.setText("1.0")
        self.sine_amplitude = QLineEdit()
        self.sine_amplitude.setText("1.0")
        self.sigma = QLineEdit()
        self.sigma.setText("1.2")
        self.noise_level = QLineEdit()
        self.noise_level.setText("1")
        
        spins_hbox = QHBoxLayout()
        spins_hbox.addWidget(QLabel('Sine Freq.'))
        spins_hbox.addWidget(self.sine_freq)
        spins_hbox.addWidget(QLabel('Sine Amplitude'))
        spins_hbox.addWidget(self.sine_amplitude)
        spins_hbox.addWidget(QLabel('Noise Level'))
        spins_hbox.addWidget(self.noise_level)
        spins_hbox.addWidget(QLabel('Kernel Width'))
        spins_hbox.addWidget(self.sigma)
        spins_hbox.addStretch(1)
        
        self.show_button = QPushButton("&Train GP")
        self.connect(self.show_button, SIGNAL('clicked()'), self.plot_gp)

        self.gen_sine_data_button = QPushButton("&Generate Sine Data")
        self.connect(self.gen_sine_data_button, SIGNAL('clicked()'), self.create_toy_data)

        
        self.learn_kernel_button = QPushButton("&Learn Kernel Width and train GP")
        self.connect(self.learn_kernel_button, SIGNAL('clicked()'), self.learn_kernel_width)


        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.canvas)
        #left_vbox.addWidget(self.mpl_toolbar)

        right2_vbox = QVBoxLayout()
        right2_vbox.addWidget(QLabel("Settings"))
        right2_vbox.addWidget(self.gen_sine_data_button)
        right2_vbox.addWidget(self.show_button)
        #right2_vbox.addWidget(self.learn_kernel_button)
        right2_vbox.addLayout(spins_hbox)
        

        right2_vbox.addStretch(1)

        right_vbox = QHBoxLayout()
        right_vbox.addLayout(right2_vbox)
    
        
        hbox = QVBoxLayout()
        hbox.addLayout(left_vbox)
        hbox.addLayout(right_vbox)
        self.main_frame.setLayout(hbox)

        self.setCentralWidget(self.main_frame)


    def create_status_bar(self):
        self.status_text = QLabel("")
        self.statusBar().addWidget(self.status_text, 1)

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

def main():
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
    
