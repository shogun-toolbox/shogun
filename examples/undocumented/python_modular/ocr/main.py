#!/usr/bin/env python

# File   : $HeadURL$
# Version: $Id$

import gtk, sys
import numpy as np

from FigureWidget import FigureWidget
from MatrixWidget import MatrixWidget
from Ai import Ai

import common as com

def button_go_clicked(button, main_window):
    coords = com.MATIX_IMAGE_SIZE * main_window.figure.get_coords()
    image = np.zeros((com.MATIX_IMAGE_SIZE, com.MATIX_IMAGE_SIZE),
                     dtype=np.float)

    for xy in coords:
        image[xy[1], xy[0]] = com.FEATURE_RANGE_MAX

    main_window.push_image(image,
                           str(main_window.ai.classify(image))
                           )

    main_window.set_focus(main_window.button_go)

def button_tr_err_clicked(button, main_window):
    if main_window.ask_long_time("training error") == gtk.RESPONSE_YES:
        main_window.ai.show_train_error(main_window)

def button_te_err_clicked(button, main_window):
    if main_window.ask_long_time("test error") == gtk.RESPONSE_YES:
        main_window.ai.show_test_error(
            main_window,
            0.01 * main_window.button_te_spin.get_value())

def button_clear_clicked(button, main_window):
    main_window.figure.clear_coords()
    main_window.set_focus(main_window.button_go)

def button_load_clicked(button, main_window):
    main_window.ai.load_classifier(com.TRAIN_X_FNAME,
                                   com.TRAIN_Y_FNAME,
                                   main_window)

def button_train_clicked(button, main_window):
    if main_window.ask_long_time("training") == gtk.RESPONSE_YES:
        main_window.ai.load_train_data_train(com.TRAIN_X_FNAME,
                                             com.TRAIN_Y_FNAME,
                                             main_window)

class MainWindow(gtk.Window):
    MIN_WIDTH = 800
    MIN_HEIGHT = 450

    MAIN_PADDING = 4
    BOX_PADDING = 4

    MARKUP_PRE = '<span size="x-large" color="red"><b>'
    MARKUP_POST = '</b></span>'

    def __init__(self):
        # Main Window
        gtk.Window.__init__(self, type=gtk.WINDOW_TOPLEVEL)
        self.connect("delete-event", MainWindow.on_delete)
        self.set_size_request(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Wait dialog
        self.wait_dialog = \
            gtk.MessageDialog(self, gtk.DIALOG_MODAL, gtk.MESSAGE_INFO,
                              gtk.BUTTONS_NONE, "")
        self.wait_dialog.set_markup("Please wait ..."
                                    "\n\n<i>*cooking coffee*</i>")
        self.wait_dialog.set_deletable(False)

        # AI
        self.ai = Ai()

        # Main Container
        self.main_align = gtk.Alignment(xalign=0.0, yalign=0.0,
                                        xscale=1.0, yscale=1.0)
        self.main_align.set_padding(self.MAIN_PADDING,
                                    self.MAIN_PADDING,
                                    self.MAIN_PADDING,
                                    self.MAIN_PADDING)
        self.add(self.main_align)

        # Main VBox
        self.main_vbox = gtk.VBox(homogeneous=False,
                                  spacing=self.BOX_PADDING)
        self.main_align.add(self.main_vbox)

        # Figure HBox
        self.figure_hbox = gtk.HBox(homogeneous=False,
                                    spacing=self.BOX_PADDING)
        self.main_vbox.pack_start(self.figure_hbox, expand=True,
                                  fill=True, padding=0)

        # Figure
        self.figure = FigureWidget()
        self.figure_hbox.pack_start(self.figure, expand=True,
                                    fill=True, padding=0)

        # VSeperator
        self.figure_vsep = gtk.VSeparator()
        self.figure_hbox.pack_start(self.figure_vsep, expand=False,
                                    fill=True, padding=0)

        # History
        self.mat_table = gtk.Table(rows=com.HISTORY_HEIGHT,
                                   columns=com.HISTORY_WIDTH,
                                   homogeneous=True)
        self.figure_hbox.pack_start(self.mat_table, expand=True,
                                    fill=True, padding=0)

        self.mat_frame = []
        self.mat_vbox = []
        self.mat_images = []
        self.mat_result = []
        for y in range(com.HISTORY_HEIGHT):
            for x in range(com.HISTORY_WIDTH):
                i = y*com.HISTORY_WIDTH + x

                self.mat_frame.append(gtk.Frame(
                        "History " + str(y*com.HISTORY_WIDTH + x)))
                if i == 0:
                    self.mat_frame[i].set_label("Current")
                    mf_style = self.mat_frame[i].get_style()
                    mf_style.bg[gtk.STATE_NORMAL] = com.COLOR_BLUE
                    self.mat_frame[i].set_style(mf_style)
                self.mat_table.attach(self.mat_frame[i],
                                      left_attach=x, right_attach=x+1,
                                      top_attach=y, bottom_attach=y+1,
                                      xpadding=self.BOX_PADDING)

                self.mat_vbox.append(gtk.VBox(homogeneous=False,
                                              spacing=self.BOX_PADDING)
                                     )
                self.mat_frame[i].add(self.mat_vbox[i])

                self.mat_result.append(gtk.Label(
                        self.MARKUP_PRE + "?" + self.MARKUP_POST))
                self.mat_result[i].set_use_markup(True)
                self.mat_vbox[i].pack_start(
                    self.mat_result[i], expand=False, fill=True,
                    padding=0)

                self.mat_images.append(MatrixWidget(
                        com.MATIX_IMAGE_SIZE))
                self.mat_vbox[i].pack_start(
                    self.mat_images[i], expand=True, fill=True,
                    padding=self.BOX_PADDING)

        # HSeperator
        self.main_hsep = gtk.HSeparator()
        self.main_vbox.pack_start(self.main_hsep, expand=False,
                                  fill=True, padding=0)

        # HBox
        self.hbox = gtk.HBox(homogeneous=False,
                             spacing=self.BOX_PADDING)
        self.main_vbox.pack_end(self.hbox, expand=False, fill=True,
                                padding=0)

        # Button Go
        self.button_go = gtk.Button(label="_Go!")
        self.button_go.connect("clicked", button_go_clicked, self)
        self.button_go.set_focus_on_click(False)
        self.button_go.set_sensitive(False)
        self.hbox.add(self.button_go)

        # Button clear
        self.button_clear = gtk.Button(label="_clear")
        self.button_clear.set_focus_on_click(False)
        self.button_clear.connect("clicked", button_clear_clicked,
                                  self)
        self.hbox.add(self.button_clear)

        # Button VSeperator 1
        self.button_vsep1 = gtk.VSeparator()
        self.hbox.pack_start(self.button_vsep1, expand=False, fill=True,
                             padding=0)

        # Button train error
        self.button_tr_err = gtk.Button(label="train _error")
        self.button_tr_err.connect("clicked", button_tr_err_clicked,
                                   self)
        self.button_tr_err.set_focus_on_click(False)
        self.button_tr_err.set_sensitive(False)
        self.hbox.add(self.button_tr_err)

        # Button test error
        self.button_te_vbox = gtk.VBox(
            homogeneous=False, spacing=self.BOX_PADDING)
        self.hbox.add(self.button_te_vbox)

        self.button_te_err = gtk.Button(label="test e_rror")
        self.button_te_err.connect("clicked", button_te_err_clicked,
                                   self)
        self.button_te_err.set_focus_on_click(False)
        self.button_te_err.set_sensitive(False)
        self.button_te_vbox.add(self.button_te_err)

        self.button_te_spin = gtk.SpinButton(
            gtk.Adjustment(value=8.0,
                           lower=0.0, upper=100.0,
                           step_incr=1.0,
                           page_incr=0, page_size=0),
            climb_rate=1.0, digits=2)
        self.button_te_spin.set_tooltip_text("Noise in %")
        self.button_te_spin.set_sensitive(False)
        self.button_te_vbox.add(self.button_te_spin)

        # Button VSeperator 2
        self.button_vsep2 = gtk.VSeparator()
        self.hbox.pack_start(self.button_vsep2, expand=False, fill=True,
                             padding=0)

        # Button load classifier
        self.button_load = gtk.Button(label="_load classifier from "
                                      "file")
        self.button_load.set_focus_on_click(False)
        self.button_load.connect("clicked", button_load_clicked,
                                 self)
        self.hbox.add(self.button_load)

        # Button train from file
        self.button_train = gtk.Button(label="SLOW: _train from file")
        self.button_train.set_focus_on_click(False)
        self.button_train.connect("clicked", button_train_clicked,
                                  self)
        self.hbox.add(self.button_train)

        self.button_go.set_flags(gtk.CAN_DEFAULT)
        self.set_default(self.button_go)

        # Title
        self.set_title("OCR - C: %.2f, epsilon: %.2e, kernel-width: "
                       "%.2f" % (self.ai.C, self.ai.EPSILON,
                                self.ai.KERNEL_WIDTH)
                       )

    def on_delete(self, event):
        gtk.Window.destroy(self)
        gtk.main_quit()
        return True

    def push_image(self, image, str):
        prev_image = image
        prev_str = str

        for i in range(com.HISTORY_WIDTH*com.HISTORY_HEIGHT):
            tmp_image = self.mat_images[i].get_image()
            tmp_str = self.mat_result[i].get_text()
            self.mat_images[i].set_image(prev_image)
            self.mat_result[i].set_markup(
                self.MARKUP_PRE + prev_str + self.MARKUP_POST)
            prev_image = tmp_image
            prev_str = tmp_str

    def ask_long_time(self, str):
        msg_dialog = gtk.MessageDialog(
            self, gtk.DIALOG_MODAL, gtk.MESSAGE_WARNING,
            gtk.BUTTONS_YES_NO,
            "The " + str + " will take much time and is not "
            "interruptible! Are you sure to start the " + str + "?")
        resp = msg_dialog.run()
        msg_dialog.destroy()
        return resp

    def idle_show_wait(self):
        self.wait_dialog.show_all()
        return False

    def idle_enable_go(self, error):
        self.wait_dialog.hide()
        if not error:
            self.button_go.set_sensitive(True)
            self.button_tr_err.set_sensitive(True)
            self.button_te_err.set_sensitive(True)
            self.button_te_spin.set_sensitive(True)
            self.set_focus(self.button_go)
        return False

    def idle_info_dialog(self, msg):
        msg_dialog = gtk.MessageDialog(
            self, gtk.DIALOG_MODAL, gtk.MESSAGE_INFO, gtk.BUTTONS_OK,
            msg)
        msg_dialog.connect("response",
                           lambda dialog, arg1: dialog.destroy())
        msg_dialog.show()
        return False

def main(argv):
    gtk.gdk.threads_init()

    window = MainWindow()
    window.show_all()
    gtk.main()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
