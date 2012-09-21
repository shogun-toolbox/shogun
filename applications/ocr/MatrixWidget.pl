# File   : $HeadURL$
# Version: $Id$

import gtk
import numpy as np

import common as com
from QuadrWidget import QuadrWidget

class MatrixWidget(QuadrWidget):
    def __init__(self, matrix_size):
        QuadrWidget.__init__(self)

        self.matrix = np.zeros((matrix_size, matrix_size),
                               dtype=np.bool)

        self.connect("expose_event", MatrixWidget.on_redraw)

    def on_redraw(self, event):
        gc = self.style.fg_gc[self.state]
        w = self.window
        width = w.get_size()[0]
        height = w.get_size()[1]

        # Backup graphic context
        self.default_fg = gc.foreground

        # Background
        gc.set_rgb_fg_color(com.COLOR_WHITE)
        w.draw_rectangle(gc, True, 0, 0, width-1, height-1)

        size_y = self.matrix.shape[0]
        size_x = self.matrix.shape[1]
        pixels_per_y = float(height)/size_y
        pixels_per_x = float(width)/size_x
        gc.set_rgb_fg_color(com.COLOR_GRAY)
        for y in range(size_y):
            w.draw_line(gc, 0, int(y*pixels_per_y),
                        width-1, int(y*pixels_per_y))
            for x in range(size_x):
                if y == 0:
                    w.draw_line(gc, int(x*pixels_per_x), 0,
                                int(x*pixels_per_x), height-1)
                if self.matrix[y, x]:
                    gc.set_rgb_fg_color(com.COLOR_BLACK)
                    w.draw_rectangle(gc, self.matrix[y, x]
                                     > com.NEAR_ZERO_POS,
                                     int(x*pixels_per_x),
                                     int(y*pixels_per_y),
                                     int(pixels_per_x+1),
                                     int(pixels_per_y+1))
                    gc.set_rgb_fg_color(com.COLOR_GRAY)

        gc.set_rgb_fg_color(com.COLOR_BLACK)
        w.draw_rectangle(gc, False, 0, 0, width-1, height-1)

        gc.foreground = self.default_fg

        return False

    def set_image(self, image):
        self.matrix = image
        self.update()

    def get_image(self):
        return self.matrix
