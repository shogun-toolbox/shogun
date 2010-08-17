# File   : $HeadURL$
# Version: $Id$

import gtk

class QuadrWidget(gtk.DrawingArea):

    # No SELF.CONNECT because we like to prevent the call of
    # gtk.DrawingArea.do_size_allocate()
    __gsignals__ = {"size-allocate": 'override'}

    def __init__(self):
        gtk.DrawingArea.__init__(self)

    def update(self):
        width = self.window.get_size()[0]
        height = self.window.get_size()[1]
        self.window.invalidate_rect(gtk.gdk.Rectangle(
                0, 0, width, height), False)
        #self.window.process_updates(False)

    def do_size_allocate(self, allocation):
        if allocation.width < allocation.height:
            allocation.y += (allocation.height-allocation.width)/2
            allocation.height = allocation.width
        elif allocation.width > allocation.height:
            allocation.x += (allocation.width-allocation.height)/2
            allocation.width = allocation.height

        gtk.DrawingArea.do_size_allocate(self, allocation)
