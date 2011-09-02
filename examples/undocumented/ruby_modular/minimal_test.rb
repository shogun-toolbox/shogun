require 'modshogun'
require 'shogun_helpers'
gk=Modshogun::GaussianKernel.new()
puts gk.get_width()
