require 'rubygems'
require 'modshogun'
require 'load'
require 'pp'

#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Heiko Strathmann
# Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
# Trancekoded (T) 2011 Justin Patera aka serialhex

def modelselection_parameter_tree_modular()

	root=Modshogun::ModelSelectionParameters.new

	combinations=root.get_combinations()
	combinations.get_num_elements()

	c=Modshogun::ModelSelectionParameters.new('C')
	root.append_child(c)
	c.build_values(1, 11, Modshogun::R_EXP)

	power_kernel=Modshogun::PowerKernel.new
	param_power_kernel=Modshogun::ModelSelectionParameters.new('kernel', power_kernel)
	root.append_child(param_power_kernel)

	param_power_kernel_degree=Modshogun::ModelSelectionParameters.new('degree')
	param_power_kernel_degree.build_values(1, 1, Modshogun::R_EXP)
	param_power_kernel.append_child(param_power_kernel_degree)

	metric1=Modshogun::MinkowskiMetric.new(10)
	param_power_kernel_metric1=Modshogun::ModelSelectionParameters.new('distance', metric1)

	param_power_kernel.append_child(param_power_kernel_metric1)

	param_power_kernel_metric1_k=Modshogun::ModelSelectionParameters.new('k')
	param_power_kernel_metric1_k.build_values(1, 12, Modshogun::R_LINEAR)
	param_power_kernel_metric1.append_child(param_power_kernel_metric1_k)

	gaussian_kernel=Modshogun::GaussianKernel.new
	param_gaussian_kernel=Modshogun::ModelSelectionParameters.new('kernel', gaussian_kernel)

	root.append_child(param_gaussian_kernel)

	param_gaussian_kernel_width=Modshogun::ModelSelectionParameters.new('width')
	param_gaussian_kernel_width.build_values(1, 2, Modshogun::R_EXP)
	param_gaussian_kernel.append_child(param_gaussian_kernel_width)

	ds_kernel=Modshogun::DistantSegmentsKernel.new
	param_ds_kernel=Modshogun::ModelSelectionParameters.new('kernel', ds_kernel)

	root.append_child(param_ds_kernel)

	param_ds_kernel_delta=Modshogun::ModelSelectionParameters.new('delta')
	param_ds_kernel_delta.build_values(1, 2, Modshogun::R_EXP)
	param_ds_kernel.append_child(param_ds_kernel_delta)

	param_ds_kernel_theta=Modshogun::ModelSelectionParameters.new('theta')
	param_ds_kernel_theta.build_values(1, 2, Modshogun::R_EXP)
	param_ds_kernel.append_child(param_ds_kernel_theta)

	#root.print_tree()
	combinations=root.get_combinations()
	#combinations.get_num_elements.times do |i|
	#	combinations.get_element(i).print_tree()
	#end

	return

end

if __FILE__ == $0
	puts 'ParameterTree'
	pp modelselection_parameter_tree_modular()
end
