# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Heiko Strathmann
# Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
#

def modelselection_parameter_tree_modular()

# *** 	root=ModelSelectionParameters()
	root=Modshogun::ModelSelectionParameters.new

	combinations=root.get_combinations()
	combinations.get_num_elements()

# *** 	c=ModelSelectionParameters('C');
	c=Modshogun::ModelSelectionParameters.new('C')
	#c.set_features('C')
	root.append_child(c)
	c.build_values(1, 11, Modshogun::R_EXP)

# *** 	power_kernel=PowerKernel()
	power_kernel=Modshogun::PowerKernel.new
	power_kernel.set_features()
# *** 	param_power_kernel=ModelSelectionParameters('kernel', power_kernel)
	param_power_kernel=Modshogun::ModelSelectionParameters.new
	param_power_kernel.set_features('kernel', power_kernel)
	root.append_child(param_power_kernel)

# *** 	param_power_kernel_degree=ModelSelectionParameters('degree')
	param_power_kernel_degree=Modshogun::ModelSelectionParameters.new
	param_power_kernel_degree.set_features('degree')
	param_power_kernel_degree.build_values(1, 1, R_EXP)
	param_power_kernel.append_child(param_power_kernel_degree)

# *** 	metric1=MinkowskiMetric(10)
	metric1=Modshogun::MinkowskiMetric.new
	metric1.set_features(10)
# *** 	param_power_kernel_metric1=ModelSelectionParameters('distance', metric1)
	param_power_kernel_metric1=Modshogun::ModelSelectionParameters.new
	param_power_kernel_metric1.set_features('distance', metric1)

	param_power_kernel.append_child(param_power_kernel_metric1)

# *** 	param_power_kernel_metric1_k=ModelSelectionParameters('k')
	param_power_kernel_metric1_k=Modshogun::ModelSelectionParameters.new
	param_power_kernel_metric1_k.set_features('k')
	param_power_kernel_metric1_k.build_values(1, 12, R_LINEAR)
	param_power_kernel_metric1.append_child(param_power_kernel_metric1_k)

# *** 	gaussian_kernel=GaussianKernel()
	gaussian_kernel=Modshogun::GaussianKernel.new
	gaussian_kernel.set_features()
# *** 	param_gaussian_kernel=ModelSelectionParameters('kernel', gaussian_kernel)
	param_gaussian_kernel=Modshogun::ModelSelectionParameters.new
	param_gaussian_kernel.set_features('kernel', gaussian_kernel)

	root.append_child(param_gaussian_kernel)

# *** 	param_gaussian_kernel_width=ModelSelectionParameters('width')
	param_gaussian_kernel_width=Modshogun::ModelSelectionParameters.new
	param_gaussian_kernel_width.set_features('width')
	param_gaussian_kernel_width.build_values(1, 2, R_EXP)
	param_gaussian_kernel.append_child(param_gaussian_kernel_width)

# *** 	ds_kernel=DistantSegmentsKernel()
	ds_kernel=Modshogun::DistantSegmentsKernel.new
	ds_kernel.set_features()
# *** 	param_ds_kernel=ModelSelectionParameters('kernel', ds_kernel)
	param_ds_kernel=Modshogun::ModelSelectionParameters.new
	param_ds_kernel.set_features('kernel', ds_kernel)

	root.append_child(param_ds_kernel)

# *** 	param_ds_kernel_delta=ModelSelectionParameters('delta')
	param_ds_kernel_delta=Modshogun::ModelSelectionParameters.new
	param_ds_kernel_delta.set_features('delta')
	param_ds_kernel_delta.build_values(1, 2, R_EXP)
	param_ds_kernel.append_child(param_ds_kernel_delta)

# *** 	param_ds_kernel_theta=ModelSelectionParameters('theta')
	param_ds_kernel_theta=Modshogun::ModelSelectionParameters.new
	param_ds_kernel_theta.set_features('theta')
	param_ds_kernel_theta.build_values(1, 2, R_EXP)
	param_ds_kernel.append_child(param_ds_kernel_theta)

	root.print_tree()
	combinations=root.get_combinations()
	for i in range(combinations.get_num_elements())
		combinations.get_element(i).print_tree()
	end

	return

end

if __FILE__ == $0
	puts 'ParameterTree'
	pp modelselection_parameter_tree_modular()
end
