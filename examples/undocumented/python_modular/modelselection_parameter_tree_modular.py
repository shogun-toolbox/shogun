#!/usr/bin/env python
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011-2012 Heiko Strathmann
# Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
#

parameter_list=[[None]]

def modelselection_parameter_tree_modular (dummy):
    from modshogun import ParameterCombination
    from modshogun import ModelSelectionParameters, R_EXP, R_LINEAR
    from modshogun import PowerKernel
    from modshogun import GaussianKernel
    from modshogun import DistantSegmentsKernel
    from modshogun import MinkowskiMetric

    root=ModelSelectionParameters()

    combinations=root.get_combinations()
    combinations.get_num_elements()

    c=ModelSelectionParameters('C');
    root.append_child(c)
    c.build_values(1, 11, R_EXP)

    power_kernel=PowerKernel()

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #power_kernel.print_modsel_params()

    param_power_kernel=ModelSelectionParameters('kernel', power_kernel)
    root.append_child(param_power_kernel)

    param_power_kernel_degree=ModelSelectionParameters('degree')
    param_power_kernel_degree.build_values(1, 1, R_EXP)
    param_power_kernel.append_child(param_power_kernel_degree)

    metric1=MinkowskiMetric(10)

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #metric1.print_modsel_params()

    param_power_kernel_metric1=ModelSelectionParameters('distance', metric1)

    param_power_kernel.append_child(param_power_kernel_metric1)

    param_power_kernel_metric1_k=ModelSelectionParameters('k')
    param_power_kernel_metric1_k.build_values(1, 12, R_LINEAR)
    param_power_kernel_metric1.append_child(param_power_kernel_metric1_k)

    gaussian_kernel=GaussianKernel()

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #gaussian_kernel.print_modsel_params()

    param_gaussian_kernel=ModelSelectionParameters('kernel', gaussian_kernel)

    root.append_child(param_gaussian_kernel)

    param_gaussian_kernel_width=ModelSelectionParameters('width')
    param_gaussian_kernel_width.build_values(1, 2, R_EXP)
    param_gaussian_kernel.append_child(param_gaussian_kernel_width)

    ds_kernel=DistantSegmentsKernel()

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #ds_kernel.print_modsel_params()

    param_ds_kernel=ModelSelectionParameters('kernel', ds_kernel)

    root.append_child(param_ds_kernel)

    param_ds_kernel_delta=ModelSelectionParameters('delta')
    param_ds_kernel_delta.build_values(1, 2, R_EXP)
    param_ds_kernel.append_child(param_ds_kernel_delta)

    param_ds_kernel_theta=ModelSelectionParameters('theta')
    param_ds_kernel_theta.build_values(1, 2, R_EXP)
    param_ds_kernel.append_child(param_ds_kernel_theta)

    #	root.print_tree()
    combinations=root.get_combinations()
    #	for i in range(combinations.get_num_elements()):
    #		combinations.get_element(i).print_tree()

    return


if __name__=='__main__':
    print('ModelSelection ParameterTree')
    modelselection_parameter_tree_modular(*parameter_list[0])



