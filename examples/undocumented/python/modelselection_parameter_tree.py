#!/usr/bin/env python
#
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Heiko Strathmann

parameter_list=[[None]]

def modelselection_parameter_tree (dummy):
    from shogun import ParameterCombination
    from shogun import ModelSelectionParameters, R_EXP, R_LINEAR
    from shogun import PowerKernel
    from shogun import GaussianKernel
    from shogun import DistantSegmentsKernel
    from shogun import MinkowskiMetric
    import math

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

    param_gaussian_kernel_width=ModelSelectionParameters('log_width')
    param_gaussian_kernel_width.build_values(0.0, 0.5*math.log(2.0), R_LINEAR)
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
    #		params = ParameterCombination.obtain_from_generic(combinations.get_element(i))
    #		params.print_tree()

    return


if __name__=='__main__':
    print('ModelSelection ParameterTree')
    modelselection_parameter_tree(*parameter_list[0])



