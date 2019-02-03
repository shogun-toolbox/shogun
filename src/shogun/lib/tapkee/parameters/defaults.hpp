/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_DEFAULTS_H_
#define TAPKEE_DEFAULTS_H_

namespace tapkee
{
namespace tapkee_internal
{

namespace {
const stichwort::ParametersSet defaults = (
	tapkee::computation_strategy = stichwort::by_default,
	tapkee::eigen_method = stichwort::by_default,
	tapkee::neighbors_method = stichwort::by_default,
	tapkee::num_neighbors = stichwort::by_default,
	tapkee::target_dimension = stichwort::by_default,
	tapkee::diffusion_map_timesteps = stichwort::by_default,
	tapkee::gaussian_kernel_width = stichwort::by_default,
	tapkee::max_iteration = stichwort::by_default,
	tapkee::spe_global_strategy = stichwort::by_default,
	tapkee::spe_num_updates = stichwort::by_default,
	tapkee::spe_tolerance = stichwort::by_default,
	tapkee::landmark_ratio = stichwort::by_default,
	tapkee::nullspace_shift = stichwort::by_default,
	tapkee::klle_shift = stichwort::by_default,
	tapkee::check_connectivity = stichwort::by_default,
	tapkee::fa_epsilon = stichwort::by_default,
	tapkee::progress_function = stichwort::by_default,
	tapkee::cancel_function = stichwort::by_default,
	tapkee::sne_perplexity = stichwort::by_default,
	tapkee::squishing_rate = stichwort::by_default,
	tapkee::sne_theta = stichwort::by_default);
}

}
}

#endif
