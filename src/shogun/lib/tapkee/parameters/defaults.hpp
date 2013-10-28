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
const ParametersSet defaults = (
	tapkee::keywords::eigen_method = tapkee::keywords::by_default,
	tapkee::keywords::neighbors_method = tapkee::keywords::by_default,
	tapkee::keywords::num_neighbors = tapkee::keywords::by_default,
	tapkee::keywords::target_dimension = tapkee::keywords::by_default,
	tapkee::keywords::diffusion_map_timesteps = tapkee::keywords::by_default,
	tapkee::keywords::gaussian_kernel_width = tapkee::keywords::by_default,
	tapkee::keywords::max_iteration = tapkee::keywords::by_default,
	tapkee::keywords::spe_global_strategy = tapkee::keywords::by_default,
	tapkee::keywords::spe_num_updates = tapkee::keywords::by_default,
	tapkee::keywords::spe_tolerance = tapkee::keywords::by_default,
	tapkee::keywords::landmark_ratio = tapkee::keywords::by_default,
	tapkee::keywords::nullspace_shift = tapkee::keywords::by_default,
	tapkee::keywords::klle_shift = tapkee::keywords::by_default,
	tapkee::keywords::check_connectivity = tapkee::keywords::by_default,
	tapkee::keywords::fa_epsilon = tapkee::keywords::by_default,
	tapkee::keywords::progress_function = tapkee::keywords::by_default,
	tapkee::keywords::cancel_function = tapkee::keywords::by_default,
	tapkee::keywords::sne_perplexity = tapkee::keywords::by_default,
	tapkee::keywords::squishing_rate = tapkee::keywords::by_default,
	tapkee::keywords::sne_theta = tapkee::keywords::by_default);

}

}
}

#endif
