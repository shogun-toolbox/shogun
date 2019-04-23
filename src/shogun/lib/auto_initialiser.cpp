/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/lib/auto_initialiser.h>

namespace shogun
{
	namespace params
	{
		const char* const GammaFeatureNumberInit::kName =
		    "GammaFeatureNumberInit";
		const char* const GammaFeatureNumberInit::kDescription =
		    "Automatic initialisation of the gamma dot product scaling "
		    "parameter. If the standard deviation of the features can be "
		    "calculated then gamma = 1 / (n_features * std(features)), else "
		    "gamma = 1 / n_features.";
	} // namespace factory
} // namespace shogun
