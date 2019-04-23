/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Soeren Sonnenburg,
 *          Evan Shelhamer, Bjoern Esser
 */

#include <shogun/converter/LocalityPreservingProjections.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

LocalityPreservingProjections::LocalityPreservingProjections() :
		LaplacianEigenmaps()
{
}

LocalityPreservingProjections::~LocalityPreservingProjections()
{
}

const char* LocalityPreservingProjections::get_name() const
{
	return "LocalityPreservingProjections";
};

std::shared_ptr<Features>
LocalityPreservingProjections::transform(std::shared_ptr<Features> features, bool inplace)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	m_distance->init(features,features);
	parameters.n_neighbors = m_k;
	parameters.gaussian_kernel_width = m_tau;
	parameters.method = SHOGUN_LOCALITY_PRESERVING_PROJECTIONS;
	parameters.target_dimension = m_target_dim;
	parameters.distance = m_distance.get();
	parameters.features = (DotFeatures*)features.get();
	return tapkee_embed(parameters);
}

