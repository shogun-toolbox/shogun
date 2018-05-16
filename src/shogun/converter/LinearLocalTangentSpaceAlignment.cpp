/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann, 
 *          Evan Shelhamer
 */

#include <shogun/converter/LinearLocalTangentSpaceAlignment.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CLinearLocalTangentSpaceAlignment::CLinearLocalTangentSpaceAlignment() :
		CLocalTangentSpaceAlignment()
{
}

CLinearLocalTangentSpaceAlignment::~CLinearLocalTangentSpaceAlignment()
{
}

const char* CLinearLocalTangentSpaceAlignment::get_name() const
{
	return "LinearLocalTangentSpaceAlignment";
}

CFeatures*
CLinearLocalTangentSpaceAlignment::apply(CFeatures* features, bool inplace)
{
	CKernel* kernel = new CLinearKernel((CDotFeatures*)features,(CDotFeatures*)features);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel;
	parameters.features = (CDotFeatures*)features;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	SG_UNREF(kernel);
	return embedding;
}

