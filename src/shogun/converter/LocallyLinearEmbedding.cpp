/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Evan Shelhamer,
 *          Heiko Strathmann
 */

#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

LocallyLinearEmbedding::LocallyLinearEmbedding() :
		EmbeddingConverter()
{
	m_k = 10;
	m_nullspace_shift = -1e-9;
	m_reconstruction_shift = 1e-3;
	init();
}

void LocallyLinearEmbedding::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", ParameterProperties::HYPER);
	SG_ADD(&m_nullspace_shift, "nullspace_shift",
      "nullspace finding regularization shift");
	SG_ADD(&m_reconstruction_shift, "reconstruction_shift",
      "shift used to regularize reconstruction step");
}


LocallyLinearEmbedding::~LocallyLinearEmbedding()
{
}

void LocallyLinearEmbedding::set_k(int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t LocallyLinearEmbedding::get_k() const
{
	return m_k;
}

void LocallyLinearEmbedding::set_nullspace_shift(float64_t nullspace_shift)
{
	m_nullspace_shift = nullspace_shift;
}

float64_t LocallyLinearEmbedding::get_nullspace_shift() const
{
	return m_nullspace_shift;
}

void LocallyLinearEmbedding::set_reconstruction_shift(float64_t reconstruction_shift)
{
	m_reconstruction_shift = reconstruction_shift;
}

float64_t LocallyLinearEmbedding::get_reconstruction_shift() const
{
	return m_reconstruction_shift;
}

const char* LocallyLinearEmbedding::get_name() const
{
	return "LocallyLinearEmbedding";
}

std::shared_ptr<Features> LocallyLinearEmbedding::transform(std::shared_ptr<Features> features, bool inplace)
{
	// oh my let me dirty cast it
	auto dot_feats = std::static_pointer_cast<DotFeatures>(features);
	auto kernel = std::make_shared<LinearKernel>(dot_feats, dot_feats);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_LOCALLY_LINEAR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel.get();
	return tapkee_embed(parameters);
}

