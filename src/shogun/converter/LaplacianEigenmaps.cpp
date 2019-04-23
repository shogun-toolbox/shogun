/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Evan Shelhamer,
 *          Heiko Strathmann
 */

#include <shogun/converter/LaplacianEigenmaps.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

LaplacianEigenmaps::LaplacianEigenmaps() :
		EmbeddingConverter()
{
	m_k = 3;
	m_tau = 1.0;

	init();
}

void LaplacianEigenmaps::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", ParameterProperties::HYPER);
	SG_ADD(&m_tau, "tau", "heat distribution coefficient", ParameterProperties::HYPER);
}

LaplacianEigenmaps::~LaplacianEigenmaps()
{
}

void LaplacianEigenmaps::set_k(int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t LaplacianEigenmaps::get_k() const
{
	return m_k;
}

void LaplacianEigenmaps::set_tau(float64_t tau)
{
	m_tau = tau;
}

float64_t LaplacianEigenmaps::get_tau() const
{
	return m_tau;
}

const char* LaplacianEigenmaps::get_name() const
{
	return "LaplacianEigenmaps";
};

std::shared_ptr<Features> LaplacianEigenmaps::transform(std::shared_ptr<Features> features, bool inplace)
{
	// shorthand for simplefeatures


	// get dimensionality and number of vectors of data
	int32_t N = features->get_num_vectors();
	ASSERT(m_k<N)
	ASSERT(m_target_dim<N)

	// compute distance matrix
	ASSERT(m_distance)
	m_distance->init(features,features);
	auto embedding = embed_distance(m_distance);
	m_distance->remove_lhs_and_rhs();

	return embedding;
}

std::shared_ptr<DenseFeatures<float64_t>> LaplacianEigenmaps::embed_distance(std::shared_ptr<Distance> distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.gaussian_kernel_width = m_tau;
	parameters.method = SHOGUN_LAPLACIAN_EIGENMAPS;
	parameters.target_dimension = m_target_dim;
	parameters.distance = distance.get();
	return tapkee_embed(parameters);
}
