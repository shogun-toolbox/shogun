/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Vladislav Horbatiuk, Bjoern Esser
 */

#include <shogun/converter/ManifoldSculpting.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;

CManifoldSculpting::CManifoldSculpting() :
		CEmbeddingConverter()
{
	// Default values
	m_k = 10;
	m_squishing_rate = 0.8;
	m_max_iteration = 80;
	init();
}

void CManifoldSculpting::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", MS_NOT_AVAILABLE);
	SG_ADD(&m_squishing_rate, "quishing_rate",
      "squishing rate",MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iteration, "max_iteration",
      "maximum number of algorithm's iterations", MS_NOT_AVAILABLE);
}

CManifoldSculpting::~CManifoldSculpting()
{
}

const char* CManifoldSculpting::get_name() const
{
	return "ManifoldSculpting";
}

void CManifoldSculpting::set_k(const int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t CManifoldSculpting::get_k() const
{
	return m_k;
}

void CManifoldSculpting::set_squishing_rate(const float64_t squishing_rate)
{
	ASSERT(squishing_rate >= 0 && squishing_rate < 1)
	m_squishing_rate = squishing_rate;
}

float64_t CManifoldSculpting::get_squishing_rate() const
{
	return m_squishing_rate;
}

void CManifoldSculpting::set_max_iteration(const int32_t max_iteration)
{
	ASSERT(max_iteration > 0)
	m_max_iteration = max_iteration;
}

int32_t CManifoldSculpting::get_max_iteration() const
{
	return m_max_iteration;
}

CFeatures* CManifoldSculpting::apply(CFeatures* features, bool inplace)
{
	CDenseFeatures<float64_t>* feats = (CDenseFeatures<float64_t>*)features;
	SG_REF(feats);
	CDistance* euclidean_distance =
	new CEuclideanDistance(feats, feats);

	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.squishing_rate = m_squishing_rate;
	parameters.max_iteration = m_max_iteration;
	parameters.features = feats;
	parameters.distance = euclidean_distance;

	parameters.method = SHOGUN_MANIFOLD_SCULPTING;
	parameters.target_dimension = m_target_dim;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);

	SG_UNREF(euclidean_distance);

	return embedding;
}

