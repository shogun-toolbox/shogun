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

ManifoldSculpting::ManifoldSculpting() :
		EmbeddingConverter()
{
	// Default values
	m_k = 10;
	m_squishing_rate = 0.8;
	m_max_iteration = 80;
	init();
}

void ManifoldSculpting::init()
{
	SG_ADD(&m_k, "k", "number of neighbors");
	SG_ADD(&m_squishing_rate, "quishing_rate",
      "squishing rate");
	SG_ADD(&m_max_iteration, "max_iteration",
      "maximum number of algorithm's iterations");
}

ManifoldSculpting::~ManifoldSculpting()
{
}

const char* ManifoldSculpting::get_name() const
{
	return "ManifoldSculpting";
}

void ManifoldSculpting::set_k(const int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t ManifoldSculpting::get_k() const
{
	return m_k;
}

void ManifoldSculpting::set_squishing_rate(const float64_t squishing_rate)
{
	ASSERT(squishing_rate >= 0 && squishing_rate < 1)
	m_squishing_rate = squishing_rate;
}

float64_t ManifoldSculpting::get_squishing_rate() const
{
	return m_squishing_rate;
}

void ManifoldSculpting::set_max_iteration(const int32_t max_iteration)
{
	ASSERT(max_iteration > 0)
	m_max_iteration = max_iteration;
}

int32_t ManifoldSculpting::get_max_iteration() const
{
	return m_max_iteration;
}

std::shared_ptr<Features> ManifoldSculpting::transform(std::shared_ptr<Features> features, bool inplace)
{
	auto feats = std::static_pointer_cast<DenseFeatures<float64_t>>(features);

	auto euclidean_distance =
		std::make_shared<EuclideanDistance>(feats, feats);

	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.squishing_rate = m_squishing_rate;
	parameters.max_iteration = m_max_iteration;
	parameters.features = feats.get();
	parameters.distance = euclidean_distance.get();

	parameters.method = SHOGUN_MANIFOLD_SCULPTING;
	parameters.target_dimension = m_target_dim;
	return tapkee_embed(parameters);
}

