/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Chiyuan Zhang, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/converter/StochasticProximityEmbedding.h>
#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

StochasticProximityEmbedding::StochasticProximityEmbedding() :
	EmbeddingConverter()
{
	// Initialize to default values
	m_k         = 12;
	m_nupdates  = 100;
	m_strategy  = SPE_GLOBAL;
	m_tolerance = 1e-5;
	m_max_iteration  = 0;

	init();
}

void StochasticProximityEmbedding::init()
{
	SG_ADD(&m_k, "m_k", "Number of neighbors");
	SG_ADD(&m_tolerance, "m_tolerance", "Regularization parameter");
	SG_ADD(&m_max_iteration, "max_iteration", "maximum number of iterations");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_strategy, "m_strategy", "SPE strategy",
	    ParameterProperties::NONE, SG_OPTIONS(SPE_GLOBAL, SPE_LOCAL));
}

StochasticProximityEmbedding::~StochasticProximityEmbedding()
{
}

void StochasticProximityEmbedding::set_k(int32_t k)
{
	if ( k <= 0 )
		SG_ERROR("Number of neighbors k must be greater than 0")

	m_k = k;
}

int32_t StochasticProximityEmbedding::get_k() const
{
	return m_k;
}

void StochasticProximityEmbedding::set_strategy(ESPEStrategy strategy)
{
	m_strategy = strategy;
}

ESPEStrategy StochasticProximityEmbedding::get_strategy() const
{
	return m_strategy;
}

void StochasticProximityEmbedding::set_tolerance(float32_t tolerance)
{
	if ( tolerance <= 0 )
		SG_ERROR("Tolerance regularization parameter must be greater "
			 "than 0");

	m_tolerance = tolerance;
}

int32_t StochasticProximityEmbedding::get_tolerance() const
{
	return m_tolerance;
}

void StochasticProximityEmbedding::set_nupdates(int32_t nupdates)
{
	if ( nupdates <= 0 )
		SG_ERROR("The number of updates must be greater than 0")

	m_nupdates = nupdates;
}

int32_t StochasticProximityEmbedding::get_nupdates() const
{
	return m_nupdates;
}

void StochasticProximityEmbedding::set_max_iteration(const int32_t max_iteration)
{
	m_max_iteration = max_iteration;
}

int32_t StochasticProximityEmbedding::get_max_iteration() const
{
	return m_max_iteration;
}

const char * StochasticProximityEmbedding::get_name() const
{
	return "StochasticProximityEmbedding";
}

std::shared_ptr<Features>
StochasticProximityEmbedding::transform(std::shared_ptr<Features> features, bool inplace)
{
	if ( !features )
		SG_ERROR("Features are required to apply SPE\n")

	// Shorthand for the DenseFeatures
	auto simple_features =
		std::static_pointer_cast<DenseFeatures<float64_t>>(features);


	// Get and check the number of vectors
	int32_t N = simple_features->get_num_vectors();
	if ( m_strategy == SPE_LOCAL && m_k >= N )
		SG_ERROR("The number of neighbors (%d) must be less than "
		         "the number of vectors (%d)\n", m_k, N);

	if ( 2*m_nupdates > N )
		SG_ERROR("The number of vectors (%d) must be at least two times "
			 "the number of updates (%d)\n", N, m_nupdates);

	m_distance->init(simple_features, simple_features);
	auto embedding = embed_distance(m_distance);
	m_distance->remove_lhs_and_rhs();


	return embedding;
}

std::shared_ptr<DenseFeatures< float64_t >> StochasticProximityEmbedding::embed_distance(std::shared_ptr<Distance> distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.method = SHOGUN_STOCHASTIC_PROXIMITY_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.spe_num_updates = m_nupdates;
	parameters.spe_tolerance = m_tolerance;
	parameters.distance = distance.get();
	parameters.spe_global_strategy = (m_strategy==SPE_GLOBAL);
	parameters.max_iteration = m_max_iteration;
	return tapkee_embed(parameters);
}

