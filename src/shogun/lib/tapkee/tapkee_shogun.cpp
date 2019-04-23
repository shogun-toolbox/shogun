/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/lib/tapkee/tapkee_shogun.hpp>

// FIXME: Random Refactor PR
// #define CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION shogun::CMath::random()
// #define CUSTOM_UNIFORM_RANDOM_FUNCTION shogun::CMath::random(static_cast<tapkee::ScalarType>(0),static_cast<tapkee::ScalarType>(1))
// #define CUSTOM_GAUSSIAN_RANDOM_FUNCTION shogun::CMath::normal_random(static_cast<tapkee::ScalarType>(0),static_cast<tapkee::ScalarType>(1))
#define TAPKEE_EIGEN_INCLUDE_FILE <shogun/mathematics/eigen3.h>

#ifdef HAVE_ARPACK
	#define TAPKEE_WITH_ARPACK
#endif
#ifdef USE_GPL_SHOGUN
	#define TAPKEE_USE_LGPL_COVERTREE
#endif
#include <shogun/lib/tapkee/tapkee.hpp>
#include <shogun/lib/tapkee/callbacks/pimpl_callbacks.hpp>

using namespace shogun;

class ShogunLoggerImplementation : public tapkee::LoggerImplementation
{
	virtual void message_info(const std::string& msg)
	{
		io::info((msg+"").c_str());
	}
	virtual void message_warning(const std::string& msg)
	{
		io::warn((msg+"").c_str());
	}
	virtual void message_error(const std::string& msg)
	{
		error((msg+"").c_str());
	}
	virtual void message_debug(const std::string& msg)
	{
		SG_DEBUG((msg+"").c_str())
	}
	virtual void message_benchmark(const std::string& msg)
	{
		io::info((msg+"").c_str());
	}
};

struct ShogunFeatureVectorCallback
{
	ShogunFeatureVectorCallback(DotFeatures* f) : dim(0), features(f) { }
	inline tapkee::IndexType dimension() const
	{
		if (features)
			return (dim = features->get_dim_feature_space());

		return 0;
	}
	inline void vector(int i, tapkee::DenseVector& v) const
	{
		v.setZero();
		features->add_to_dense_vec(1.0,i,v.data(),dim);
	}
	mutable int32_t dim;
	DotFeatures* features;
};


std::shared_ptr<DenseFeatures<float64_t>> shogun::tapkee_embed(const shogun::TAPKEE_PARAMETERS_FOR_SHOGUN& parameters)
{
	tapkee::LoggingSingleton::instance().set_logger_impl(new ShogunLoggerImplementation);
	tapkee::LoggingSingleton::instance().enable_benchmark();
	tapkee::LoggingSingleton::instance().enable_info();

	pimpl_kernel_callback<Kernel> kernel_callback(parameters.kernel);
	pimpl_distance_callback<Distance> distance_callback(parameters.distance);
	ShogunFeatureVectorCallback features_callback(parameters.features);

	tapkee::DimensionReductionMethod method = tapkee::PCA;
#ifdef HAVE_ARPACK
	tapkee::EigenMethod eigen_method = tapkee::Arpack;
#else
	tapkee::EigenMethod eigen_method = tapkee::Dense;
#endif
#ifdef TAPKEE_USE_LGPL_COVERTREE
	tapkee::NeighborsMethod neighbors_method = tapkee::CoverTree;
#else
	tapkee::NeighborsMethod neighbors_method = tapkee::VpTree;
#endif
	size_t N = 0;

	switch (parameters.method)
	{
		case SHOGUN_KERNEL_LOCALLY_LINEAR_EMBEDDING:
		case SHOGUN_LOCALLY_LINEAR_EMBEDDING:
			method = tapkee::KernelLocallyLinearEmbedding;
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_NEIGHBORHOOD_PRESERVING_EMBEDDING:
			method = tapkee::NeighborhoodPreservingEmbedding;
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_LOCAL_TANGENT_SPACE_ALIGNMENT:
			method = tapkee::KernelLocalTangentSpaceAlignment;
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			method = tapkee::LinearLocalTangentSpaceAlignment;
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			method = tapkee::HessianLocallyLinearEmbedding;
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_DIFFUSION_MAPS:
			method = tapkee::DiffusionMap;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LAPLACIAN_EIGENMAPS:
			method = tapkee::LaplacianEigenmaps;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LOCALITY_PRESERVING_PROJECTIONS:
			method = tapkee::LocalityPreservingProjections;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_MULTIDIMENSIONAL_SCALING:
			method = tapkee::MultidimensionalScaling;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LANDMARK_MULTIDIMENSIONAL_SCALING:
			method = tapkee::LandmarkMultidimensionalScaling;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_ISOMAP:
			method = tapkee::Isomap;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LANDMARK_ISOMAP:
			method = tapkee::LandmarkIsomap;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_STOCHASTIC_PROXIMITY_EMBEDDING:
			method = tapkee::StochasticProximityEmbedding;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_FACTOR_ANALYSIS:
			method = tapkee::FactorAnalysis;
			N = parameters.features->get_num_vectors();
			break;
		case SHOGUN_TDISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING:
			method = tapkee::tDistributedStochasticNeighborEmbedding;
			N = parameters.features->get_num_vectors();
			break;
		case SHOGUN_MANIFOLD_SCULPTING:
			method = tapkee::ManifoldSculpting;
			N = parameters.features->get_num_vectors();
			break;
	}

	std::vector<int32_t> indices(N);
	for (size_t i=0; i<N; i++)
		indices[i] = i;

	tapkee::ParametersSet parameters_set =
		(tapkee::method=method,
		 tapkee::eigen_method=eigen_method,
		 tapkee::neighbors_method=neighbors_method,
		 tapkee::num_neighbors=parameters.n_neighbors,
		 tapkee::diffusion_map_timesteps = parameters.n_timesteps,
		 tapkee::target_dimension = parameters.target_dimension,
		 tapkee::spe_num_updates = parameters.spe_num_updates,
		 tapkee::nullspace_shift = parameters.eigenshift,
		 tapkee::landmark_ratio = parameters.landmark_ratio,
		 tapkee::gaussian_kernel_width = parameters.gaussian_kernel_width,
		 tapkee::spe_tolerance = parameters.spe_tolerance,
		 tapkee::spe_global_strategy = parameters.spe_global_strategy,
		 tapkee::max_iteration = parameters.max_iteration,
		 tapkee::fa_epsilon = parameters.fa_epsilon,
		 tapkee::sne_perplexity = parameters.sne_perplexity,
		 tapkee::sne_theta = parameters.sne_theta,
		 tapkee::squishing_rate = parameters.squishing_rate
		 );

	tapkee::TapkeeOutput output = tapkee::embed(indices.begin(),indices.end(),
			kernel_callback,distance_callback,features_callback,parameters_set);
	tapkee::DenseMatrix result_embedding = output.embedding;
	// destroy projecting function
	output.projection.clear();

	SGMatrix<float64_t> feature_matrix(parameters.target_dimension,N);
	// TODO avoid copying
	for (uint32_t i=0; i<N; i++)
	{
		for (uint32_t j=0; j<parameters.target_dimension; j++)
		{
			feature_matrix(j,i) = result_embedding(i,j);
		}
	}
	return std::make_shared<DenseFeatures<float64_t>>(feature_matrix);
}

