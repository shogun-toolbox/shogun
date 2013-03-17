/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/lib/tapkee/tapkee_shogun.hpp>

#ifdef HAVE_EIGEN3

#define TAPKEE_EIGEN_INCLUDE_FILE <shogun/mathematics/eigen3.h>
#ifdef HAVE_ARPACK
	#define TAPKEE_WITH_ARPACK 
#endif
#define TAPKEE_USE_LGPL_COVERTREE
#include <shogun/lib/tapkee/tapkee.hpp>
#include <shogun/lib/tapkee/callback/pimpl_callbacks.hpp>

TAPKEE_CALLBACK_IS_KERNEL(pimpl_kernel_callback<CKernel>);
TAPKEE_CALLBACK_IS_DISTANCE(pimpl_distance_callback<CDistance>);

using namespace shogun;

class ShogunLoggerImplementation : public tapkee::LoggerImplementation
{
	virtual void message_info(const std::string& msg)
	{
		SG_SINFO((msg+"\n").c_str())
	}
	virtual void message_warning(const std::string& msg)
	{
		SG_SWARNING((msg+"\n").c_str())
	}
	virtual void message_error(const std::string& msg)
	{
		SG_SERROR((msg+"\n").c_str())
	}
	virtual void message_debug(const std::string& msg)
	{
		SG_SDEBUG((msg+"\n").c_str())
	}
	virtual void message_benchmark(const std::string& msg)
	{
		SG_SINFO((msg+"\n").c_str())
	}
};

struct ShogunFeatureVectorCallback
{
	ShogunFeatureVectorCallback(CDotFeatures* f) : features(f) { }
	inline void operator()(int i, tapkee::DenseVector& vector) const
	{
		vector.setZero();
		features->add_to_dense_vec(1.0,i,vector.data(),features->get_dim_feature_space());
	}
	CDotFeatures* features;
};


CDenseFeatures<float64_t>* shogun::tapkee_embed(const shogun::TAPKEE_PARAMETERS_FOR_SHOGUN& parameters)
{
	tapkee::LoggingSingleton::instance().set_logger_impl(new ShogunLoggerImplementation);
	tapkee::LoggingSingleton::instance().enable_benchmark();
	tapkee::LoggingSingleton::instance().enable_info();

	pimpl_kernel_callback<CKernel> kernel_callback(parameters.kernel);
	pimpl_distance_callback<CDistance> distance_callback(parameters.distance);
	ShogunFeatureVectorCallback features_callback(parameters.features);

	tapkee::ParametersMap tapkee_parameters;
#ifdef HAVE_ARPACK
	tapkee_parameters[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee::ARPACK;
#else
	tapkee_parameters[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee::EIGEN_DENSE_SELFADJOINT_SOLVER;
#endif
	tapkee_parameters[tapkee::NEIGHBORS_METHOD] = tapkee::COVER_TREE;
	tapkee_parameters[tapkee::TARGET_DIMENSION] = static_cast<tapkee::IndexType>(parameters.target_dimension);
	tapkee_parameters[tapkee::EIGENSHIFT] = static_cast<tapkee::ScalarType>(parameters.eigenshift);
	tapkee_parameters[tapkee::CHECK_CONNECTIVITY] = true;
	tapkee_parameters[tapkee::OUTPUT_FEATURE_VECTORS_ARE_COLUMNS] = true;
	size_t N = 0;

	switch (parameters.method) 
	{
		case SHOGUN_KERNEL_LOCALLY_LINEAR_EMBEDDING:
		case SHOGUN_LOCALLY_LINEAR_EMBEDDING:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_NEIGHBORHOOD_PRESERVING_EMBEDDING:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			tapkee_parameters[tapkee::CURRENT_DIMENSION] = 
				static_cast<tapkee::IndexType>(parameters.features->get_dim_feature_space());
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_LOCAL_TANGENT_SPACE_ALIGNMENT:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			tapkee_parameters[tapkee::CURRENT_DIMENSION] = 
				static_cast<tapkee::IndexType>(parameters.features->get_dim_feature_space());
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			N = parameters.kernel->get_num_vec_lhs();
			break;
		case SHOGUN_DIFFUSION_MAPS:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::DIFFUSION_MAP;
			tapkee_parameters[tapkee::DIFFUSION_MAP_TIMESTEPS] =
				static_cast<tapkee::IndexType>(parameters.n_timesteps);
			tapkee_parameters[tapkee::GAUSSIAN_KERNEL_WIDTH] =
				static_cast<tapkee::ScalarType>(parameters.gaussian_kernel_width);
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LAPLACIAN_EIGENMAPS:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::LAPLACIAN_EIGENMAPS;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			tapkee_parameters[tapkee::GAUSSIAN_KERNEL_WIDTH] =
				static_cast<tapkee::ScalarType>(parameters.gaussian_kernel_width);
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LOCALITY_PRESERVING_PROJECTIONS:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::LOCALITY_PRESERVING_PROJECTIONS;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			tapkee_parameters[tapkee::GAUSSIAN_KERNEL_WIDTH] =
				static_cast<tapkee::ScalarType>(parameters.gaussian_kernel_width);
			tapkee_parameters[tapkee::CURRENT_DIMENSION] = 
				static_cast<tapkee::IndexType>(parameters.features->get_dim_feature_space());
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_MULTIDIMENSIONAL_SCALING:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::MULTIDIMENSIONAL_SCALING;
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LANDMARK_MULTIDIMENSIONAL_SCALING:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING;
			tapkee_parameters[tapkee::LANDMARK_RATIO] = 
				static_cast<tapkee::ScalarType>(parameters.landmark_ratio);
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_ISOMAP:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::ISOMAP;
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_LANDMARK_ISOMAP:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::LANDMARK_ISOMAP;
			tapkee_parameters[tapkee::LANDMARK_RATIO] = 
				static_cast<tapkee::ScalarType>(parameters.landmark_ratio);
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			N = parameters.distance->get_num_vec_lhs();
			break;
		case SHOGUN_STOCHASTIC_PROXIMITY_EMBEDDING:
			tapkee_parameters[tapkee::REDUCTION_METHOD] = 
				tapkee::STOCHASTIC_PROXIMITY_EMBEDDING;
			N = parameters.distance->get_num_vec_lhs();
			tapkee_parameters[tapkee::NUMBER_OF_NEIGHBORS] =
				static_cast<tapkee::IndexType>(parameters.n_neighbors);
			tapkee_parameters[tapkee::SPE_TOLERANCE] = 
				static_cast<tapkee::ScalarType>(parameters.spe_tolerance);
			tapkee_parameters[tapkee::SPE_NUM_UPDATES] = 
				static_cast<tapkee::IndexType>(parameters.spe_num_updates);
			tapkee_parameters[tapkee::MAX_ITERATION] =
				static_cast<tapkee::IndexType>(parameters.max_iteration);
			if (parameters.spe_global_strategy) 
				tapkee_parameters[tapkee::SPE_GLOBAL_STRATEGY] = true;
			else
				tapkee_parameters[tapkee::SPE_GLOBAL_STRATEGY] = false;
			break;
		case SHOGUN_FACTOR_ANALYSIS:
			tapkee_parameters[tapkee::CURRENT_DIMENSION] = 
				static_cast<tapkee::IndexType>(parameters.features->get_dim_feature_space());
			N = parameters.features->get_num_vectors();
			tapkee_parameters[tapkee::REDUCTION_METHOD] =
				tapkee::FACTOR_ANALYSIS;
			tapkee_parameters[tapkee::MAX_ITERATION] =
				static_cast<tapkee::IndexType>(parameters.max_iteration);
			tapkee_parameters[tapkee::FA_EPSILON] =
				static_cast<tapkee::ScalarType>(parameters.fa_epsilon);
			break;
	}
	
	std::vector<int32_t> indices(N);
	for (size_t i=0; i<N; i++)
		indices[i] = i;

	tapkee::ReturnResult result = tapkee::embed(indices.begin(),indices.end(),
			kernel_callback,distance_callback,features_callback,tapkee_parameters);
	tapkee::DenseMatrix result_embedding = result.first;
	// destroy projecting function
	result.second.clear();

	SGMatrix<float64_t> feature_matrix(parameters.target_dimension,N);
	// TODO avoid copying
	for (uint32_t i=0; i<N; i++) 
	{
		for (uint32_t j=0; j<parameters.target_dimension; j++) 
		{
			feature_matrix(j,i) = result_embedding(j,i);
		}
	}
	return new CDenseFeatures<float64_t>(feature_matrix);
}

#endif
