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
#include <shogun/lib/tapkee/callbacks/pimpl_callbacks.hpp>

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
	ShogunFeatureVectorCallback(CDotFeatures* f) : dim(0), features(f) { }
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
	CDotFeatures* features;
};

void prepare_tapkee_parameters_set(const TAPKEE_PARAMETERS_FOR_SHOGUN& parameters, tapkee::ParametersSet& parameters_set, 	std::vector<int32_t>& indices)
{	tapkee::LoggingSingleton::instance().set_logger_impl(new ShogunLoggerImplementation);
	tapkee::LoggingSingleton::instance().enable_benchmark();
	tapkee::LoggingSingleton::instance().enable_info();

	tapkee::DimensionReductionMethod method;
#ifdef HAVE_ARPACK
	tapkee::EigenMethod eigen_method = tapkee::Arpack;
#else
	tapkee::EigenMethod eigen_method = tapkee::Dense;
#endif
	tapkee::NeighborsMethod neighbors_method = tapkee::CoverTree;
	size_t N = 0;
	uint32_t num_neighbors = parameters.n_neighbors;
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

                case SHOGUN_KERNEL_PCA:
                        method = tapkee::KernelPCA;
                        N = parameters.kernel->get_num_vec_lhs();
                        num_neighbors = N-1;//KPCA doesn't need it.
                        break;

		case SHOGUN_TDISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING:
			method = tapkee::tDistributedStochasticNeighborEmbedding;
			N = parameters.features->get_num_vectors();
			break;

	}

	std::vector<int32_t> tmp_indices(N);
	for (size_t i=0; i<N; i++)
		tmp_indices[i] = i;
        indices = tmp_indices;
	parameters_set = 
		(tapkee::keywords::method=method,
		 tapkee::keywords::eigen_method=eigen_method,
		 tapkee::keywords::neighbors_method=neighbors_method,
		 tapkee::keywords::num_neighbors=num_neighbors,
		 tapkee::keywords::diffusion_map_timesteps = parameters.n_timesteps,
		 tapkee::keywords::target_dimension = parameters.target_dimension,
		 tapkee::keywords::spe_num_updates = parameters.spe_num_updates,
		 tapkee::keywords::nullspace_shift = parameters.eigenshift,
		 tapkee::keywords::landmark_ratio = parameters.landmark_ratio,
		 tapkee::keywords::gaussian_kernel_width = parameters.gaussian_kernel_width,
		 tapkee::keywords::spe_tolerance = parameters.spe_tolerance,
		 tapkee::keywords::spe_global_strategy = parameters.spe_global_strategy,
		 tapkee::keywords::max_iteration = parameters.max_iteration,
		 tapkee::keywords::fa_epsilon = parameters.fa_epsilon,
		 tapkee::keywords::sne_perplexity = parameters.sne_perplexity,
		 tapkee::keywords::sne_theta = parameters.sne_theta
		 );

}

CDenseFeatures<float64_t>* shogun::tapkee_embed(const shogun::TAPKEE_PARAMETERS_FOR_SHOGUN& parameters)
{
        tapkee::ParametersSet parameters_set;
        std::vector<int32_t> indices;
        pimpl_kernel_callback<CKernel> kernel_callback(parameters.kernel);
	pimpl_distance_callback<CDistance> distance_callback(parameters.distance);
	ShogunFeatureVectorCallback features_callback(parameters.features);
        prepare_tapkee_parameters_set(parameters, parameters_set, indices);
	tapkee::TapkeeOutput output = 
			tapkee::embed(indices.begin(), indices.end(),
			kernel_callback,distance_callback,features_callback,parameters_set);
	tapkee::DenseMatrix result_embedding = output.embedding;
	// destroy projecting function
	output.projection.clear();
	size_t N = indices.size();
	SGMatrix<float64_t> feature_matrix(parameters.target_dimension,N);
	// TODO avoid copying
	for (uint32_t i=0; i<N; i++) 
	{
		for (uint32_t j=0; j<parameters.target_dimension; j++) 
		{
			feature_matrix(j,i) = result_embedding(i,j);
		}
	}
	return new CDenseFeatures<float64_t>(feature_matrix);
}


void shogun::ProjectingFunction::clear()
{
	if ( m_tapkee_projecting_function )
	{
		m_tapkee_projecting_function->clear();
		m_tapkee_projecting_function = NULL;
	}
}

CDenseFeatures<float64_t>* shogun::ProjectingFunction::operator () ( CDenseFeatures<float64_t>* features)
{
	ASSERT(m_tapkee_projecting_function)
	size_t num_features = features->get_num_features();
	size_t num_vectors  = features->get_num_vectors();
	Eigen::Map<tapkee::DenseMatrix> tmp_matrix( (features->get_feature_matrix()).matrix,
	                                    num_features, num_vectors);
	tapkee::DenseVector tt = tmp_matrix.col(0);
	
	tapkee::DenseVector tmp_vec = (*m_tapkee_projecting_function)(tt); return NULL;
	size_t target_dim = tmp_vec.rows();
	float64_t* new_feature_matrix = SG_MALLOC(float64_t, target_dim*num_vectors);
	size_t i,j;
	for (j = 0;j<target_dim; ++j)
		new_feature_matrix[j] = tmp_vec(0,j);
	for (i = 1;i<num_vectors; ++i)
	{
		tmp_vec = (*m_tapkee_projecting_function)(tmp_matrix.col(i));
		for (j = 0; j<target_dim; ++j)
			new_feature_matrix[i * target_dim + j] = tmp_vec(0,j);
	}
	CDenseFeatures<float64_t>* result_features = new CDenseFeatures<float64_t>();
	result_features->set_feature_matrix(SGMatrix<float64_t>(new_feature_matrix, target_dim,num_vectors));
	return result_features;
}

shogun::ProjectingFunction shogun::calc_projecting_function(const TAPKEE_PARAMETERS_FOR_SHOGUN& parameters)
{

	tapkee::ParametersSet parameters_set;
	std::vector<int32_t> indices;
	pimpl_kernel_callback<CKernel> kernel_callback(parameters.kernel);
	pimpl_distance_callback<CDistance> distance_callback(parameters.distance);
	ShogunFeatureVectorCallback features_callback(parameters.features);
	prepare_tapkee_parameters_set(parameters, parameters_set, indices);
	tapkee::TapkeeOutput output = tapkee::embed(indices.begin(),indices.end(),
			kernel_callback,distance_callback,features_callback,parameters_set);
	shogun::ProjectingFunction res;
	res.m_tapkee_projecting_function = new tapkee::ProjectingFunction( output.projection.implementation );
	 
	return res;
}




#endif
