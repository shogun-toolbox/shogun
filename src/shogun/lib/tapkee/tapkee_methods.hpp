/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

#include <shogun/lib/tapkee/tapkee_defines.hpp>
#include <shogun/lib/tapkee/routines/locally_linear.hpp>
#include <shogun/lib/tapkee/routines/eigen_embedding.hpp>
#include <shogun/lib/tapkee/routines/generalized_eigen_embedding.hpp>
#include <shogun/lib/tapkee/routines/multidimensional_scaling.hpp>
#include <shogun/lib/tapkee/routines/diffusion_maps.hpp>
#include <shogun/lib/tapkee/routines/laplacian_eigenmaps.hpp>
#include <shogun/lib/tapkee/routines/isomap.hpp>
#include <shogun/lib/tapkee/routines/pca.hpp>
#include <shogun/lib/tapkee/routines/random_projection.hpp>
#include <shogun/lib/tapkee/routines/spe.hpp>
#include <shogun/lib/tapkee/routines/fa.hpp>
#include <shogun/lib/tapkee/neighbors/neighbors.hpp>

namespace tapkee
{
namespace tapkee_internal
{

std::string get_method_name(TAPKEE_METHOD m)
{
	switch (m)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING: return "Locally Linear Embedding";
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT: return "Local Tangent Space Alignment";
		case DIFFUSION_MAP: return "Diffusion Map";
		case MULTIDIMENSIONAL_SCALING: return "Classic MultiDimensional Scaling";
		case LANDMARK_MULTIDIMENSIONAL_SCALING: return "Landmark MultiDimensional Scaling";
		case ISOMAP: return "Isomap";
		case LANDMARK_ISOMAP: return "Landmark Isomap";
		case NEIGHBORHOOD_PRESERVING_EMBEDDING: return "Neighborhood Preserving Embedding";
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT: return "Linear Local Tangent Space Alignment";
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING: return "Hessian Locally Linear Embedding";
		case LAPLACIAN_EIGENMAPS: return "Laplacian Eigenmaps";
		case LOCALITY_PRESERVING_PROJECTIONS: return "Locality Preserving Embedding";
		case PCA: return "Principal Component Analysis";
		case KERNEL_PCA: return "Kernel Principal Component Analysis";
		case STOCHASTIC_PROXIMITY_EMBEDDING: return "Stochastic Proximity Embedding";
		case PASS_THRU: return "passing through";
		case FACTOR_ANALYSIS: return "Factor Analysis";
		default: return "Method name unknown (yes this is a bug)";
	}
}

template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback, int IMPLEMENTATION>
struct embedding_impl
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback kernel_callback, DistanceCallback distance_callback,
                       FeatureVectorCallback feature_vector_callback, ParametersMap options);
};

#define CONCRETE_IMPLEMENTATION(METHOD) \
	template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback> \
	struct embedding_impl<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,METHOD>
#define OBTAIN_PARAMETER(TYPE,NAME,CODE) \
	if (!options.count(CODE)) \
	{ \
		LoggingSingleton::instance().message_error("No "#NAME" ("#TYPE") parameter set. Should be in map as "#CODE); \
		return ReturnResult(); \
	} \
	TYPE NAME = options[CODE].cast<TYPE>()
#define SKIP_ONE_EIGENVALUE 1
#define SKIP_NO_EIGENVALUES 0

CONCRETE_IMPLEMENTATION(KERNEL_LOCALLY_LINEAR_EMBEDDING)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(DefaultScalarType,eigenshift,EIGENSHIFT);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);

		timed_context context("Embedding with KLLE");
		Neighbors neighbors =
			find_neighbors(neighbors_method,begin,end,kernel_callback,k,check_connectivity);
		SparseWeightMatrix weight_matrix =
			linear_weight_matrix(begin,end,neighbors,kernel_callback,eigenshift);
		return ReturnResult(eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(DefaultScalarType,eigenshift,EIGENSHIFT);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		
		timed_context context("Embedding with KLTSA");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,kernel_callback,k,check_connectivity);
		SparseWeightMatrix weight_matrix = 
			tangent_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension,eigenshift);
		return ReturnResult(eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(DIFFUSION_MAP)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(unsigned int,timesteps,DIFFUSION_MAP_TIMESTEPS);
		OBTAIN_PARAMETER(DefaultScalarType,width,GAUSSIAN_KERNEL_WIDTH);
		
		timed_context context("Embedding with diffusion map");
		DenseSymmetricMatrix diffusion_matrix =
			compute_diffusion_matrix(begin,end,distance_callback,timesteps,width);
		return ReturnResult(eigen_embedding<DenseSymmetricMatrix,
				#ifdef TAPKEE_GPU
					GPUDenseImplicitSquareMatrixOperation
				#else 
					DenseImplicitSquareMatrixOperation 
				#endif
				>(eigen_method,
			diffusion_matrix,target_dimension,SKIP_NO_EIGENVALUES).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(MULTIDIMENSIONAL_SCALING)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);

		timed_context context("Embedding with MDS");
		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance_callback);
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EmbeddingResult result = eigen_embedding<DenseSymmetricMatrix,
				#ifdef TAPKEE_GPU
						GPUDenseMatrixOperation
				#else
						DenseMatrixOperation
				#endif
				>(eigen_method,
			distance_matrix,target_dimension,SKIP_NO_EIGENVALUES);

		for (unsigned int i=0; i<target_dimension; i++)
			result.first.col(i).array() *= sqrt(result.second(i));
		return ReturnResult(result.first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(LANDMARK_MULTIDIMENSIONAL_SCALING)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(DefaultScalarType,ratio,LANDMARK_RATIO);

		timed_context context("Embedding with Landmark MDS");
		Landmarks landmarks = 
			select_landmarks_random(begin,end,ratio);
		DenseSymmetricMatrix distance_matrix = 
			compute_distance_matrix(begin,landmarks,distance_callback);
		DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EmbeddingResult landmarks_embedding = 
			eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
					distance_matrix,target_dimension,SKIP_NO_EIGENVALUES);
		for (unsigned int i=0; i<target_dimension; i++)
			landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
		return ReturnResult(triangulate(begin,end,distance_callback,landmarks,
			landmark_distances_squared,landmarks_embedding,target_dimension).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(ISOMAP)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);

		timed_context context("Embedding with Isomap");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,distance_callback,k,check_connectivity);
		DenseSymmetricMatrix shortest_distances_matrix = 
			compute_shortest_distances_matrix(begin,end,neighbors,distance_callback);
		shortest_distances_matrix = shortest_distances_matrix.array().square();
		centerMatrix(shortest_distances_matrix);
		shortest_distances_matrix.array() *= -0.5;
		
		EmbeddingResult embedding = eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
			shortest_distances_matrix,target_dimension,SKIP_NO_EIGENVALUES);

		for (unsigned int i=0; i<target_dimension; i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		
		return ReturnResult(embedding.first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(LANDMARK_ISOMAP)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(DefaultScalarType,ratio,LANDMARK_RATIO);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);

		timed_context context("Embedding with Landmark Isomap");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,distance_callback,k,check_connectivity);
		Landmarks landmarks = 
			select_landmarks_random(begin,end,ratio);
		DenseMatrix distance_matrix = 
			compute_shortest_distances_matrix(begin,end,landmarks,neighbors,distance_callback);
		distance_matrix = distance_matrix.array().square();
		
		DenseVector col_means = distance_matrix.colwise().mean();
		DenseVector row_means = distance_matrix.rowwise().mean();
		DefaultScalarType grand_mean = distance_matrix.mean();
		distance_matrix.array() += grand_mean;
		distance_matrix.colwise() -= row_means;
		distance_matrix.rowwise() -= col_means.transpose();
		distance_matrix.array() *= -0.5;

		EmbeddingResult landmarks_embedding = 
			eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
					distance_matrix*distance_matrix.transpose(),target_dimension,SKIP_NO_EIGENVALUES);

		DenseMatrix embedding = distance_matrix.transpose()*landmarks_embedding.first;

		for (unsigned int i=0; i<target_dimension; i++)
			embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
		return ReturnResult(embedding,tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(NEIGHBORHOOD_PRESERVING_EMBEDDING)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback kernel_callback, DistanceCallback,
                       FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(unsigned int,dimension,CURRENT_DIMENSION);
		OBTAIN_PARAMETER(DefaultScalarType,eigenshift,EIGENSHIFT);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		
		timed_context context("Embedding with NPE");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,kernel_callback,k,check_connectivity);
		SparseWeightMatrix weight_matrix = 
			linear_weight_matrix(begin,end,neighbors,kernel_callback,eigenshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
				feature_vector_callback,dimension);
		ProjectionResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseMatrixOperation>(
				eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SKIP_NO_EIGENVALUES);
		DenseVector mean_vector = 
			compute_mean(begin,end,feature_vector_callback,dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,feature_vector_callback,dimension),projecting_function);
	}
};

CONCRETE_IMPLEMENTATION(HESSIAN_LOCALLY_LINEAR_EMBEDDING)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback kernel_callback, DistanceCallback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		
		timed_context context("Embedding with HLLE");
		Neighbors neighbors =
			find_neighbors(neighbors_method,begin,end,kernel_callback,k,check_connectivity);
		SparseWeightMatrix weight_matrix =
			hessian_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension);
		return ReturnResult(eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(LAPLACIAN_EIGENMAPS)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback distance_callback,
                          FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(DefaultScalarType,width,GAUSSIAN_KERNEL_WIDTH);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		
		timed_context context("Embedding with Laplacian Eigenmaps");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,distance_callback,k,check_connectivity);
		Laplacian laplacian = 
			compute_laplacian(begin,end,neighbors,distance_callback,width);
		return ReturnResult(generalized_eigen_embedding<SparseWeightMatrix,DenseSymmetricMatrix,InverseSparseMatrixOperation>(
			eigen_method,laplacian.first,laplacian.second,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(LOCALITY_PRESERVING_PROJECTIONS)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(DefaultScalarType,width,GAUSSIAN_KERNEL_WIDTH);
		OBTAIN_PARAMETER(unsigned int,dimension,CURRENT_DIMENSION);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		
		timed_context context("Embedding with LPP");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,distance_callback,k,check_connectivity);
		Laplacian laplacian = 
			compute_laplacian(begin,end,neighbors,distance_callback,width);
		DenseSymmetricMatrixPair eigenproblem_matrices =
			construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
					feature_vector_callback,dimension);
		ProjectionResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseMatrixOperation>(
				eigen_method,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension,SKIP_NO_EIGENVALUES);
		DenseVector mean_vector = 
			compute_mean(begin,end,feature_vector_callback,dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,feature_vector_callback,dimension),projecting_function);
	}
};

CONCRETE_IMPLEMENTATION(PCA)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback,
                          FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,dimension,CURRENT_DIMENSION);
		
		timed_context context("Embedding with PCA");
		DenseVector mean_vector = 
			compute_mean(begin,end,feature_vector_callback,dimension);
		DenseSymmetricMatrix centered_covariance_matrix = 
			compute_covariance_matrix(begin,end,mean_vector,feature_vector_callback,dimension);
		ProjectionResult projection_result = 
			eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_covariance_matrix,target_dimension,SKIP_NO_EIGENVALUES);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,feature_vector_callback,dimension), projecting_function);
	}
};

CONCRETE_IMPLEMENTATION(RANDOM_PROJECTION)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
	                   KernelCallback, DistanceCallback, FeatureVectorCallback feature_vector_callback,
	                   ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(unsigned int,dimension,CURRENT_DIMENSION);

		timed_context context("Embedding with Random Projection");

		DenseMatrix projection_matrix = 
			gaussian_projection_matrix(dimension, target_dimension);

		DenseVector mean_vector = 
			compute_mean(begin,end,feature_vector_callback,dimension);

		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_matrix,mean_vector));

		return ReturnResult(project(projection_matrix,mean_vector,begin,end,feature_vector_callback,dimension), projecting_function);
	}
};

CONCRETE_IMPLEMENTATION(KERNEL_PCA)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);

		timed_context context("Embedding with kPCA");
		DenseSymmetricMatrix centered_kernel_matrix = 
			compute_centered_kernel_matrix(begin,end,kernel_callback);
		return ReturnResult(eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
			centered_kernel_matrix,target_dimension,SKIP_NO_EIGENVALUES).first, tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback kernel_callback, DistanceCallback,
                       FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD,eigen_method,EIGEN_EMBEDDING_METHOD);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(unsigned int,dimension,CURRENT_DIMENSION);
		OBTAIN_PARAMETER(DefaultScalarType,eigenshift,EIGENSHIFT);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		
		timed_context context("Embedding with LLTSA");
		Neighbors neighbors = 
			find_neighbors(neighbors_method,begin,end,kernel_callback,k,check_connectivity);
		SparseWeightMatrix weight_matrix = 
			tangent_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension,eigenshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_lltsa_eigenproblem(weight_matrix,begin,end,
				feature_vector_callback,dimension);
		ProjectionResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseMatrixOperation>(
				eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SKIP_NO_EIGENVALUES);
		DenseVector mean_vector = 
			compute_mean(begin,end,feature_vector_callback,dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,feature_vector_callback,dimension),
				projecting_function);
	}
};

CONCRETE_IMPLEMENTATION(STOCHASTIC_PROXIMITY_EMBEDDING)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback distance_callback,
                       FeatureVectorCallback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(unsigned int,k,NUMBER_OF_NEIGHBORS);
		OBTAIN_PARAMETER(TAPKEE_NEIGHBORS_METHOD,neighbors_method,NEIGHBORS_METHOD);
		OBTAIN_PARAMETER(bool,global_strategy,SPE_GLOBAL_STRATEGY);
		OBTAIN_PARAMETER(DefaultScalarType,tolerance,SPE_TOLERANCE);
		OBTAIN_PARAMETER(unsigned int,nupdates,SPE_NUM_UPDATES);
		OBTAIN_PARAMETER(bool,check_connectivity,CHECK_CONNECTIVITY);
		OBTAIN_PARAMETER(unsigned int,max_iteration,MAX_ITERATION);

		Neighbors neighbors;
		if (!global_strategy)
		{
			neighbors = find_neighbors(neighbors_method,begin,end,distance_callback,k,check_connectivity);
		}

		timed_context context("Embedding with SPE");
		return ReturnResult(spe_embedding(begin,end,distance_callback,neighbors,
				target_dimension,global_strategy,tolerance,nupdates,max_iteration), tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(PASS_THRU)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback, FeatureVectorCallback feature_callback, 
                       ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,dimension,CURRENT_DIMENSION);

		DenseMatrix feature_matrix(dimension,(end-begin));
		DenseVector feature_vector(dimension);
		for (RandomAccessIterator iter=begin; iter!=end; ++iter)
		{
			feature_callback(*iter,feature_vector);
			feature_matrix.col(iter-begin).array() = feature_vector;
		}
		return ReturnResult(feature_matrix.transpose(),tapkee::ProjectingFunction());
	}
};

CONCRETE_IMPLEMENTATION(FACTOR_ANALYSIS)
{
	ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                       KernelCallback, DistanceCallback,
                       FeatureVectorCallback callback, ParametersMap options)
	{
		OBTAIN_PARAMETER(unsigned int,current_dimension,CURRENT_DIMENSION);
		OBTAIN_PARAMETER(unsigned int,target_dimension,TARGET_DIMENSION);
		OBTAIN_PARAMETER(unsigned int,max_iteration,MAX_ITERATION);
		OBTAIN_PARAMETER(DefaultScalarType,epsilon,FA_EPSILON);

		timed_context context("Embedding with FA");
		DenseVector mean_vector = compute_mean(begin,end,callback,current_dimension);
		return ReturnResult(project(begin,end,callback,current_dimension,max_iteration,epsilon,
                                    target_dimension, mean_vector), tapkee::ProjectingFunction());
	}
};

}
}
#undef CONCRETE_IMPLEMENTATION
#undef OBTAIN_PARAMETER
#undef SKIP_ONE_EIGENVALUE
#undef SKIP_NO_EIGENVALUES
#endif
