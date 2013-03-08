/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando J. Iglesias García
 *
 * This code uses Any type developed by C. Diggins under Boost license, version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 */

#ifndef TAPKEE_DEFINES_H_
#define TAPKEE_DEFINES_H_

#include <shogun/lib/tapkee/tapkee_exceptions.hpp>
#include <shogun/lib/tapkee/utils/any.hpp>
#include <shogun/lib/tapkee/callbacks/traits.hpp>
#include <shogun/lib/tapkee/routines/methods_traits.hpp>

#include <map>
#include <vector>
#include <utility>

/// Eigen 3 library includes

#ifdef TAPKEE_EIGEN_INCLUDE_FILE
	#include TAPKEE_EIGEN_INCLUDE_FILE
#else 
	#ifndef TAPKEE_DEBUG
		#define EIGEN_NO_DEBUG
	#endif
	#define EIGEN_RUNTIME_NO_MALLOC
	#include <Eigen/Eigen>
	#include <Eigen/Dense>
	#if EIGEN_VERSION_AT_LEAST(3,1,0)
		#include <Eigen/Sparse>
		#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
			#include <Eigen/SuperLUSupport>
		#endif
	#else
		#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		#include <unsupported/Eigen/SparseExtra>
	#endif
#endif

#ifdef EIGEN_RUNTIME_NO_MALLOC
	#define RESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(false);
	#define UNRESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(true);
#else
	#define RESTRICT_ALLOC
	#define UNRESTRICT_ALLOC
#endif

/// end of Eigen 3 library includes

namespace tapkee 
{
	//! Parameters that are used by the library
	enum TAPKEE_PARAMETERS
	{
		/* TAPKEE_METHOD */	REDUCTION_METHOD,
		/* IndexType */ NUMBER_OF_NEIGHBORS,
		/* IndexType */ TARGET_DIMENSION,
		/* IndexType */ CURRENT_DIMENSION,
		/* TAPKEE_EIGEN_EMBEDDING_METHOD */ EIGEN_EMBEDDING_METHOD,
		/* TAPKEE_NEIGHBORS_METHOD */ NEIGHBORS_METHOD,
		/* IndexType */ DIFFUSION_MAP_TIMESTEPS,
		/* ScalarType */ GAUSSIAN_KERNEL_WIDTH,
		/* IndexType */ MAX_ITERATION,
		/* bool */ SPE_GLOBAL_STRATEGY,
		/* IndexType */ SPE_NUM_UPDATES,
		/* ScalarType */ SPE_TOLERANCE,
		/* ScalarType */ LANDMARK_RATIO,
		/* ScalarType */ EIGENSHIFT,
		/* bool */ CHECK_CONNECTIVITY,
		/* ScalarType */ FA_EPSILON,
		/* ScalarType */ SNE_PERPLEXITY,
		/* ScalarType */ SNE_THETA
	};

	//! Dimension reduction method
	//! All methods require 
	enum TAPKEE_METHOD
	{
		KERNEL_LOCALLY_LINEAR_EMBEDDING,
		NEIGHBORHOOD_PRESERVING_EMBEDDING,
		KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT,
		LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT,
		HESSIAN_LOCALLY_LINEAR_EMBEDDING,
		LAPLACIAN_EIGENMAPS,
		LOCALITY_PRESERVING_PROJECTIONS,
		DIFFUSION_MAP,
		ISOMAP,
		LANDMARK_ISOMAP,
		MULTIDIMENSIONAL_SCALING,
		LANDMARK_MULTIDIMENSIONAL_SCALING,
		STOCHASTIC_PROXIMITY_EMBEDDING,
		KERNEL_PCA,
		PCA,
		RANDOM_PROJECTION,
		PASS_THRU,
		FACTOR_ANALYSIS,
		TSNE,
		UNKNOWN_METHOD
	};

	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KERNEL_LOCALLY_LINEAR_EMBEDDING);
	METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(NEIGHBORHOOD_PRESERVING_EMBEDDING);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT);
	METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(HESSIAN_LOCALLY_LINEAR_EMBEDDING);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LAPLACIAN_EIGENMAPS);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(LOCALITY_PRESERVING_PROJECTIONS);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(DIFFUSION_MAP);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(ISOMAP);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LANDMARK_ISOMAP);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(MULTIDIMENSIONAL_SCALING);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LANDMARK_MULTIDIMENSIONAL_SCALING);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(STOCHASTIC_PROXIMITY_EMBEDDING);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KERNEL_PCA);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(PCA);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(RANDOM_PROJECTION);
	METHOD_THAT_NEEDS_NOTHING_IS(PASS_THRU);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(FACTOR_ANALYSIS);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(TSNE);
	METHOD_THAT_NEEDS_NOTHING_IS(UNKNOWN_METHOD);

	//! Neighbors computation method
	enum TAPKEE_NEIGHBORS_METHOD
	{
		//! Brute force method with not least than 
		//! \f$ O(N N \log k) \f$ time complexity.
		//! Recommended to be used only in debug purposes.
		BRUTE_FORCE,
		//! Covertree-based method with \f$ O(\log N) \f$ time complexity.
		//! Recommended to be used as a default method.
		COVER_TREE,
		UNKNOWN_NEIGHBORS_METHOD
	};

	//! Eigendecomposition-based embedding methods enumeration
	enum TAPKEE_EIGEN_EMBEDDING_METHOD
	{
		//! ARPACK-based method (requires the ARPACK library
		//! binaries to be available around). Recommended to be used as a 
		//! default method. Supports both generalized and standard eigenproblems.
		ARPACK,
		//! Randomized method (implementation taken from the redsvd lib). 
		//! Supports only standard but not generalized eigenproblems.
		RANDOMIZED,
		//! Eigen library dense method (useful for debugging). Computes
		//! all eigenvectors thus can be very slow doing large-scale.
		EIGEN_DENSE_SELFADJOINT_SOLVER,
		UNKNOWN_EIGEN_METHOD
	};

#ifdef TAPKEE_CUSTOM_INTERNAL_NUMTYPE
	typedef TAPKEE_CUSTOM_INTERNAL_NUMTYPE ScalarType;
#else
	//! default scalar value (currently only double is supported and tested, float is unstable)
	typedef double ScalarType;
#endif
	typedef unsigned int IndexType;
	//! dense vector type 
	typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> DenseVector;
	//! dense matrix type
	typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;
	//! dense symmetric matrix (currently just dense matrix, can be improved later)
	typedef DenseMatrix DenseSymmetricMatrix;
	//! sparse weight matrix type
	typedef Eigen::SparseMatrix<ScalarType> SparseWeightMatrix;
	//! default selfadjoint solver
	typedef Eigen::SelfAdjointEigenSolver<DenseMatrix> DenseSelfAdjointEigenSolver;
	//! default dense solver
	typedef Eigen::LDLT<DenseMatrix> DenseSolver;
	//! default sparse solver
#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	typedef Eigen::SimplicialCholesky<SparseWeightMatrix> SparseSolver;
#else
	#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
		typedef Eigen::SuperLU<SparseWeightMatrix> SparseSolver;
	#else
		typedef Eigen::SimplicialLDLT<SparseWeightMatrix> SparseSolver;
	#endif
#endif

#ifdef TAPKEE_CUSTOM_PROPERTIES
	#include TAPKEE_CUSTOM_PROPERTIES
#else
	#define COVERTREE_BASE 1.3
#endif

// Internal types (can be overrided)
#ifndef TAPKEE_INTERNAL_VECTOR
	#define TAPKEE_INTERNAL_VECTOR std::vector
#endif
#ifndef TAPKEE_INTERNAL_PAIR
	#define TAPKEE_INTERNAL_PAIR std::pair
#endif
#ifndef TAPKEE_INTERNAL_MAP
	#define TAPKEE_INTERNAL_MAP std::map
#endif

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	namespace tapkee {
	template <typename T>
	struct Triplet
	{
		Triplet(IndexType colIndex, unsigned int rowIndex, T valueT) : 
			col_(colIndex), row_(rowIndex), value_(valueT)
		{
		}
		IndexType col() const { return col_; };
		IndexType row() const { return row_; };
		T value() const { return value_; };
		IndexType col_;
		IndexType row_;
		T value_;
	};
	};
	typedef tapkee::Triplet<ScalarType> SparseTriplet;
#else
	typedef Eigen::Triplet<ScalarType> SparseTriplet;
#endif

typedef TAPKEE_INTERNAL_MAP<TAPKEE_PARAMETERS, any> ParametersMap;
typedef TAPKEE_INTERNAL_VECTOR<SparseTriplet> SparseTriplets;
typedef TAPKEE_INTERNAL_VECTOR<IndexType> LocalNeighbors;
typedef TAPKEE_INTERNAL_VECTOR<LocalNeighbors> Neighbors;
typedef TAPKEE_INTERNAL_PAIR<DenseMatrix,DenseVector> EmbeddingResult;
typedef Eigen::DiagonalMatrix<ScalarType,Eigen::Dynamic> DenseDiagonalMatrix;
typedef TAPKEE_INTERNAL_VECTOR<IndexType> Landmarks;
typedef TAPKEE_INTERNAL_PAIR<SparseWeightMatrix,DenseDiagonalMatrix> Laplacian;
typedef TAPKEE_INTERNAL_PAIR<DenseSymmetricMatrix,DenseSymmetricMatrix> DenseSymmetricMatrixPair;

#include <shogun/lib/tapkee/tapkee_projection.hpp>
typedef TAPKEE_INTERNAL_PAIR<DenseMatrix,tapkee::ProjectingFunction> ReturnResult;

} // namespace tapkee

#endif
