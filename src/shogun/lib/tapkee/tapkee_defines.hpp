/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * This code also uses Any type developed by C. Diggins under Boost license, version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_H_
#define TAPKEE_DEFINES_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_exceptions.hpp>
#include <shogun/lib/tapkee/utils/any.hpp>
#include <shogun/lib/tapkee/callback/callback_traits.hpp>
#include <shogun/lib/tapkee/routines/methods_traits.hpp>
/* End of Tapkee includes */

#include <map>
#include <vector>
#include <utility>

//// Eigen 3 library includes

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

//// end of Eigen 3 library includes

//! Main namespace of the library, contains all public API definitions
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
#ifdef TAPKEE_USE_GPL_TSNE
		/* ScalarType */ SNE_PERPLEXITY,
		/* ScalarType */ SNE_THETA
#endif
	};


	//! Dimension reduction methods
	enum TAPKEE_METHOD
	{
		/** Kernel Locally Linear Embedding as described in \cite Decoste2001 */
		KERNEL_LOCALLY_LINEAR_EMBEDDING,
		/** Neighborhood Preserving Embedding as described in \cite He2005 */
		NEIGHBORHOOD_PRESERVING_EMBEDDING,
		/** Local Tangent Space Alignment as described in \cite Zhang2002 */
		KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT,
		/** Linear Local Tangent Space Alignment as described in \cite Zhang2007 */
		LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT,
		/** Hessian Locally Linear Embedding as described in \cite Donoho2003 */
		HESSIAN_LOCALLY_LINEAR_EMBEDDING,
		/** Laplacian Eigenmaps as described in \cite Belkin2002 */
		LAPLACIAN_EIGENMAPS,
		/** Locality Preserving Projections as described in \cite He2003 */
		LOCALITY_PRESERVING_PROJECTIONS,
		/** Diffusion map as described in \cite Coifman2006 */
		DIFFUSION_MAP,
		/** Isomap as described in \cite Tenenbaum2000 */
		ISOMAP,
		/** Landmark Isomap as described in \cite deSilva2002 */
		LANDMARK_ISOMAP,
		/** Multidimensional scaling as described in \cite Cox2000 */
		MULTIDIMENSIONAL_SCALING,
		/** Landmark multidimensional scaling as described in \cite deSilva2004 */
		LANDMARK_MULTIDIMENSIONAL_SCALING,
		/** Stochastic Proximity Embedding as described in \cite Agrafiotis2003 */
		STOCHASTIC_PROXIMITY_EMBEDDING,
		/** Kernel PCA as described in \cite Scholkopf1997 */
		KERNEL_PCA,
		/** Principal Component Analysis */
		PCA,
		/** Random Projection \cite Kaski1998*/
		RANDOM_PROJECTION,
		/** Factor Analysis */
		FACTOR_ANALYSIS,
#ifdef TAPKEE_USE_GPL_TSNE
		/** t-SNE and Barnes-Hut-SNE as described in \cite tSNE and \cite Barnes-Hut-SNE */
		TSNE,
#endif
		/** Passing through (doing nothing just passed data through) */
		PASS_THRU,
		/** unknown method */
		UNKNOWN_METHOD
	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#ifdef TAPKEE_USE_GPL_TSNE
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(TSNE);
#endif
	METHOD_THAT_NEEDS_NOTHING_IS(UNKNOWN_METHOD);
#endif

	//! Neighbors computation methods
	enum TAPKEE_NEIGHBORS_METHOD
	{
		//! Brute force method with not least than 
		//! \f$ O(N N \log k) \f$ time complexity.
		//! Recommended to be used only in debug purposes.
		BRUTE_FORCE,
#ifdef TAPKEE_USE_LGPL_COVERTREE
		//! Covertree-based method with approximate \f$ O(\log N) \f$ time complexity.
		//! Recommended to be used as a default method.
		COVER_TREE,
#endif
		//! Unknown method
		UNKNOWN_NEIGHBORS_METHOD
	};

	//! Eigendecomposition methods
	enum TAPKEE_EIGEN_EMBEDDING_METHOD
	{
#ifdef TAPKEE_WITH_ARPACK
		//! ARPACK-based method (requires the ARPACK library
		//! binaries to be available around). Recommended to be used as a 
		//! default method. Supports both generalized and standard eigenproblems.
		ARPACK,
#endif
		//! Randomized method (implementation taken from the redsvd lib). 
		//! Supports only standard but not generalized eigenproblems.
		RANDOMIZED,
		//! Eigen library dense method (could be useful for debugging). Computes
		//! all eigenvectors thus can be very slow doing large-scale.
		EIGEN_DENSE_SELFADJOINT_SOLVER,
		//! Unknown method
		UNKNOWN_EIGEN_METHOD
	};


#ifdef TAPKEE_CUSTOM_INTERNAL_NUMTYPE
	typedef TAPKEE_CUSTOM_INTERNAL_NUMTYPE ScalarType;
#else
	//! default scalar value (can be overrided with TAPKEE_CUSTOM_INTERNAL_NUMTYPE define)
	typedef double ScalarType;
#endif
	//! indexing type (non-overridable)
	//! set to int for compatibility with OpenMP 2.0
	typedef int IndexType;
	//! dense vector type (non-overridable)
	typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> DenseVector;
	//! dense matrix type (non-overridable) 
	typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;
	//! dense symmetric matrix (non-overridable, currently just dense matrix, can be improved later)
	typedef DenseMatrix DenseSymmetricMatrix;
	//! sparse weight matrix type (non-overridable)
	typedef Eigen::SparseMatrix<ScalarType> SparseWeightMatrix;
	//! selfadjoint solver (non-overridable)
	typedef Eigen::SelfAdjointEigenSolver<DenseMatrix> DenseSelfAdjointEigenSolver;
	//! dense solver (non-overridable)
	typedef Eigen::LDLT<DenseMatrix> DenseSolver;
#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	//! sparse solver (it is Eigen::SimplicialCholesky in case of eigen version <3.1.0,
	//! in case of TAPKEE_USE_SUPERLU being defined it is Eigen::SuperLU, by default
	//! it is Eigen::SimplicialLDLT)
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
	//! Base of covertree. Could be overrided if TAPKEE_CUSTOM_PROPERTIES file is defined.
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
	
	//! Parameters map with keys being values of @ref TAPKEE_PARAMETERS and 
	//! values set to corresponding values wrapped to @ref any type
	typedef TAPKEE_INTERNAL_MAP<TAPKEE_PARAMETERS, any> ParametersMap;


namespace tapkee_internal 
{
#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
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
	typedef Triplet<ScalarType> SparseTriplet;
#else
	typedef Eigen::Triplet<ScalarType> SparseTriplet;
#endif
};

namespace tapkee_internal
{
	typedef TAPKEE_INTERNAL_VECTOR<SparseTriplet> SparseTriplets;
	typedef TAPKEE_INTERNAL_VECTOR<IndexType> LocalNeighbors;
	typedef TAPKEE_INTERNAL_VECTOR<LocalNeighbors> Neighbors;
	typedef TAPKEE_INTERNAL_PAIR<DenseMatrix,DenseVector> EmbeddingResult;
	typedef Eigen::DiagonalMatrix<ScalarType,Eigen::Dynamic> DenseDiagonalMatrix;
	typedef TAPKEE_INTERNAL_VECTOR<IndexType> Landmarks;
	typedef TAPKEE_INTERNAL_PAIR<SparseWeightMatrix,DenseDiagonalMatrix> Laplacian;
	typedef TAPKEE_INTERNAL_PAIR<DenseSymmetricMatrix,DenseSymmetricMatrix> DenseSymmetricMatrixPair;
};

} // namespace tapkee

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_projection.hpp>
/* End of Tapkee includes */

namespace tapkee 
{
//! Return result of the library - a pair of @ref DenseMatrix (embedding) and @ref ProjectingFunction
typedef TAPKEE_INTERNAL_PAIR<DenseMatrix,tapkee::ProjectingFunction> ReturnResult;
}

#endif
