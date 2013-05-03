/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_H_
#define TAPKEE_DEFINES_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_exceptions.hpp>
#include <shogun/lib/tapkee/parameters/parameters.hpp>
#include <shogun/lib/tapkee/traits/callbacks_traits.hpp>
#include <shogun/lib/tapkee/traits/methods_traits.hpp>
/* End of Tapkee includes */

#include <map>
#include <vector>
#include <utility>
#include <string>

#define TAPKEE_WORLD_VERSION 1
#define TAPKEE_MAJOR_VERSION 0
#define TAPKEE_MINOR_VERSION 0

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
	#if EIGEN_VERSION_AT_LEAST(3,0,93)
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
	#define RESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(false)
	#define UNRESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(true)
#else
	#define RESTRICT_ALLOC
	#define UNRESTRICT_ALLOC
#endif
//// end of Eigen 3 library includes

//! Main namespace of the library, contains all public API definitions
namespace tapkee 
{
	//! Dimension reduction methods
	enum DimensionReductionMethod
	{
		/** Kernel Locally Linear Embedding as described 
		 * in @cite Decoste2001 */
		KernelLocallyLinearEmbedding,
		/** Neighborhood Preserving Embedding as described 
		 * in @cite He2005 */
		NeighborhoodPreservingEmbedding,
		/** Local Tangent Space Alignment as described 
		 * in @cite Zhang2002 */
		KernelLocalTangentSpaceAlignment,
		/** Linear Local Tangent Space Alignment as described 
		 * in @cite Zhang2007 */
		LinearLocalTangentSpaceAlignment,
		/** Hessian Locally Linear Embedding as described in 
		 * @cite Donoho2003 */
		HessianLocallyLinearEmbedding,
		/** Laplacian Eigenmaps as described in 
		 * @cite Belkin2002 */
		LaplacianEigenmaps,
		/** Locality Preserving Projections as described in 
		 * @cite He2003 */
		LocalityPreservingProjections,
		/** Diffusion map as described in 
		 * @cite Coifman2006 */
		DiffusionMap,
		/** Isomap as described in 
		 * @cite Tenenbaum2000 */
		Isomap,
		/** Landmark Isomap as described in 
		 * @cite deSilva2002 */
		LandmarkIsomap,
		/** Multidimensional scaling as described in 
		 * @cite Cox2000 */
		MultidimensionalScaling,
		/** Landmark multidimensional scaling as described in 
		 * @cite deSilva2004 */
		LandmarkMultidimensionalScaling,
		/** Stochastic Proximity Embedding as described in 
		 * @cite Agrafiotis2003 */
		StochasticProximityEmbedding,
		/** Kernel PCA as described in 
		 * @cite Scholkopf1997 */
		KernelPCA,
		/** Principal Component Analysis */
		PCA,
		/** Random Projection as described in
		 * @cite Kaski1998*/
		RandomProjection,
		/** Factor Analysis */
		FactorAnalysis,
		/** t-SNE and Barnes-Hut-SNE as described in 
		 * @cite vanDerMaaten2008 and @cite vanDerMaaten2013 */
		tDistributedStochasticNeighborEmbedding,
		/**SammonMapping Implementation*/
		SammonMapping,
		/** Passing through (doing nothing just passes the 
		 * data through) */
		PassThru
	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	// Methods identification
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KernelLocallyLinearEmbedding);
	METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(NeighborhoodPreservingEmbedding);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KernelLocalTangentSpaceAlignment);
	METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(LinearLocalTangentSpaceAlignment);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(HessianLocallyLinearEmbedding);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LaplacianEigenmaps);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(LocalityPreservingProjections);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(DiffusionMap);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(Isomap);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LandmarkIsomap);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(MultidimensionalScaling);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LandmarkMultidimensionalScaling);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(StochasticProximityEmbedding);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KernelPCA);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(PCA);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(RandomProjection);
	METHOD_THAT_NEEDS_NOTHING_IS(PassThru);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(FactorAnalysis);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(SammonMapping);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(tDistributedStochasticNeighborEmbedding);
#endif // DOXYGEN_SHOULD_SKIP_THS

	//! Neighbors computation methods
	enum NeighborsMethod
	{
		//! Brute force method with not least than 
		//! \f$ O(N N \log k) \f$ time complexity.
		//! Recommended to be used only in debug purposes.
		Brute,
#ifdef TAPKEE_USE_LGPL_COVERTREE
		//! Covertree-based method with approximate \f$ O(\log N) \f$ time complexity.
		//! Recommended to be used as a default method.
		CoverTree
#endif
	};
#ifdef TAPKEE_USE_LGPL_COVERTREE
	static NeighborsMethod default_neighbors_method = CoverTree;
#else
	static NeighborsMethod default_neighbors_method = Brute;
#endif

	//! Eigendecomposition methods
	enum EigenMethod
	{
#ifdef TAPKEE_WITH_ARPACK
		//! ARPACK-based method (requires the ARPACK library
		//! binaries to be available around). Recommended to be used as a 
		//! default method. Supports both generalized and standard eigenproblems.
		Arpack,
#endif
		//! Randomized method (implementation taken from the redsvd lib). 
		//! Supports only standard but not generalized eigenproblems.
		Randomized,
		//! Eigen library dense method (could be useful for debugging). Computes
		//! all eigenvectors thus can be very slow doing large-scale.
		Dense
	};
#ifdef TAPKEE_WITH_ARPACK
	static EigenMethod default_eigen_method = Arpack;
#else
	static EigenMethod default_eigen_method = Dense;
#endif

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

	//! The namespace that contains keywords for all parameters
	namespace keywords
	{
		//! The namespace that contains implementations for the keywords
		namespace keywords_internal
		{
			/** DefaultValue instance is useful 
			 * to set a parameter its default value.
			 *
			 * Once assigned to a keyword it produces a parameter
			 * with the default value assigned to the keyword.
			 */
			struct DefaultValue
			{
				DefaultValue() { }
			};

			/** ParameterKeyword instance is used to represent
			 * a keyword that is assigned to some value. Such
			 * an assignment results to instance of @ref Parameter
			 * class which can be later checked and casted back 
			 * to the value it represents.
			 *
			 * Usage is 
			 * @code
			 * 	ParameterKeyword<int> keyword;
			 * 	Parameter p = (keyword = 5);
			 * 	int p_value = p;
			 * @endcode
			 */
			template <typename T> 
			struct ParameterKeyword
			{
				typedef std::string Name;

				ParameterKeyword(const Name& n, const T& dv) : name(n), default_value(dv) { }
				ParameterKeyword(const ParameterKeyword& pk);
				ParameterKeyword operator=(const ParameterKeyword& pk); 

				Parameter operator=(const T& value) const
				{
					return Parameter::create(name,value);
				}
				Parameter operator=(const DefaultValue&) const
				{
					return Parameter::create(name,default_value);
				}
				operator Name() const
				{
					return name;
				}

				Name name;
				T default_value;
			};
		}
		using namespace keywords_internal;

		namespace {
			/** The keyword for the value that stands for the dimension reduction 
			 * method to be used.
			 *
			 * Should always be set with a value (no default value is provided).
			 *
			 * The corresponding value should be of type @ref tapkee::DimensionReductionMethod. 
			 */
			const ParameterKeyword<DimensionReductionMethod>
				method("dimension reduction method", PassThru);

			/** The keyword for the value that stands for the eigendecomposition
			 * method to be used.
			 * 
			 * Used by the following eigendecomposition-based methods:
			 *
			 * - @ref tapkee::KernelLocallyLinearEmbedding
			 * - @ref tapkee::NeighborhoodPreservingEmbedding
			 * - @ref tapkee::KernelLocalTangentSpaceAlignment
			 * - @ref tapkee::LinearLocalTangentSpaceAlignment
			 * - @ref tapkee::HessianLocallyLinearEmbedding
			 * - @ref tapkee::LaplacianEigenmaps
			 * - @ref tapkee::LocalityPreservingProjections
			 * - @ref tapkee::DiffusionMap
			 * - @ref tapkee::Isomap
			 * - @ref tapkee::LandmarkIsomap
			 * - @ref tapkee::MultidimensionalScaling
			 * - @ref tapkee::LandmarkMultidimensionalScaling
			 * - @ref tapkee::KernelPCA
			 * - @ref tapkee::PCA
			 *
			 * Default value is @ref tapkee::Arpack if available, @ref tapkee::Dense otherwise.
			 *
			 * The corresponding value should have type 
			 * @ref tapkee::EigenMethod. 
			 */
			const ParameterKeyword<EigenMethod>
				eigen_method("eigendecomposition method", default_eigen_method);

			/** The keyword for the value that stands for the neighbors
			 * finding method to be used.
			 * 
			 * Used by the following local methods:
			 * 
			 * - @ref tapkee::KernelLocallyLinearEmbedding
			 * - @ref tapkee::NeighborhoodPreservingEmbedding
			 * - @ref tapkee::KernelLocalTangentSpaceAlignment
			 * - @ref tapkee::LinearLocalTangentSpaceAlignment
			 * - @ref tapkee::HessianLocallyLinearEmbedding
			 * - @ref tapkee::LaplacianEigenmaps
			 * - @ref tapkee::LocalityPreservingProjections
			 * - @ref tapkee::Isomap
			 * - @ref tapkee::LandmarkIsomap
			 * - @ref tapkee::StochasticProximityEmbedding (with local strategy, i.e. 
			 *        when @ref tapkee::keywords::spe_global_strategy is set to false)
			 *
			 * Default value is @ref tapkee::CoverTree if available, @ref tapkee::Brute otherwise.
			 *
			 * The corresponding value should have type
			 * @ref tapkee::NeighborsMethod.
			 */
			const ParameterKeyword<NeighborsMethod> 
				neighbors_method("nearest neighbors method", default_neighbors_method);

			/** The keyword for the value that stores the number of neighbors.
			 *
			 * Used by all local methods such as:
			 * 
			 * - @ref tapkee::KernelLocallyLinearEmbedding
			 * - @ref tapkee::NeighborhoodPreservingEmbedding
			 * - @ref tapkee::KernelLocalTangentSpaceAlignment
			 * - @ref tapkee::LinearLocalTangentSpaceAlignment
			 * - @ref tapkee::HessianLocallyLinearEmbedding
			 * - @ref tapkee::LaplacianEigenmaps
			 * - @ref tapkee::LocalityPreservingProjections
			 * - @ref tapkee::Isomap
			 * - @ref tapkee::LandmarkIsomap
			 * - @ref tapkee::StochasticProximityEmbedding (with local strategy, i.e. 
			 *        when @ref tapkee::keywords::spe_global_strategy is set to false)
			 *
			 * Default value is 5.
			 *
			 * The corresponding value should be of type @ref tapkee::IndexType, 
			 * greater than 3 and less than the total number of vectors. 
			 */
			const ParameterKeyword<IndexType>
				num_neighbors("number of neighbors", 5);

			/** The keyword for the value that stores the target dimension.
			 *
			 * It is used by all the implemented methods. 
			 *
			 * Default value is 2.
			 * 
			 * The corresponding value should have type 
			 * @ref tapkee::IndexType and be greater than 
			 * 1 and less than the minimum of the total number 
			 * of vectors and the current dimension. 
			 */
			const ParameterKeyword<IndexType> 
				target_dimension("target dimension", 2);

			/** The keyword for the value that stores the number of
			 * timesteps in the diffusion map model.
			 * 
			 * Used by @ref tapkee::DiffusionMap.
			 *
			 * Default value is 3.
			 * 
			 * The corresponding value should have type @ref tapkee::IndexType.
			 */
			const ParameterKeyword<IndexType>
				diffusion_map_timesteps("diffusion map timesteps", 3);

			/** The keyword for the value that stores the width of
			 * the gaussian kernel.
			 *
			 * Used by the following methods:
			 *
			 * - @ref tapkee::LaplacianEigenmaps
			 * - @ref tapkee::LocalityPreservingProjections
			 * - @ref tapkee::DiffusionMap
			 *
			 * Default value is 1.0.
			 *
			 * The corresponding value should have type @ref tapkee::ScalarType.
			 */
			const ParameterKeyword<ScalarType>
				gaussian_kernel_width("the width of the gaussian kernel", 1.0);

			/** The keyword for the value that stores the maximal 
			 * iteration that could be reached.
			 *
			 * Used by the following iterative methods:
			 * - @ref tapkee::StochasticProximityEmbedding
			 * - @ref tapkee::FactorAnalysis
			 * 
			 * Default value is 100.
			 *
			 * The corresponding value should have type @ref tapkee::IndexType.
			 */
			const ParameterKeyword<IndexType>
				max_iteration("maximal iteration", 100);

			/** The keyword for the value that indicates
			 * whether global strategy of SPE should be used.
			 *
			 * Used by @ref tapkee::StochasticProximityEmbedding.
			 *
			 * Default value is true.
			 *
			 * The corresponding value should have type bool.
			 */
			const ParameterKeyword<bool>
				spe_global_strategy("SPE global strategy", true);

			/** The keyword for the value that stores the number of
			 * updates to be done in SPE.
			 *
			 * Used by @ref tapkee::StochasticProximityEmbedding.
			 *
			 * Default value is 100.
			 *
			 * The corresponding value should have type @ref tapkee::IndexType.
			 */
			const ParameterKeyword<IndexType>
				spe_num_updates("SPE number of updates", 100);
				
			/**parameters for Sammonmapping*/
			const ParameterKeyword<IndexType>
				opts_MaxHalves("Maximum Number of Halves",20);
			const ParameterKeyword<IndexType>
				opts_MaxIter("Maximum Number of Iterations",500);
			const ParameterKeyword<ScalarType>
				opts_TolFun("Tolerance in Function",1e-9);
			/** The keyword for the value that stores the tolerance of
			 * the SPE algorithm. 
			 * 
			 * Used by @ref tapkee::StochasticProximityEmbedding.
			 *
			 * Default value is 1e-9.
			 *
			 * The corresponding value should have type @ref tapkee::ScalarType.
			 */
			const ParameterKeyword<ScalarType>
				spe_tolerance("SPE tolerance", 1e-9);

			/** The keyword for the value that stores the ratio
			 * of landmark points to be used (1.0 means all
			 * points are landmarks and the reciprocal of the
			 * number of vectors means only one landmark).
			 *
			 * Used by the following landmark methods:
			 *
			 * - @ref tapkee::LandmarkIsomap
			 * - @ref tapkee::LandmarkMultidimensionalScaling
			 *
			 * Default is 0.5.
			 *  
			 * The corresponding value should have type @ref tapkee::ScalarType
			 * and be in [0,1] range.
			 */
			const ParameterKeyword<ScalarType>
				landmark_ratio("ratio of landmark points", 0.5);

			/** The keyword for the value that stores 
			 * the diagonal shift regularizer for
			 * eigenproblems.
			 *
			 * Default is 1e-9.
			 *
			 * Used by the following methods:
			 *
			 * - @ref tapkee::KernelLocallyLinearEmbedding
			 * - @ref tapkee::NeighborhoodPreservingEmbedding
			 * - @ref tapkee::KernelLocalTangentSpaceAlignment
			 * - @ref tapkee::LinearLocalTangentSpaceAlignment
			 * - @ref tapkee::HessianLocallyLinearEmbedding
			 * - @ref tapkee::LocalityPreservingProjections
			 *
			 * The corresponding value should have type @ref tapkee::ScalarType and
			 * be quite close to zero.
			 */
			const ParameterKeyword<ScalarType>
				nullspace_shift("diagonal shift of nullspace", 1e-9);

			/** The keyword for the value that stores
			 * the regularization shift of the locally linear embedding
			 * algorithm weights computation problem.
			 *
			 * Used by @ref tapkee::KernelLocallyLinearEmbedding.
			 *
			 * Default is 1e-3.
			 *
			 * The corresponding value should have type @ref tapkee::ScalarType and
			 * be quite close to zero.
			 */
			const ParameterKeyword<ScalarType>
				klle_shift("KLLE regularizer", 1e-3);

			/** The keyword for the value that indicates
			 * whether graph connectivity check should be done.
			 *
			 * Used by the following local methods:
			 * 
			 * - @ref tapkee::KernelLocallyLinearEmbedding
			 * - @ref tapkee::NeighborhoodPreservingEmbedding
			 * - @ref tapkee::KernelLocalTangentSpaceAlignment
			 * - @ref tapkee::LinearLocalTangentSpaceAlignment
			 * - @ref tapkee::HessianLocallyLinearEmbedding
			 * - @ref tapkee::LaplacianEigenmaps
			 * - @ref tapkee::LocalityPreservingProjections
			 * - @ref tapkee::Isomap
			 * - @ref tapkee::LandmarkIsomap
			 * - @ref tapkee::StochasticProximityEmbedding (with local strategy, i.e. 
			 *        when @ref tapkee::keywords::spe_global_strategy is set to false)
			 *
			 * Default is true. 
			 *
			 * The corresponding value should have type bool.
			 */
			const ParameterKeyword<bool>
				check_connectivity("check connectivity", true);

			/** The keyword for the value that stores the epsilon
			 * parameter of the Factor Analysis algorithm.
			 * 
			 * Used by @ref tapkee::FactorAnalysis.
			 *
			 * Default is 1e-9.
			 *
			 * The corresponding value should have type @ref tapkee::ScalarType.
			 */
			const ParameterKeyword<ScalarType>
				fa_epsilon("epsilon of FA", 1e-9);

			/** The keyword for the value that stores a pointer
			 * to the function which could be called to indicate progress
			 * that has been made (it is called with an argument in range [0,1],
			 * where 0 means 0% progress and 1 means 100% progress).
			 *
			 * Is not supported yet thus won't be used.
			 *
			 * The corresponding value should have type 
			 * @code void (*)(double) @endcode 
			 * (i.e. a pointer to some function that takes
			 *  double argument and returns nothing).
			 */
			const ParameterKeyword<void (*)(double)>  
				progress_function("progress function", NULL);

			/** The keyword for the value that stores a pointer 
			 * to the function which could be called to check whether
			 * computations were cancelled (the function should return 
			 * true if computations were cancelled).
			 *
			 * Currently, it is called only once when 
			 * method is starting to work.
			 * 
			 * If function returns true the library immediately 
			 * throws @ref tapkee::cancelled_exception.
			 *
			 * The corresponding value should have type
			 * @code bool (*)() @endcode 
			 * (i.e. a pointer to some function that takes
			 *  nothing and returns boolean).
			 */
			const ParameterKeyword<bool (*)()>
				cancel_function("cancel function", NULL);

			/** The keyword for the value that stores perplelixity
			 * parameter of t-SNE.
			 *
			 * Used by @ref tapkee::tDistributedStochasticNeighborEmbedding.
			 *
			 * Default value is 30.0;
			 *
			 * The corresponding value should have type @ref tapkee::ScalarType.
			 */
			const ParameterKeyword<ScalarType> sne_perplexity("SNE perplexity", 30.0);

			/** The keyword for the value that stores the theta 
			 * parameter of the t-SNE algorithm.
			 *
			 * Used by @ref tapkee::tDistributedStochasticNeighborEmbedding.
			 *
			 * Default value is 0.5.
			 * 
			 * The corresponding value should have type @ref tapkee::ScalarType.
			 */
			const ParameterKeyword<ScalarType> sne_theta("SNE theta", 0.5);

			/** The default value - assigning any keyword to this
			 * static struct produces a parameter with its default value.
			 */
			const DefaultValue by_default;
		}

	}

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

namespace tapkee_internal 
{
#if defined(TAPKEE_USE_PRIORITY_QUEUE) && defined(TAPKEE_USE_FIBONACCI_HEAP)
	#error "Can't use both priority queue and fibonacci heap at the same time"
#endif
#if !defined(TAPKEE_USE_PRIORITY_QUEUE) && !defined(TAPKEE_USE_FIBONACCI_HEAP)
	#define TAPKEE_USE_PRIORITY_QUEUE
#endif

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	template <typename T> struct Triplet
	{
		Triplet(IndexType colIndex, IndexType rowIndex, T valueT) : 
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
#else // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	typedef Eigen::Triplet<ScalarType> SparseTriplet;
#endif // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
} // End of namespace tapkee_internal

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

} // End of namespace tapkee_internal
} // End of namespace tapkee

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_projection.hpp>
/* End of Tapkee includes */

namespace tapkee 
{

	//! Return result of the library - a pair of @ref DenseMatrix (embedding) and @ref ProjectingFunction
	struct TapkeeOutput
	{
		TapkeeOutput() :
			embedding(), projection()
		{
		}
		TapkeeOutput(const DenseMatrix& e, const ProjectingFunction& p) :
			embedding(), projection(p)
		{
			embedding.swap(e);
		}
		TapkeeOutput(const TapkeeOutput& that) :
			embedding(), projection(that.projection)
		{
			this->embedding.swap(that.embedding);
		}
		DenseMatrix embedding;
		ProjectingFunction projection;
	};

} // End of namespace tapkee

#endif // TAPKEE_DEFINES_H_
