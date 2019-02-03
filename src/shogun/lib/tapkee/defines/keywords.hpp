/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_KEYWORDS_H_
#define TAPKEE_DEFINES_KEYWORDS_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines/types.hpp>

#include <shogun/lib/tapkee/stichwort/keywords.hpp>
/* End of Tapkee includes */

namespace tapkee
{
	namespace {

		/** The keyword for the value that stands for the computation strategy
		 * to be used.
		 *
		 * Default value is @ref tapkee::HomogeneousCPUStrategy.
		 *
		 */
		const stichwort::ParameterKeyword<ComputationStrategy>
			computation_strategy("computation strategy (cpu, cpu+gpu)", HomogeneousCPUStrategy);

		/** The keyword for the value that stands for the dimension reduction
		 * method to be used.
		 *
		 * Should always be set with a value (no default value is provided).
		 *
		 * The corresponding value should be of type @ref tapkee::DimensionReductionMethod.
		 */
		const stichwort::ParameterKeyword<DimensionReductionMethod>
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
		const stichwort::ParameterKeyword<EigenMethod>
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
		 *        when @ref tapkee::spe_global_strategy is set to false)
		 * - @ref tapkee::ManifoldSculpting
		 *
		 * Default value is @ref tapkee::CoverTree if available, @ref tapkee::Brute otherwise.
		 *
		 * The corresponding value should have type
		 * @ref tapkee::NeighborsMethod.
		 */
		const stichwort::ParameterKeyword<NeighborsMethod>
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
		 * - @ref tapkee::ManifoldSculpting
		 *
		 * Default value is 5.
		 *
		 * The corresponding value should be of type @ref tapkee::IndexType,
		 * greater than 3 and less than the total number of vectors.
		 */
		const stichwort::ParameterKeyword<IndexType>
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
		const stichwort::ParameterKeyword<IndexType>
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
		const stichwort::ParameterKeyword<IndexType>
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
		const stichwort::ParameterKeyword<ScalarType>
			gaussian_kernel_width("the width of the gaussian kernel", 1.0);

		/** The keyword for the value that stores the maximal
		 * iteration that could be reached.
		 *
		 * Used by the following iterative methods:
		 * - @ref tapkee::StochasticProximityEmbedding
		 * - @ref tapkee::FactorAnalysis
		 * - @ref tapkee::ManifoldSculpting
		 *
		 * Default value is 100.
		 *
		 * The corresponding value should have type @ref tapkee::IndexType.
		 */
		const stichwort::ParameterKeyword<IndexType>
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
		const stichwort::ParameterKeyword<bool>
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
		const stichwort::ParameterKeyword<IndexType>
			spe_num_updates("SPE number of updates", 100);

		/** The keyword for the value that stores the tolerance of
		 * the SPE algorithm.
		 *
		 * Used by @ref tapkee::StochasticProximityEmbedding.
		 *
		 * Default value is 1e-9.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		const stichwort::ParameterKeyword<ScalarType>
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
		const stichwort::ParameterKeyword<ScalarType>
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
		const stichwort::ParameterKeyword<ScalarType>
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
		const stichwort::ParameterKeyword<ScalarType>
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
		 * - @ref tapkee::ManifoldSculpting
		 *
		 * Default is true.
		 *
		 * The corresponding value should have type bool.
		 */
		const stichwort::ParameterKeyword<bool>
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
		const stichwort::ParameterKeyword<ScalarType>
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
		const stichwort::ParameterKeyword<void (*)(double)>
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
		const stichwort::ParameterKeyword<bool (*)()>
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
		const stichwort::ParameterKeyword<ScalarType> sne_perplexity("SNE perplexity", 30.0);

		/** The keyword for the value that stores the theta
		 * parameter of the t-SNE algorithm.
		 *
		 * Used by @ref tapkee::tDistributedStochasticNeighborEmbedding.
		 *
		 * Default value is 0.5.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		const stichwort::ParameterKeyword<ScalarType> sne_theta("SNE theta", 0.5);

		/** The keyword for the value that stores the squishingRate
		 * parameter of the Manifold Sculpting algorithm.
		 *
		 * Used by @ref tapkee::ManifoldSculpting.
		 *
		 * Default value is 0.99.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		const stichwort::ParameterKeyword<ScalarType> squishing_rate("squishing rate", 0.99);
	}
}

#endif
