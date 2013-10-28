/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_METHODS_H_
#define TAPKEE_DEFINES_METHODS_H_

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
		/** Manifold Sculpting as described in
		* @cite Gashler2007 */
		ManifoldSculpting,
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
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(tDistributedStochasticNeighborEmbedding);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(ManifoldSculpting);
#endif // DOXYGEN_SHOULD_SKIP_THS

	//! Neighbors computation methods
	enum NeighborsMethod
	{
		//! Brute force method with not least than
		//! \f$ O(N N \log k) \f$ time complexity.
		//! Recommended to be used only in debug purposes.
		Brute,
		VpTree
#ifdef TAPKEE_USE_LGPL_COVERTREE
		//! Covertree-based method with approximate \f$ O(\log N) \f$ time complexity.
		//! Recommended to be used as a default method.
		, CoverTree
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

}

#endif

