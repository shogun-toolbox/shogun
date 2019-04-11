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

	template <typename M>
	struct Method
	{
		Method(const char* n) : name_(n)
		{
		}
		Method(const M& m) : name_(m.name_)
		{
		}
		Method& operator=(const Method& m)
		{
			this->name_ = m.name_;
			return *this;
		}
		const char* name() const
		{
			return name_;
		}
		bool is(const M& m) const
		{
			return this->name()==m.name();
		}
		bool operator==(const M& m)
		{
			return this->name()==m.name();
		}
		const char* name_;
	};

	struct NeighborsMethod : public Method<NeighborsMethod>
	{
		NeighborsMethod(const char* n) : Method<NeighborsMethod>(n)
		{
		}
	};

	//! Brute force method with not least than
	//! \f$ O(N N \log k) \f$ time complexity.
	//! Recommended to be used only in debug purposes.
	static const NeighborsMethod Brute("Brute-force");
	//! Vantage point tree -based method.
	static const NeighborsMethod VpTree("Vantage point tree");
#ifdef TAPKEE_USE_LGPL_COVERTREE
	//! Covertree-based method with approximate \f$ O(\log N) \f$ time complexity.
	//! Recommended to be used as a default method.
	static const NeighborsMethod CoverTree("Cover tree");
#endif

#ifdef TAPKEE_USE_LGPL_COVERTREE
	static NeighborsMethod default_neighbors_method = CoverTree;
#else
	static NeighborsMethod default_neighbors_method = Brute;
#endif

	struct EigenMethod : public Method<EigenMethod>
	{
		EigenMethod(const char* n) : Method<EigenMethod>(n)
		{
		}
	};

#ifdef TAPKEE_WITH_ARPACK
	//! ARPACK-based method (requires the ARPACK library
	//! binaries to be available around). Recommended to be used as a
	//! default method. Supports both generalized and standard eigenproblems.
	static const EigenMethod Arpack("Arpack");
#endif
	//! Randomized method (implementation taken from the redsvd lib).
	//! Supports only standard but not generalized eigenproblems.
	static const EigenMethod Randomized("Randomized");
	//! Eigen library dense method (could be useful for debugging). Computes
	//! all eigenvectors thus can be very slow doing large-scale.
	static const EigenMethod Dense("Dense");

#ifdef TAPKEE_WITH_ARPACK
	static EigenMethod default_eigen_method = Arpack;
#else
	static EigenMethod default_eigen_method = Dense;
#endif

	struct ComputationStrategy : public Method<ComputationStrategy>
	{
		ComputationStrategy(const char* n) : Method<ComputationStrategy>(n)
		{
		}
	};

#ifdef TAPKEE_WITH_VIENNACL
	static const ComputationStrategy HeterogeneousOpenCLStrategy("OpenCL");
#endif
	static const ComputationStrategy HomogeneousCPUStrategy("CPU");

	static ComputationStrategy default_computation_strategy = HomogeneousCPUStrategy;

	namespace tapkee_internal
	{

		struct EigendecompositionStrategy : public Method<EigendecompositionStrategy>
		{
			EigendecompositionStrategy(const char* n, IndexType skp) : Method<EigendecompositionStrategy>(n),
				skip_(skp)
			{
			}
			IndexType skip() const
			{
				return skip_;
			}
			IndexType skip_;
		};

		static const EigendecompositionStrategy LargestEigenvalues("Largest eigenvalues", 0);
		static const EigendecompositionStrategy SquaredLargestEigenvalues("Largest eigenvalues of squared matrix", 0);
		static const EigendecompositionStrategy SmallestEigenvalues("Smallest eigenvalues", 1);

	}

}

#endif

