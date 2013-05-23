/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_NAMING_H_
#define TAPKEE_NAMING_H_

namespace tapkee
{

/** Returns the name of the provided method */
std::string get_method_name(DimensionReductionMethod m)
{
	switch (m)
	{
		case KernelLocallyLinearEmbedding: return "Kernel Locally Linear Embedding";
		case KernelLocalTangentSpaceAlignment: return "Local Tangent Space Alignment";
		case DiffusionMap: return "Diffusion Map";
		case MultidimensionalScaling: return "Classic Multidimensional Scaling";
		case LandmarkMultidimensionalScaling: return "Landmark Multidimensional Scaling";
		case Isomap: return "Isomap";
		case LandmarkIsomap: return "Landmark Isomap";
		case NeighborhoodPreservingEmbedding: return "Neighborhood Preserving Embedding";
		case LinearLocalTangentSpaceAlignment: return "Linear Local Tangent Space Alignment";
		case HessianLocallyLinearEmbedding: return "Hessian Locally Linear Embedding";
		case LaplacianEigenmaps: return "Laplacian Eigenmaps";
		case LocalityPreservingProjections: return "Locality Preserving Embedding";
		case PCA: return "Principal Component Analysis";
		case KernelPCA: return "Kernel Principal Component Analysis";
		case StochasticProximityEmbedding: return "Stochastic Proximity Embedding";
		case PassThru: return "passing through";
		case RandomProjection: return "Random Projection";
		case FactorAnalysis: return "Factor Analysis";
		case tDistributedStochasticNeighborEmbedding: return "t-distributed Stochastic Neighbor Embedding";
		case ManifoldSculpting: return "manifold sculpting";
	}
	return "hello";
}

/** Returns the name of the provided neighbors method */
std::string get_neighbors_method_name(NeighborsMethod m)
{
	switch (m)
	{
		case Brute: return "Brute-force";
		case VpTree: return "VP-tree";
#ifdef TAPKEE_USE_LGPL_COVERTREE
		case CoverTree: return "Cover Tree";
#endif
	}
	return "hello";
}

/** Returns the name of the provided eigen method */
std::string get_eigen_method_name(EigenMethod m)
{
	switch (m)
	{
#ifdef TAPKEE_WITH_ARPACK
		case Arpack: return "Arpack";
#endif
		case Dense: return "Dense";
		case Randomized: return "Randomized";
	}
	return "hello";
}

}
#endif
