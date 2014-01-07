/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

/* Tapkee includes */
#include <lib/tapkee/defines.hpp>
#include <lib/tapkee/utils/naming.hpp>
#include <lib/tapkee/utils/time.hpp>
#include <lib/tapkee/utils/logging.hpp>
#include <lib/tapkee/utils/conditional_select.hpp>
#include <lib/tapkee/utils/features.hpp>
#include <lib/tapkee/parameters/defaults.hpp>
#include <lib/tapkee/parameters/context.hpp>
#include <lib/tapkee/routines/locally_linear.hpp>
#include <lib/tapkee/routines/eigendecomposition.hpp>
#include <lib/tapkee/routines/generalized_eigendecomposition.hpp>
#include <lib/tapkee/routines/multidimensional_scaling.hpp>
#include <lib/tapkee/routines/diffusion_maps.hpp>
#include <lib/tapkee/routines/laplacian_eigenmaps.hpp>
#include <lib/tapkee/routines/isomap.hpp>
#include <lib/tapkee/routines/pca.hpp>
#include <lib/tapkee/routines/random_projection.hpp>
#include <lib/tapkee/routines/spe.hpp>
#include <lib/tapkee/routines/fa.hpp>
#include <lib/tapkee/routines/manifold_sculpting.hpp>
#include <lib/tapkee/neighbors/neighbors.hpp>
#include <lib/tapkee/external/barnes_hut_sne/tsne.hpp>
/* End of Tapkee includes */

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeaturesCallback>
class ImplementationBase
{
public:

	ImplementationBase(RandomAccessIterator b, RandomAccessIterator e,
	                   KernelCallback k, DistanceCallback d, FeaturesCallback f,
	                   ParametersSet& pmap, const Context& ctx) :
		parameters(pmap), context(ctx), kernel(k), distance(d), features(f),
		plain_distance(PlainDistance<RandomAccessIterator,DistanceCallback>(distance)),
		kernel_distance(KernelDistance<RandomAccessIterator,KernelCallback>(kernel)),
		begin(b), end(e),
		eigen_method(), neighbors_method(), eigenshift(), traceshift(),
		check_connectivity(), n_neighbors(), width(), timesteps(),
		ratio(), max_iteration(), tolerance(), n_updates(), perplexity(),
		theta(), squishing_rate(), global_strategy(), epsilon(), target_dimension(),
		n_vectors(0), current_dimension(0)
	{
		n_vectors = (end-begin);

		target_dimension = parameters(keywords::target_dimension);
		n_neighbors = parameters(keywords::num_neighbors).checked().positive();

		if (n_vectors > 0)
		{
			target_dimension.checked()
				.inRange(static_cast<IndexType>(1),static_cast<IndexType>(n_vectors));
			n_neighbors.checked()
				.inRange(static_cast<IndexType>(3),static_cast<IndexType>(n_vectors));
		}

		eigen_method = parameters(keywords::eigen_method);
		neighbors_method = parameters(keywords::neighbors_method);
		check_connectivity = parameters(keywords::check_connectivity);
		width = parameters(keywords::gaussian_kernel_width).checked().positive();
		timesteps = parameters(keywords::diffusion_map_timesteps).checked().positive();
		eigenshift = parameters(keywords::nullspace_shift);
		traceshift = parameters(keywords::klle_shift);
		max_iteration = parameters(keywords::max_iteration);
		tolerance = parameters(keywords::spe_tolerance).checked().positive();
		n_updates = parameters(keywords::spe_num_updates).checked().positive();
		theta = parameters(keywords::sne_theta).checked().nonNegative();
		squishing_rate = parameters(keywords::squishing_rate);
		global_strategy = parameters(keywords::spe_global_strategy);
		epsilon = parameters(keywords::fa_epsilon).checked().nonNegative();
		perplexity = parameters(keywords::sne_perplexity).checked().nonNegative();
		ratio = parameters(keywords::landmark_ratio);

		if (!is_dummy<FeaturesCallback>::value)
		{
			current_dimension = features.dimension();
		}
		else
		{
			current_dimension = 0;
		}
	}

	TapkeeOutput embedUsing(DimensionReductionMethod method)
	{
		if (context.is_cancelled())
			throw cancelled_exception();

		using std::mem_fun_ref_t;
		using std::mem_fun_ref;
		typedef std::mem_fun_ref_t<TapkeeOutput,ImplementationBase> ImplRef;

#define tapkee_method_handle(X)																	\
		case X:																					\
		{																						\
			timed_context tctx__("[+] embedding with " # X);									\
			ImplRef ref = conditional_select<													\
				((!MethodTraits<X>::needs_kernel)   || (!is_dummy<KernelCallback>::value))   &&	\
				((!MethodTraits<X>::needs_distance) || (!is_dummy<DistanceCallback>::value)) &&	\
				((!MethodTraits<X>::needs_features) || (!is_dummy<FeaturesCallback>::value)),	\
					ImplRef>()(mem_fun_ref(&ImplementationBase::embed##X),						\
					           mem_fun_ref(&ImplementationBase::embedEmpty));					\
			return ref(*this);																	\
		}																						\
		break																					\

		switch (method)
		{
			tapkee_method_handle(KernelLocallyLinearEmbedding);
			tapkee_method_handle(KernelLocalTangentSpaceAlignment);
			tapkee_method_handle(DiffusionMap);
			tapkee_method_handle(MultidimensionalScaling);
			tapkee_method_handle(LandmarkMultidimensionalScaling);
			tapkee_method_handle(Isomap);
			tapkee_method_handle(LandmarkIsomap);
			tapkee_method_handle(NeighborhoodPreservingEmbedding);
			tapkee_method_handle(LinearLocalTangentSpaceAlignment);
			tapkee_method_handle(HessianLocallyLinearEmbedding);
			tapkee_method_handle(LaplacianEigenmaps);
			tapkee_method_handle(LocalityPreservingProjections);
			tapkee_method_handle(PCA);
			tapkee_method_handle(KernelPCA);
			tapkee_method_handle(RandomProjection);
			tapkee_method_handle(StochasticProximityEmbedding);
			tapkee_method_handle(PassThru);
			tapkee_method_handle(FactorAnalysis);
			tapkee_method_handle(tDistributedStochasticNeighborEmbedding);
			tapkee_method_handle(ManifoldSculpting);
		}
#undef tapkee_method_handle
		return TapkeeOutput();
	}

private:

	static const IndexType SkipOneEigenvalue = 1;
	static const IndexType SkipNoEigenvalues = 0;

	ParametersSet parameters;
	Context context;
	KernelCallback kernel;
	DistanceCallback distance;
	FeaturesCallback features;
	PlainDistance<RandomAccessIterator,DistanceCallback> plain_distance;
	KernelDistance<RandomAccessIterator,KernelCallback> kernel_distance;

	RandomAccessIterator begin;
	RandomAccessIterator end;

	Parameter eigen_method;
	Parameter neighbors_method;
	Parameter eigenshift;
	Parameter traceshift;
	Parameter check_connectivity;
	Parameter n_neighbors;
	Parameter width;
	Parameter timesteps;
	Parameter ratio;
	Parameter max_iteration;
	Parameter tolerance;
	Parameter n_updates;
	Parameter perplexity;
	Parameter theta;
	Parameter squishing_rate;
	Parameter global_strategy;
	Parameter epsilon;
	Parameter target_dimension;

	IndexType n_vectors;
	IndexType current_dimension;

	template<class Distance>
	Neighbors findNeighborsWith(Distance d)
	{
		return find_neighbors(neighbors_method,begin,end,d,n_neighbors,check_connectivity);
	}

	static tapkee::ProjectingFunction unimplementedProjectingFunction()
	{
		return tapkee::ProjectingFunction();
	}

	TapkeeOutput embedEmpty()
	{
		throw unsupported_method_error("Some callback is missed");
		return TapkeeOutput();
	}

	TapkeeOutput embedKernelLocallyLinearEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			linear_weight_matrix(begin,end,neighbors,kernel,eigenshift,traceshift);
		DenseMatrix embedding =
			eigendecomposition<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
				weight_matrix,target_dimension,SkipOneEigenvalue).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedKernelLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			tangent_weight_matrix(begin,end,neighbors,kernel,target_dimension,eigenshift);
		DenseMatrix embedding =
			eigendecomposition<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
				weight_matrix,target_dimension,SkipOneEigenvalue).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedDiffusionMap()
	{
		#ifdef TAPKEE_GPU
			#define DM_MATRIX_OP GPUDenseImplicitSquareMatrixOperation
		#else
			#define DM_MATRIX_OP DenseImplicitSquareSymmetricMatrixOperation
		#endif

		DenseSymmetricMatrix diffusion_matrix =
			compute_diffusion_matrix(begin,end,distance,timesteps,width);
		DenseMatrix embedding =
			eigendecomposition<DenseSymmetricMatrix,DM_MATRIX_OP>(eigen_method,diffusion_matrix,
				target_dimension,SkipNoEigenvalues).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());

		#undef DM_MATRIX_OP
	}

	TapkeeOutput embedMultidimensionalScaling()
	{
		#ifdef TAPKEE_GPU
			#define MDS_MATRIX_OP GPUDenseImplicitSquareMatrixOperation
		#else
			#define MDS_MATRIX_OP DenseMatrixOperation
		#endif

		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance);
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EigendecompositionResult embedding =
			eigendecomposition<DenseSymmetricMatrix,MDS_MATRIX_OP>(eigen_method,
				distance_matrix,target_dimension,SkipNoEigenvalues);

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
		#undef MDS_MATRIX_OP
	}

	TapkeeOutput embedLandmarkMultidimensionalScaling()
	{
		ratio.checked()
			.inClosedRange(static_cast<ScalarType>(3.0/n_vectors),
			               static_cast<ScalarType>(1.0));

		Landmarks landmarks =
			select_landmarks_random(begin,end,ratio);
		DenseSymmetricMatrix distance_matrix =
			compute_distance_matrix(begin,end,landmarks,distance);
		DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EigendecompositionResult landmarks_embedding =
			eigendecomposition<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
				distance_matrix,target_dimension,SkipNoEigenvalues);
		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
		return TapkeeOutput(triangulate(begin,end,distance,landmarks,
			landmark_distances_squared,landmarks_embedding,target_dimension), unimplementedProjectingFunction());
	}

	TapkeeOutput embedIsomap()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		DenseSymmetricMatrix shortest_distances_matrix =
			compute_shortest_distances_matrix(begin,end,neighbors,distance);
		shortest_distances_matrix = shortest_distances_matrix.array().square();
		centerMatrix(shortest_distances_matrix);
		shortest_distances_matrix.array() *= -0.5;

		EigendecompositionResult embedding =
			eigendecomposition<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
				shortest_distances_matrix,target_dimension,SkipNoEigenvalues);

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));

		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLandmarkIsomap()
	{
		ratio.checked()
			.inClosedRange(static_cast<ScalarType>(3.0/n_vectors),
			               static_cast<ScalarType>(1.0));

		Neighbors neighbors = findNeighborsWith(plain_distance);
		Landmarks landmarks =
			select_landmarks_random(begin,end,ratio);
		DenseMatrix distance_matrix =
			compute_shortest_distances_matrix(begin,end,landmarks,neighbors,distance);
		distance_matrix = distance_matrix.array().square();

		DenseVector col_means = distance_matrix.colwise().mean();
		DenseVector row_means = distance_matrix.rowwise().mean();
		ScalarType grand_mean = distance_matrix.mean();
		distance_matrix.array() += grand_mean;
		distance_matrix.colwise() -= row_means;
		distance_matrix.rowwise() -= col_means.transpose();
		distance_matrix.array() *= -0.5;

		EigendecompositionResult landmarks_embedding;

		if (eigen_method.is(Dense))
		{
			DenseMatrix distance_matrix_sym = distance_matrix*distance_matrix.transpose();
			landmarks_embedding = eigendecomposition<DenseSymmetricMatrix,DenseImplicitSquareMatrixOperation>
				(eigen_method,distance_matrix_sym,target_dimension,SkipNoEigenvalues);
		}
		else
		{
			landmarks_embedding = eigendecomposition<DenseSymmetricMatrix,DenseImplicitSquareMatrixOperation>
				(eigen_method,distance_matrix,target_dimension,SkipNoEigenvalues);
		}

		DenseMatrix embedding = distance_matrix.transpose()*landmarks_embedding.first;

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
		return TapkeeOutput(embedding,unimplementedProjectingFunction());
	}

	TapkeeOutput embedNeighborhoodPreservingEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			linear_weight_matrix(begin,end,neighbors,kernel,eigenshift,traceshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
				features,current_dimension);
		EigendecompositionResult projection_result =
			generalized_eigendecomposition<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
				eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SkipNoEigenvalues);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension),projecting_function);
	}

	TapkeeOutput embedHessianLocallyLinearEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			hessian_weight_matrix(begin,end,neighbors,kernel,target_dimension);
		return TapkeeOutput(eigendecomposition<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SkipOneEigenvalue).first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLaplacianEigenmaps()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		Laplacian laplacian =
			compute_laplacian(begin,end,neighbors,distance,width);
		return TapkeeOutput(generalized_eigendecomposition<SparseWeightMatrix,DenseDiagonalMatrix,SparseInverseMatrixOperation>(
			eigen_method,laplacian.first,laplacian.second,target_dimension,SkipOneEigenvalue).first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLocalityPreservingProjections()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		Laplacian laplacian =
			compute_laplacian(begin,end,neighbors,distance,width);
		DenseSymmetricMatrixPair eigenproblem_matrices =
			construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
					features,current_dimension);
		EigendecompositionResult projection_result =
			generalized_eigendecomposition<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
				eigen_method,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension,SkipNoEigenvalues);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedPCA()
	{
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);
		DenseSymmetricMatrix centered_covariance_matrix =
			compute_covariance_matrix(begin,end,mean_vector,features,current_dimension);
		EigendecompositionResult projection_result =
			eigendecomposition<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_covariance_matrix,target_dimension,SkipNoEigenvalues);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedRandomProjection()
	{
		DenseMatrix projection_matrix =
			gaussian_projection_matrix(current_dimension, target_dimension);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);

		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_matrix,mean_vector));
		return TapkeeOutput(project(projection_matrix,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedKernelPCA()
	{
		DenseSymmetricMatrix centered_kernel_matrix =
			compute_centered_kernel_matrix(begin,end,kernel);
		EigendecompositionResult embedding = eigendecomposition<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
			centered_kernel_matrix,target_dimension,SkipNoEigenvalues);
		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLinearLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			tangent_weight_matrix(begin,end,neighbors,kernel,target_dimension,eigenshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_lltsa_eigenproblem(weight_matrix,begin,end,
				features,current_dimension);
		EigendecompositionResult projection_result =
			generalized_eigendecomposition<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
				eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SkipNoEigenvalues);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension),
				projecting_function);
	}

	TapkeeOutput embedStochasticProximityEmbedding()
	{
		Neighbors neighbors;
		if (global_strategy.is(false))
		{
			neighbors = findNeighborsWith(plain_distance);
		}

		return TapkeeOutput(spe_embedding(begin,end,distance,neighbors,
				target_dimension,global_strategy,tolerance,n_updates,max_iteration), unimplementedProjectingFunction());
	}

	TapkeeOutput embedPassThru()
	{
		DenseMatrix feature_matrix =
			dense_matrix_from_features(features, current_dimension, begin, end);
		return TapkeeOutput(feature_matrix.transpose(),tapkee::ProjectingFunction());
	}

	TapkeeOutput embedFactorAnalysis()
	{
		DenseVector mean_vector = compute_mean(begin,end,features,current_dimension);
		return TapkeeOutput(project(begin,end,features,current_dimension,max_iteration,epsilon,
									target_dimension, mean_vector), unimplementedProjectingFunction());
	}

	TapkeeOutput embedtDistributedStochasticNeighborEmbedding()
	{
		perplexity.checked()
			.inClosedRange(static_cast<ScalarType>(0.0),
			               static_cast<ScalarType>((n_vectors-1)/3.0));

		DenseMatrix data =
			dense_matrix_from_features(features, current_dimension, begin, end);

		DenseMatrix embedding(static_cast<IndexType>(target_dimension),n_vectors);
		tsne::TSNE tsne;
		tsne.run(data.data(),n_vectors,current_dimension,embedding.data(),target_dimension,perplexity,theta);

		return TapkeeOutput(embedding.transpose(), unimplementedProjectingFunction());
	}

	TapkeeOutput embedManifoldSculpting()
	{
		squishing_rate.checked()
			.inRange(static_cast<ScalarType>(0.0),
			         static_cast<ScalarType>(1.0));

		DenseMatrix embedding =
			dense_matrix_from_features(features, current_dimension, begin, end);

		Neighbors neighbors = findNeighborsWith(plain_distance);

		manifold_sculpting_embed(begin, end, embedding, target_dimension, neighbors, distance, max_iteration, squishing_rate);

		return TapkeeOutput(embedding, tapkee::ProjectingFunction());
	}

};

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeaturesCallback>
ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeaturesCallback>
	initialize(RandomAccessIterator begin, RandomAccessIterator end,
	           KernelCallback kernel, DistanceCallback distance, FeaturesCallback features,
	           ParametersSet& pmap, const Context& ctx)
{
	return ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeaturesCallback>(
			begin,end,kernel,distance,features,pmap,ctx);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
