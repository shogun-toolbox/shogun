/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
#include <shogun/lib/tapkee/utils/naming.hpp>
#include <shogun/lib/tapkee/utils/time.hpp>
#include <shogun/lib/tapkee/utils/logging.hpp>
#include <shogun/lib/tapkee/utils/features.hpp>
#include <shogun/lib/tapkee/parameters/defaults.hpp>
#include <shogun/lib/tapkee/parameters/context.hpp>
#include <shogun/lib/tapkee/routines/locally_linear.hpp>
#include <shogun/lib/tapkee/routines/eigendecomposition.hpp>
#include <shogun/lib/tapkee/routines/generalized_eigendecomposition.hpp>
#include <shogun/lib/tapkee/routines/multidimensional_scaling.hpp>
#include <shogun/lib/tapkee/routines/diffusion_maps.hpp>
#include <shogun/lib/tapkee/routines/laplacian_eigenmaps.hpp>
#include <shogun/lib/tapkee/routines/isomap.hpp>
#include <shogun/lib/tapkee/routines/pca.hpp>
#include <shogun/lib/tapkee/routines/random_projection.hpp>
#include <shogun/lib/tapkee/routines/spe.hpp>
#include <shogun/lib/tapkee/routines/fa.hpp>
#include <shogun/lib/tapkee/routines/manifold_sculpting.hpp>
#include <shogun/lib/tapkee/neighbors/neighbors.hpp>
#include <shogun/lib/tapkee/external/barnes_hut_sne/tsne.hpp>
/* End of Tapkee includes */

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{

template <typename T>
struct Positivity
{
	inline bool operator()(T v) const
	{
		return v>0;
	}
	inline std::string failureMessage(const stichwort::Parameter& p) const
	{
		return formatting::format("Positivity check failed for {}, its value is {}", p.name(), p.repr());
	}
};

template <typename T>
struct NonNegativity
{
	inline bool operator()(T v) const
	{
		return v>=0;
	}
	inline std::string failureMessage(const stichwort::Parameter& p) const
	{
		return formatting::format("Non-negativity check failed for {}, its value is {}", p.name(), p.repr());
	}
};

template <typename T>
struct InRange
{
	InRange(T l, T u) : lower(l), upper(u) { }
	inline bool operator()(T v) const
	{
		return (v>=lower) && (v<upper);
	}
	T lower;
	T upper;
	inline std::string failureMessage(const stichwort::Parameter& p) const
	{
		return formatting::format("[{}, {}) range check failed for {}, its value is {}", lower, upper, p.name(), p.repr());
	}
};

template <typename T>
struct InClosedRange
{
	InClosedRange(T l, T u) : lower(l), upper(u) { }
	inline bool operator()(T v) const
	{
		return (v>=lower) && (v<=upper);
	}
	T lower;
	T upper;
	inline std::string failureMessage(const stichwort::Parameter& p) const
	{
		return formatting::format("[{}, {}] range check failed for {}, its value is {}", lower, upper, p.name(), p.repr());
	}
};

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
		begin(b), end(e), p_computation_strategy(),
		p_eigen_method(), p_neighbors_method(), p_eigenshift(), p_traceshift(),
		p_check_connectivity(), p_n_neighbors(), p_width(), p_timesteps(),
		p_ratio(), p_max_iteration(), p_tolerance(), p_n_updates(), p_perplexity(),
		p_theta(), p_squishing_rate(), p_global_strategy(), p_epsilon(), p_target_dimension(),
		n_vectors(0), current_dimension(0)
	{
		n_vectors = (end-begin);

		p_target_dimension = parameters[target_dimension];
		p_n_neighbors = parameters[num_neighbors].checked().satisfies(Positivity<IndexType>());

		if (n_vectors > 0)
		{
			p_target_dimension.checked().satisfies(InRange<IndexType>(1,n_vectors));
			p_n_neighbors.checked().satisfies(InRange<IndexType>(3,n_vectors));
		}
		if (n_vectors == 0)
			throw no_data_error();


		p_computation_strategy = parameters[computation_strategy];
		p_eigen_method = parameters[eigen_method];
		p_neighbors_method = parameters[neighbors_method];
		p_check_connectivity = parameters[check_connectivity];
		p_width = parameters[gaussian_kernel_width].checked().satisfies(Positivity<ScalarType>());
		p_timesteps = parameters[diffusion_map_timesteps].checked().satisfies(Positivity<IndexType>());
		p_eigenshift = parameters[nullspace_shift];
		p_traceshift = parameters[klle_shift];
		p_max_iteration = parameters[max_iteration];
		p_tolerance = parameters[spe_tolerance].checked().satisfies(Positivity<ScalarType>());
		p_n_updates = parameters[spe_num_updates].checked().satisfies(Positivity<IndexType>());
		p_theta = parameters[sne_theta].checked().satisfies(NonNegativity<ScalarType>());
		p_squishing_rate = parameters[squishing_rate];
		p_global_strategy = parameters[spe_global_strategy];
		p_epsilon = parameters[fa_epsilon].checked().satisfies(NonNegativity<ScalarType>());
		p_perplexity = parameters[sne_perplexity].checked().satisfies(NonNegativity<ScalarType>());
		p_ratio = parameters[landmark_ratio];

		if (!is_dummy<FeaturesCallback>::value)
			current_dimension = features.dimension();
		else
			current_dimension = 0;
	}

	TapkeeOutput embedUsing(DimensionReductionMethod method)
	{
		if (context.is_cancelled())
			throw cancelled_exception();

#define tapkee_method_handle(X)																	\
		case X:																					\
		{																						\
			timed_context tctx__("[+] embedding with " # X);									\
			if (																				\
				((!MethodTraits<X>::needs_kernel)   || (!is_dummy<KernelCallback>::value))   &&	\
				((!MethodTraits<X>::needs_distance) || (!is_dummy<DistanceCallback>::value)) &&	\
				((!MethodTraits<X>::needs_features) || (!is_dummy<FeaturesCallback>::value))	\
			) {																					\
				return ImplementationBase::embed##X();											\
			} else {																			\
				return ImplementationBase::embedEmpty();										\
			}																					\
		}																						\
		break;																					\

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

	ParametersSet parameters;
	Context context;
	KernelCallback kernel;
	DistanceCallback distance;
	FeaturesCallback features;
	PlainDistance<RandomAccessIterator,DistanceCallback> plain_distance;
	KernelDistance<RandomAccessIterator,KernelCallback> kernel_distance;

	RandomAccessIterator begin;
	RandomAccessIterator end;

	Parameter p_computation_strategy;
	Parameter p_eigen_method;
	Parameter p_neighbors_method;
	Parameter p_eigenshift;
	Parameter p_traceshift;
	Parameter p_check_connectivity;
	Parameter p_n_neighbors;
	Parameter p_width;
	Parameter p_timesteps;
	Parameter p_ratio;
	Parameter p_max_iteration;
	Parameter p_tolerance;
	Parameter p_n_updates;
	Parameter p_perplexity;
	Parameter p_theta;
	Parameter p_squishing_rate;
	Parameter p_global_strategy;
	Parameter p_epsilon;
	Parameter p_target_dimension;

	IndexType n_vectors;
	IndexType current_dimension;

	template<class Distance>
	Neighbors findNeighborsWith(Distance d)
	{
		return find_neighbors(p_neighbors_method,begin,end,d,p_n_neighbors,p_check_connectivity);
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
			linear_weight_matrix(begin,end,neighbors,kernel,p_eigenshift,p_traceshift);
		DenseMatrix embedding =
			eigendecomposition(p_eigen_method,p_computation_strategy,SmallestEigenvalues,
					weight_matrix,p_target_dimension).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedKernelLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			tangent_weight_matrix(begin,end,neighbors,kernel,p_target_dimension,p_eigenshift);
		DenseMatrix embedding =
			eigendecomposition(p_eigen_method,p_computation_strategy,SmallestEigenvalues,
					weight_matrix,p_target_dimension).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedDiffusionMap()
	{
		IndexType target_dimension = static_cast<IndexType>(p_target_dimension);
		Parameter target_dimension_add = Parameter::create("target_dimension", target_dimension + 1);
		DenseSymmetricMatrix diffusion_matrix =
			compute_diffusion_matrix(begin,end,distance,p_width);
		EigendecompositionResult decomposition_result = eigendecomposition(p_eigen_method,p_computation_strategy,
							LargestEigenvalues,diffusion_matrix,target_dimension_add);
		DenseMatrix embedding = (decomposition_result.first).leftCols(target_dimension);
		// scaling with lambda_i^t
		for (IndexType i=0; i<target_dimension; i++)
			embedding.col(i).array() *= pow(decomposition_result.second(i), static_cast<IndexType>(p_timesteps));
		// scaling by eigenvector to largest eigenvalue 1
		for (IndexType i=0; i<target_dimension; i++)
			embedding.col(i).array() /= decomposition_result.first.col(target_dimension).array();
		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedMultidimensionalScaling()
	{
		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance);
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EigendecompositionResult embedding =
			eigendecomposition(p_eigen_method,p_computation_strategy,LargestEigenvalues,
					distance_matrix,p_target_dimension);

		for (IndexType i=0; i<static_cast<IndexType>(p_target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
		#undef MDS_MATRIX_OP
	}

	TapkeeOutput embedLandmarkMultidimensionalScaling()
	{
		p_ratio.checked().satisfies(InClosedRange<ScalarType>(3.0/n_vectors,1.0));

		Landmarks landmarks =
			select_landmarks_random(begin,end,p_ratio);
		DenseSymmetricMatrix distance_matrix =
			compute_distance_matrix(begin,end,landmarks,distance);
		DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EigendecompositionResult landmarks_embedding =
			eigendecomposition(p_eigen_method,p_computation_strategy,LargestEigenvalues,
					distance_matrix,p_target_dimension);
		for (IndexType i=0; i<static_cast<IndexType>(p_target_dimension); i++)
			landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
		return TapkeeOutput(triangulate(begin,end,distance,landmarks,
			landmark_distances_squared,landmarks_embedding,p_target_dimension), unimplementedProjectingFunction());
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
			eigendecomposition(p_eigen_method,p_computation_strategy,LargestEigenvalues,
					shortest_distances_matrix,p_target_dimension);

		for (IndexType i=0; i<static_cast<IndexType>(p_target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));

		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLandmarkIsomap()
	{
		p_ratio.checked().satisfies(InClosedRange<ScalarType>(3.0/n_vectors,1.0));

		Neighbors neighbors = findNeighborsWith(plain_distance);
		Landmarks landmarks =
			select_landmarks_random(begin,end,p_ratio);
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

		if (p_eigen_method.is(Dense))
		{
			DenseMatrix distance_matrix_sym = distance_matrix*distance_matrix.transpose();
			landmarks_embedding = eigendecomposition(p_eigen_method,p_computation_strategy,
					LargestEigenvalues,distance_matrix_sym,p_target_dimension);
		}
		else
		{
			landmarks_embedding = eigendecomposition(p_eigen_method,p_computation_strategy,
					SquaredLargestEigenvalues,distance_matrix,p_target_dimension);
		}

		DenseMatrix embedding = distance_matrix.transpose()*landmarks_embedding.first;

		for (IndexType i=0; i<static_cast<IndexType>(p_target_dimension); i++)
			embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
		return TapkeeOutput(embedding,unimplementedProjectingFunction());
	}

	TapkeeOutput embedNeighborhoodPreservingEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			linear_weight_matrix(begin,end,neighbors,kernel,p_eigenshift,p_traceshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
				features,current_dimension);
		EigendecompositionResult projection_result =
			generalized_eigendecomposition(p_eigen_method,p_computation_strategy,
					SmallestEigenvalues,eig_matrices.first,eig_matrices.second,p_target_dimension);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension),projecting_function);
	}

	TapkeeOutput embedHessianLocallyLinearEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			hessian_weight_matrix(begin,end,neighbors,kernel,p_target_dimension);
		return TapkeeOutput(eigendecomposition(p_eigen_method,p_computation_strategy,
					SmallestEigenvalues,weight_matrix,p_target_dimension).first,
				unimplementedProjectingFunction());
	}

	TapkeeOutput embedLaplacianEigenmaps()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		Laplacian laplacian =
			compute_laplacian(begin,end,neighbors,distance,p_width);
		return TapkeeOutput(generalized_eigendecomposition(p_eigen_method,p_computation_strategy,
					SmallestEigenvalues,laplacian.first,laplacian.second,p_target_dimension).first,
				unimplementedProjectingFunction());
	}

	TapkeeOutput embedLocalityPreservingProjections()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		Laplacian laplacian =
			compute_laplacian(begin,end,neighbors,distance,p_width);
		DenseSymmetricMatrixPair eigenproblem_matrices =
			construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
					features,current_dimension);
		EigendecompositionResult projection_result =
			generalized_eigendecomposition(p_eigen_method,p_computation_strategy,
					SmallestEigenvalues,eigenproblem_matrices.first,eigenproblem_matrices.second,p_target_dimension);
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
			eigendecomposition(p_eigen_method,p_computation_strategy,
					LargestEigenvalues,centered_covariance_matrix,p_target_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedRandomProjection()
	{
		DenseMatrix projection_matrix =
			gaussian_projection_matrix(current_dimension,p_target_dimension);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);

		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_matrix,mean_vector));
		return TapkeeOutput(project(projection_matrix,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedKernelPCA()
	{
		DenseSymmetricMatrix centered_kernel_matrix =
			compute_centered_kernel_matrix(begin,end,kernel);
		EigendecompositionResult embedding = eigendecomposition(p_eigen_method,p_computation_strategy,
				LargestEigenvalues,centered_kernel_matrix,p_target_dimension);
		for (IndexType i=0; i<static_cast<IndexType>(p_target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLinearLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			tangent_weight_matrix(begin,end,neighbors,kernel,p_target_dimension,p_eigenshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_lltsa_eigenproblem(weight_matrix,begin,end,
				features,current_dimension);
		EigendecompositionResult projection_result =
			generalized_eigendecomposition(p_eigen_method,p_computation_strategy,SmallestEigenvalues,
					eig_matrices.first,eig_matrices.second,p_target_dimension);
		DenseVector mean_vector =
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension),
				projecting_function);
	}

	TapkeeOutput embedStochasticProximityEmbedding()
	{
		Neighbors neighbors;
		if (p_global_strategy.is(false))
		{
			neighbors = findNeighborsWith(plain_distance);
		}

		return TapkeeOutput(spe_embedding(begin,end,distance,neighbors,
				p_target_dimension,p_global_strategy,p_tolerance,p_n_updates,p_max_iteration), unimplementedProjectingFunction());
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
		return TapkeeOutput(project(begin,end,features,current_dimension,p_max_iteration,p_epsilon,
									p_target_dimension, mean_vector), unimplementedProjectingFunction());
	}

	TapkeeOutput embedtDistributedStochasticNeighborEmbedding()
	{
		p_perplexity.checked().satisfies(InClosedRange<ScalarType>(0.0,(n_vectors-1)/3.0));

		DenseMatrix data =
			dense_matrix_from_features(features, current_dimension, begin, end);

		DenseMatrix embedding(static_cast<IndexType>(p_target_dimension),n_vectors);
		tsne::TSNE tsne;
		tsne.run(data,data.cols(),data.rows(),embedding.data(),p_target_dimension,p_perplexity,p_theta);

		return TapkeeOutput(embedding.transpose(), unimplementedProjectingFunction());
	}

	TapkeeOutput embedManifoldSculpting()
	{
		p_squishing_rate.checked().satisfies(InRange<ScalarType>(0.0,1.0));

		DenseMatrix embedding =
			dense_matrix_from_features(features, current_dimension, begin, end);

		Neighbors neighbors = findNeighborsWith(plain_distance);

		manifold_sculpting_embed(begin, end, embedding, p_target_dimension, neighbors, distance, p_max_iteration, p_squishing_rate);

		return TapkeeOutput(embedding, tapkee::ProjectingFunction());
	}

};

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeaturesCallback>
ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeaturesCallback>
	initialize(RandomAccessIterator begin, RandomAccessIterator end,
	           KernelCallback kernel, DistanceCallback distance, FeaturesCallback features,
	           stichwort::ParametersSet& pmap, const Context& ctx)
{
	return ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeaturesCallback>(
			begin,end,kernel,distance,features,pmap,ctx);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
