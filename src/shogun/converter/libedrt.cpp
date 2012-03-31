#include <shogun/lib/config.h>
#include <shogun/converter/libedrt.h>
#include <pthread.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/memory.h>
#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/base/Parallel.h>
#include <shogun/converter/libedrt_methods.h>

#define REQUIRE(x) if (!x) SG_SERROR("REQUIRE ERROR\n")

using namespace shogun;

int edrt_embedding(
		const edrt_options_t& options,
		const int target_dimension,
		const int N,
		const int dimension,
		const int k,
		double (*distance)(int, int, const void*),
		double (*kernel)(int, int, const void*),
		double* (*access_feature_vector)(int, const void*),
		void (*free_feature_vector)(int, const void*),
		const void* user_data,
		double **output)
{
	int* neighborhood_matrix = NULL;
	double* weight_matrix = NULL;

	REQUIRE(user_data);
	switch (options.method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			REQUIRE(kernel);

			neighborhood_matrix = kernel_neighbors_matrix(N, k, kernel, user_data);
			REQUIRE(neighborhood_matrix);

			weight_matrix = klle_weight_matrix(neighborhood_matrix, N, k, k, 
			                                   options.num_threads, 
			                                   options.klle_reconstruction_shift,
			                                   kernel, user_data);
			REQUIRE(weight_matrix);

			*output = eigendecomposition_embedding(weight_matrix, N, 
			                                       target_dimension, 
			                                       options.use_arpack,
			                                       options.nullspace_shift);
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			REQUIRE(kernel);
			REQUIRE(access_feature_vector);
			REQUIRE(free_feature_vector);
			//neighborhood_matrix = distance_neighbors_matrix(N, k, distance, user_data);
			//weight_matrix = lle_weight_matrix(neighborhood_matrix, feature_vector, user_data);
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			REQUIRE(kernel);
			
			neighborhood_matrix = kernel_neighbors_matrix(N, k, kernel, user_data);
			REQUIRE(neighborhood_matrix);

			weight_matrix = kltsa_weight_matrix(neighborhood_matrix, N, k, k,
			                                    target_dimension,
			                                    options.num_threads,
			                                    kernel, user_data);
			REQUIRE(weight_matrix);

			*output = eigendecomposition_embedding(weight_matrix, N, 
			                                       target_dimension, 
			                                       options.use_arpack,
			                                       options.nullspace_shift);

			break;
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			REQUIRE(kernel);
			REQUIRE(access_feature_vector);
			REQUIRE(free_feature_vector);
			//neighborhood_matrix = distance_neighbors_matrix(N, k, distance, user_data);
			//weight_matrix = ltsa_weight_matrix(neighborhood_matrix, feature_vector, user_data);
			break;
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			REQUIRE(kernel);
			REQUIRE(access_feature_vector);
			REQUIRE(free_feature_vector);
			//neighborhood_matrix = distance_neighbors_matrix(N, k, distance, user_data);
			//weight_matrix = hlle_weight_matrix(neighborhood_matrix, feature_vector, user_data);
			break;
		case LAPLACIAN_EIGENMAPS:
			REQUIRE(distance);
			//neighborhood_matrix = distance_neighbors_matrix(N, k, distance, user_data);
			//weight_matrix = laplacian_weight_matrix(neighborhood_matrix, feature_vector, user_data);
			break;
		case LOCALITY_PRESERVING_PROJECTIONS:
			REQUIRE(distance);
			REQUIRE(access_feature_vector);
			REQUIRE(free_feature_vector);
			//neighborhood_matrix = distance_neighbors_matrix(N, k, distance, user_data);
			//weight_matrix = laplacian_weight_matrix(neighborhood_matrix, feature_vector, user_data);
			break;
		case DIFFUSION_MAPS:
			REQUIRE(kernel);
			*output = diffusion_maps_embedding(N, options.diffusion_maps_t, target_dimension, kernel, user_data);
			break;
		case ISOMAP:
			REQUIRE(distance);
			break;
		case MULTIDIMENSIONAL_SCALING:
			REQUIRE(distance);
			break;
	}
	if (neighborhood_matrix)
		SG_FREE(neighborhood_matrix);
	if (weight_matrix)
		SG_FREE(weight_matrix);


	return 1;
}



