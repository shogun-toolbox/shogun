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

#define REQUIRE(x) if (!x) SG_SERROR("REQUIRE ERROR\n")
#define MAX(x,y) x>y ? x : y

using namespace shogun;

struct thread_parameters_t
{
	int N, k, target_dimension, thread, num_threads;
	const int* neighborhood_matrix;
	double (*distance)(int, int, void*);
	double (*kernel)(int, int, void*);
	void* user_data;
};

struct klle_thread_parameters_t : thread_parameters_t
{
	double *thread_gram_matrix, *id_vector, *W_matrix;
};

int edrt_embedding(
		const edrt_options_t& options,
		const int target_dimension,
		const int N,
		const int dimension,
		const int k,
		double (*distance)(int, int, void*),
		double (*kernel)(int, int, void*),
		double* (*access_feature_vector)(int, void*),
		void (*free_feature_vector)(int, void*),
		void* user_data,
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
			weight_matrix = klle_weight_matrix(neighborhood_matrix, N, k, k, options.num_threads, kernel, user_data);
			SG_FREE(neighborhood_matrix);
			*output = eigendecomposition_embedding(weight_matrix, N, target_dimension, options.use_arpack);
			SG_FREE(weight_matrix);
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
			//neighborhood_matrix = kernel_neighbors_matrix(N, k, kernel, user_data);
			//weight_matrix = kltsa_weight_matrix(neighborhood_matrix, kernel, user_data);
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
			break;
		case ISOMAP:
			REQUIRE(distance);
			break;
		case MULTIDIMENSIONAL_SCALING:
			REQUIRE(distance);
			break;
	}
	return 1;
}

double* klle_weight_matrix(
		int* neighborhood_matrix,
		int N,
		int k,
		int matrix_k,
		int num_threads,
		double (*kernel)(int, int, void*),
		void* user_data)
{
#ifdef HAVE_PTHREAD
	int t;
	// allocate threads
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	klle_thread_parameters_t* parameters = 
		new klle_thread_parameters_t[num_threads];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
#else
	int num_threads = 1;
#endif 
	// init storages to be used
	double* thread_gram_matrix = SG_MALLOC(double, k*k*num_threads);
	double* id_vector = SG_MALLOC(double, k*num_threads);
	double* weight_matrix = SG_MALLOC(double, N*N);

#ifdef HAVE_PTHREAD
	for (t=0; t<num_threads; t++)
	{
		parameters[t].thread = t;
		parameters[t].num_threads = num_threads;
		parameters[t].k = k;
		parameters[t].N = N;
		parameters[t].neighborhood_matrix = neighborhood_matrix;
		parameters[t].kernel = kernel;
		parameters[t].thread_gram_matrix = thread_gram_matrix+(k*k)*t;
		parameters[t].id_vector = id_vector+k*t;
		parameters[t].W_matrix = weight_matrix;
		parameters[t].user_data = user_data;
		pthread_create(&threads[t], &attr, klle_weight_matrix_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	// TODO
#endif

	// clean
	SG_FREE(id_vector);
	SG_FREE(thread_gram_matrix);

	return weight_matrix;
}

void* klle_weight_matrix_thread(void* params)
{
	klle_thread_parameters_t* parameters = (klle_thread_parameters_t*)params;
	int thread = parameters->thread;
	int num_threads = parameters->num_threads;
	int k = parameters->k;
	int N = parameters->N;
	const int* neighborhood_matrix = parameters->neighborhood_matrix;
	double* thread_gram_matrix = parameters->thread_gram_matrix;
	double (*kernel)(int, int, void*) = parameters->kernel;
	double* id_vector = parameters->id_vector;
	double* W_matrix = parameters->W_matrix;
	void* user_data = parameters->user_data;
	double reconstruction_shift = 0.0;

	int i,q,p;
	double norming,trace;

	double* ini_dots = SG_MALLOC(double, k);

	for (i=thread; i<N; i+=num_threads)
	{
		double kii = kernel(i,i,user_data);
		for (q=0; q<k; q++)
			ini_dots[q] = kernel(i,neighborhood_matrix[i*k+q],user_data);

		for (q=0; q<k; q++)
		{
			for (p=0; p<k; p++)
			{
				thread_gram_matrix[q*k+p] = 
					kii -
					ini_dots[q] -
					ini_dots[p] +
					kernel(neighborhood_matrix[i*k+q],neighborhood_matrix[i*k+p],user_data);
			}
		}

		for (q=0; q<k; q++)
			id_vector[q] = 1.0;

		// compute tr(C)
		if (reconstruction_shift != 0.0)
		{
			trace = 0.0;
			for (q=0; q<k; q++)
				trace += thread_gram_matrix[q*k+q];
		
			// regularize gram matrix
			for (q=0; q<k; q++)
				thread_gram_matrix[q*k+q] += reconstruction_shift*trace;
		}

		clapack_dposv(CblasColMajor,CblasLower,k,1,thread_gram_matrix,k,id_vector,k);

		// normalize weights
		norming=0.0;
		for (q=0; q<k; q++)
			norming += id_vector[q];

		cblas_dscal(k,1.0/norming,id_vector,1);

		memset(thread_gram_matrix,0,sizeof(double)*k*k);
		cblas_dger(CblasColMajor,k,k,1.0,id_vector,1,id_vector,1,thread_gram_matrix,k);

		// put weights into W matrix
		W_matrix[N*i+i] += 1.0;
		for (q=0; q<k; q++)
		{
			W_matrix[N*i+neighborhood_matrix[i*k+q]] -= id_vector[q];
			W_matrix[N*neighborhood_matrix[i*k+q]+i] -= id_vector[q];
		}
		for (q=0; q<k; q++)
		{
			for (p=0; p<k; p++)
				W_matrix[N*neighborhood_matrix[i*k+q]+neighborhood_matrix[i*k+p]]+=thread_gram_matrix[q*k+p];
		}
	}
	SG_FREE(ini_dots);

	return NULL;
}

double* eigendecomposition_embedding(
		double* weight_matrix, 
		int N, 
		int target_dimension, 
		bool use_arpack)
{
	int i,j;
	// get eigenvectors with ARPACK or LAPACK
	int eigenproblem_status = 0;

	double* eigenvalues_vector = NULL;
	double* eigenvectors = NULL;
	double* nullspace_features = NULL;
	if (use_arpack)
	{
#ifndef HAVE_ARPACK
		SG_ERROR("ARPACK is not supported in this configuration.\n");
#endif
		// using ARPACK (faster)
		eigenvalues_vector = SG_MALLOC(double, target_dimension+1);
#ifdef HAVE_ARPACK
		shogun::arpack_dsxupd(weight_matrix,NULL,false,N,target_dimension+1,"LA",true,3,true,false,-1e-6,0.0,
		                      eigenvalues_vector,weight_matrix,eigenproblem_status);
#endif
		nullspace_features = SG_MALLOC(float64_t, N*target_dimension);
		for (i=0; i<target_dimension; i++)
		{
			for (j=0; j<N; j++)
				nullspace_features[j*target_dimension+i] = weight_matrix[j*(target_dimension+1)+i+1];
		}
		SG_FREE(eigenvalues_vector);
	}
	else
	{
		// using LAPACK (slower)
		eigenvalues_vector = SG_MALLOC(float64_t, N);
		eigenvectors = SG_MALLOC(float64_t,(target_dimension+1)*N);
		shogun::wrap_dsyevr('V','U',N,weight_matrix,N,2,target_dimension+2,eigenvalues_vector,eigenvectors,&eigenproblem_status);
		nullspace_features = SG_MALLOC(float64_t, N*target_dimension);
		// LAPACKed eigenvectors
		for (i=0; i<target_dimension; i++)
		{
			for (j=0; j<N; j++)
				nullspace_features[j*target_dimension+i] = eigenvectors[i*N+j];
		}
		SG_FREE(eigenvectors);
		SG_FREE(eigenvalues_vector);
	}
	return nullspace_features;
}

int* kernel_neighbors_matrix(
		int N, 
		int k, 
		double (*kernel)(int, int, void*), 
		void* user_data)
{
	int i;
	int* neighborhood_matrix = SG_MALLOC(int, N*k);
	double max_dist = 0.0;
	for (i=0; i<N; i++)
		max_dist = MAX(kernel(i,i,user_data),max_dist);

	shogun::CoverTree<DISTANCE_COVERTREE_POINT>* cover_tree = new shogun::CoverTree<DISTANCE_COVERTREE_POINT>(max_dist);

	for (i=0; i<N; i++)
		cover_tree->insert(DISTANCE_COVERTREE_POINT(i,kernel,user_data));
	
	for (i=0; i<N; i++)
	{
		std::vector<DISTANCE_COVERTREE_POINT> neighbors = 
		   cover_tree->kNearestNeighbors(DISTANCE_COVERTREE_POINT(i,kernel,user_data),k+1);
		
		for (std::size_t m=1; m < unsigned(k+1); m++)
			neighborhood_matrix[i*k+m-1] = neighbors[m].point_index;
	}

	delete cover_tree;

	return neighborhood_matrix;
}

