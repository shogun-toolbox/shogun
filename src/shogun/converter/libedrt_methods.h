#ifndef libedrt_methods_h_
#define libedrt_methods_h
#include <shogun/lib/config.h>
#include <shogun/base/Parallel.h>

struct covertree_point_t
{
public:

	covertree_point_t(int index, double (*distance_f)(int, int, const void*), const void* user_d)
	{
		point_index = index;
		distance_function = distance_f;
		user_data = user_d;
		kii = distance_function(point_index, point_index, user_data);
	}

	inline double distance(const covertree_point_t& p) const
	{
		return kii+p.kii-2.0*distance_function(point_index, p.point_index, user_data); 
	}
	inline bool operator==(const covertree_point_t& p) const
	{ 
		return (p.point_index==point_index); 
	}

	int point_index;
	double kii;
	double (*distance_function)(int, int, const void*);
	const void* user_data;
};

struct thread_parameters_t
{
	int N, k, target_dimension, thread, num_threads;
	const int* neighborhood_matrix;
	double (*distance)(int, int, const void*);
	double (*kernel)(int, int, const void*);
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T* pthread_lock;
#endif
	const void* user_data;
	double* W_matrix;
};

int* kernel_neighbors_matrix(
		int N, 
		int k, 
		double (*kernel)(int, int, const void*), 
		const void* user_data);

// Kernel Locally Linear Embedding

double* klle_weight_matrix(
		int* neighborhood_matrix,
		int N,
		int k,
		int matrix_k,
		int num_threads,
		double reconstruction_shift,
		double (*kernel)(int, int, const void*),
		const void* user_data);

void* klle_weight_matrix_thread(void* params);

// Kernel Local Tangent Space Alignment

double* kltsa_weight_matrix(
		int* neighborhood_matrix,
		int N,
		int k,
		int matrix_k,
		int target_dim,
		int num_threads,
		double (*kernel)(int, int, const void*),
		const void* user_data);

void* kltsa_weight_matrix_thread(void* params);

// Diffusion Maps

double* diffusion_maps_embedding(
		int N,
		int t,
		int target_dimension,
		double (*kernel)(int, int, const void*),
		const void* user_data);

// Embedding methods

double* eigendecomposition_embedding(
		double* weight_matrix, 
		int N, 
		int target_dimension,
		bool use_arpack,
		double nullspace_shift);

#endif
