#ifndef libedrt_h_
#define libedrt_h_
#define libedrt_malloc(type, length) malloc(length,sizeof(type))
#define libedrt_calloc(type, length) calloc(length,sizeof(type))

enum edrt_method_t
{
	KERNEL_LOCALLY_LINEAR_EMBEDDING,
	NEIGHBORHOOD_PRESERVING_EMBEDDING,
	KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT,
	LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT,
	HESSIAN_LOCALLY_LINEAR_EMBEDDING,
	LAPLACIAN_EIGENMAPS,
	LOCALITY_PRESERVING_PROJECTIONS,
	DIFFUSION_MAPS,
	ISOMAP,
	MULTIDIMENSIONAL_SCALING
};

struct edrt_options_t
{
	edrt_options_t()
	{
		method = KERNEL_LOCALLY_LINEAR_EMBEDDING;
		num_threads = 1;
		use_arpack = true;
		use_superlu = true;
		klle_reconstruction_shift = 1e-3;
		diffusion_maps_t = 1;
		nullspace_shift = 1e-9;
	}
	// EDRT method
	edrt_method_t method;
	// number of threads
	int num_threads;
	// true if ARPACK should be used
	bool use_arpack;
	// true if SuperLU should be used
	bool use_superlu;
	// kernel LLE reconstruction shift
	double klle_reconstruction_shift;
	// diffusion maps t
	int diffusion_maps_t;
	// nullspace regularization shift
	double nullspace_shift;
};

int edrt_embedding(
		const edrt_options_t& options,
		const int target_dimension, /* target dimensionality of embedding */
		const int N, /* number of vectors to embed */
		const int dimension, /* dimension of feature vectors */
		const int k, /* number of neighbors */
		double (*distance)(int, int, const void*), /* distance function between feature vectors */
		double (*kernel)(int, int, const void*), /* kernel function between feature vectors */
		double* (*access_feature_vector)(int, const void*), /* feature vector access */
		void (*free_feature_vector)(int, const void*), /* free feature vector */
		const void* user_data, /* user data for callbacks */
		double **output /* output */);

#endif /* libedrt_h_ */
