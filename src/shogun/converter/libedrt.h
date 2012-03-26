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

struct DISTANCE_COVERTREE_POINT
{
public:

	DISTANCE_COVERTREE_POINT(int index, double (*distance_f)(int, int, void*), void* user_d)
	{
		point_index = index;
		distance_function = distance_f;
		user_data = user_d;
		kii = distance_function(point_index, point_index, user_data);
	}

	inline double distance(const DISTANCE_COVERTREE_POINT& p) const
	{
		return kii+p.kii-2.0*distance_function(point_index, p.point_index, user_data); 
	}
	inline bool operator==(const DISTANCE_COVERTREE_POINT& p) const
	{ 
		return (p.point_index==point_index); 
	}

	int point_index;
	double kii;
	double (*distance_function)(int, int, void*);
	void* user_data;
};

struct edrt_options_t
{
	edrt_options_t()
	{
		method = KERNEL_LOCALLY_LINEAR_EMBEDDING;
		num_threads = 1;
		use_arpack = true;
		use_superlu = true;
	}
	edrt_method_t method;
	int num_threads;
	bool use_arpack;
	bool use_superlu;
};

int edrt_embedding(
		const edrt_options_t& options,
		const int target_dimension, /* target dimensionality of embedding */
		const int N, /* number of vectors to embed */
		const int dimension, /* dimension of feature vectors */
		const int k, /* number of neighbors */
		double (*distance)(int, int, void*), /* distance function between feature vectors */
		double (*kernel)(int, int, void*), /* kernel function between feature vectors */
		double* (*access_feature_vector)(int, void*), /* feature vector access */
		void (*free_feature_vector)(int, void*), /* free feature vector */
		void* user_data, /* user data for callbacks */
		double **output /* output */);

double* klle_weight_matrix(
		int* neighborhood_matrix, /* indices of neighbors for each vector*/
		int N, /* number of vectors to embed */
		int k, /* number of neighbors */
		int matrix_k, /* actual k for neighborhood matrix */
		int num_threads, /* number of threads */
		double (*kernel)(int, int, void*), /* kernel function between feature vectors */
		void* user_data /* user data for callbacks */);

void* klle_weight_matrix_thread(void* parameter_map);

double* eigendecomposition_embedding(
		double* weight_matrix, 
		int N, 
		int target_dimension, 
		bool use_arpack);

int* kernel_neighbors_matrix(
		int N, 
		int k, 
		double (*kernel)(int, int, void*),
		void* user_data);

#endif /* libedrt_h_ */
