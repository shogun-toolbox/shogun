#include <shogun/base/init.h>
#include <shogun/features/hashed/HashedDenseFeatures.h>
#include <shogun/features/hashed/HashedSparseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/PolyKernel.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	int32_t num_vectors = 5;
	int32_t dim = 20;

	SGMatrix<int32_t> mat(dim, num_vectors);
	for (index_t v=0; v<num_vectors; v++)
	{
		for (index_t d=0; d<dim; d++)
			mat(d,v) = CMath::random(-dim, dim);
	}

	int32_t hashing_dim = 12;
	CHashedDenseFeatures<int32_t>* h_dense_feats = new CHashedDenseFeatures<int32_t>(mat, hashing_dim);

	CSparseFeatures<int32_t>* sparse_feats = new CSparseFeatures<int32_t>(mat);
	CHashedSparseFeatures<int32_t>* h_sparse_feats = new CHashedSparseFeatures<int32_t>(sparse_feats, hashing_dim);

	SG_REF(h_dense_feats);
	CPolyKernel* kernel = new CPolyKernel(h_dense_feats, h_dense_feats, 1, false);
	SGMatrix<float64_t> dense_mt = kernel->get_kernel_matrix();
	SG_UNREF(kernel);

	SG_REF(h_sparse_feats);
	kernel = new CPolyKernel(h_sparse_feats, h_sparse_feats, 1, false);
	SGMatrix<float64_t> sparse_mt = kernel->get_kernel_matrix();
	SG_UNREF(kernel);

	for (index_t i=0; i<dense_mt.num_rows; i++)
	{
		for (index_t j=0; j<dense_mt.num_cols; j++)
			ASSERT(dense_mt(i,j)==sparse_mt(i,j))
	}

	dense_mt.display_matrix("Dense matrix");
	sparse_mt.display_matrix("Sparse matrix");

	SG_UNREF(h_dense_feats);
	SG_UNREF(h_sparse_feats);
	exit_shogun();
}
