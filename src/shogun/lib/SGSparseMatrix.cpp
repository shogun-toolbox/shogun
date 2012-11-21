#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>

namespace shogun {

template <class T>
SGSparseMatrix<T>::SGSparseMatrix() : SGReferencedData()
{
	init_data();
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
		index_t num_vec, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_vectors(num_vec), num_features(num_feat),
	sparse_matrix(vecs)
{
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(index_t num_feat, index_t num_vec, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_vectors(num_vec), num_features(num_feat)
{
	sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
	for (int32_t i=0; i<num_vectors; i++)
	{
		new (&sparse_matrix[i]) SGSparseVector<T>();
		sparse_matrix[i] = SGSparseVector<T>(num_feat);
	}
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(const SGSparseMatrix &orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

template <class T>
SGSparseMatrix<T>::~SGSparseMatrix()
{
	unref();
}

template <class T>
void SGSparseMatrix<T>::copy_data(const SGReferencedData& orig)
{
	sparse_matrix = ((SGSparseMatrix*)(&orig))->sparse_matrix;
	num_vectors = ((SGSparseMatrix*)(&orig))->num_vectors;
	num_features = ((SGSparseMatrix*)(&orig))->num_features;
}

template <class T>
void SGSparseMatrix<T>::init_data()
{
	sparse_matrix = NULL;
	num_vectors = 0;
	num_features = 0;
}

template <class T>
void SGSparseMatrix<T>::free_data()
{
	for (int32_t i=0; i<num_vectors; i++)
		(&sparse_matrix[i])->~SGSparseVector<T>();

	SG_FREE(sparse_matrix);
	num_vectors = 0;
	num_features = 0;
}

template class SGSparseMatrix<bool>;
template class SGSparseMatrix<char>;
template class SGSparseMatrix<int8_t>;
template class SGSparseMatrix<uint8_t>;
template class SGSparseMatrix<int16_t>;
template class SGSparseMatrix<uint16_t>;
template class SGSparseMatrix<int32_t>;
template class SGSparseMatrix<uint32_t>;
template class SGSparseMatrix<int64_t>;
template class SGSparseMatrix<uint64_t>;
template class SGSparseMatrix<float32_t>;
template class SGSparseMatrix<float64_t>;
template class SGSparseMatrix<floatmax_t>;
}

