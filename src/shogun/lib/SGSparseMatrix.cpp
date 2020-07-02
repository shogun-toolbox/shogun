#include <shogun/base/range.h>
#include <shogun/io/File.h>
#include <shogun/io/LibSVMFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
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
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(SGMatrix<T> dense) : SGReferencedData()
{
	from_dense(dense);
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

template <> template <>
const SGVector<complex128_t> SGSparseMatrix<complex128_t>::operator*(
	SGVector<float64_t> v) const
{
	SGVector<complex128_t> result(num_vectors);
	require(v.vlen==num_features,
		"Dimension mismatch! {} vs {}",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template <> template <>
const SGVector<complex128_t> SGSparseMatrix<complex128_t>::operator*(
	SGVector<int32_t> v) const
{
	SGVector<complex128_t> result(num_vectors);
	require(v.vlen==num_features,
		"Dimension mismatch! {} vs {}",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template <> template <>
const SGVector<float64_t> SGSparseMatrix<float64_t>::operator*(
	SGVector<int32_t> v) const
{
	SGVector<float64_t> result(num_vectors);
	require(v.vlen==num_features,
		"Dimension mismatch! {} vs {}",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template<class T>
void SGSparseMatrix<T>::load(const std::shared_ptr<File>& loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	loader->get_sparse_matrix(sparse_matrix, num_features, num_vectors);
	SG_RESET_LOCALE;
}

template<>
void SGSparseMatrix<complex128_t>::load(const std::shared_ptr<File>& loader)
{
	error("SGSparseMatrix::load():: Not supported for complex128_t");
}

template<class T> SGVector<float64_t> SGSparseMatrix<T>::load_with_labels(const std::shared_ptr<File>& file, bool do_sort_features)
{
	ASSERT(file)
	auto libsvm_file = std::dynamic_pointer_cast<LibSVMFile>(file);
	require(libsvm_file, "Expected file to be of type LibSVMFile");

	float64_t* raw_labels;
	libsvm_file->get_sparse_matrix(sparse_matrix, num_features, num_vectors,
					raw_labels, true);

	SGVector<float64_t> labels(raw_labels, num_vectors);

	if (do_sort_features)
		sort_features();

	return labels;
}

template<> SGVector<float64_t> SGSparseMatrix<complex128_t>::load_with_labels(const std::shared_ptr<File>& file, bool do_sort_features) { return SGVector<float64_t>(); }


template<class T>
void SGSparseMatrix<T>::save(const std::shared_ptr<File>& saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_sparse_matrix(sparse_matrix, num_features, num_vectors);
	SG_RESET_LOCALE;
}

template<>
void SGSparseMatrix<complex128_t>::save(const std::shared_ptr<File>& saver)
{
	error("SGSparseMatrix::save():: Not supported for complex128_t");
}

template<class T> void SGSparseMatrix<T>::save_with_labels(const std::shared_ptr<File>& file,
		SGVector<float64_t> labels)
{
	ASSERT(file)
	auto libsvm_file = std::dynamic_pointer_cast<LibSVMFile>(file);
	require(libsvm_file, "Expected file to be of type LibSVMFile");
	int32_t num=labels.vlen;
	ASSERT(num>0)
	ASSERT(num==num_vectors)

	float64_t* raw_labels=labels.vector;
	libsvm_file->set_sparse_matrix(sparse_matrix, num_features, num_vectors,
			raw_labels);
}

template <> void SGSparseMatrix<complex128_t>::save_with_labels(const std::shared_ptr<File>& saver, SGVector<float64_t> labels) { }


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
	SG_FREE(sparse_matrix);
	num_vectors = 0;
	num_features = 0;
}

template<class T> SGSparseMatrix<T> SGSparseMatrix<T>::get_transposed()
{
	SGSparseMatrix<T> sfm(num_vectors, num_features);

	int32_t* hist=SG_CALLOC(int32_t, num_features);

	// count the lengths of future feature vectors
	for (int32_t v=0; v<num_vectors; v++)
	{
		SGSparseVector<T> sv=sparse_matrix[v];

		for (int32_t i=0; i<sv.num_feat_entries; i++)
			hist[sv.features[i].feat_index]++;
	}

	for (int32_t v=0; v<num_features; v++)
		sfm[v]=SGSparseVector<T>(hist[v]);

	SG_FREE(hist);

	int32_t* index=SG_CALLOC(int32_t, num_features);

	// fill future feature vectors with content
	for (int32_t v=0; v<num_vectors; v++)
	{
		SGSparseVector<T> sv=sparse_matrix[v];

		for (int32_t i=0; i<sv.num_feat_entries; i++)
		{
			int32_t vidx=sv.features[i].feat_index;
			int32_t fidx=v;
			sfm[vidx].features[index[vidx]].feat_index=fidx;
			sfm[vidx].features[index[vidx]].entry=sv.features[i].entry;
			index[vidx]++;
		}
	}

	SG_FREE(index);
	return sfm;
}


template<class T> void SGSparseMatrix<T>::sort_features()
{
	for (int32_t i=0; i<num_vectors; i++)
	{
		sparse_matrix[i].sort_features();
	}
}

template<class T> void SGSparseMatrix<T>::from_dense(SGMatrix<T> full)
{
	T* src=full.matrix;
	int32_t num_feat=full.num_rows;
	int32_t num_vec=full.num_cols;

	require(num_vec>0, "Matrix should have > 0 vectors!");

	io::info("converting dense feature matrix to sparse one");
		int32_t* num_feat_entries=SG_MALLOC(int, num_vec);


	int64_t num_total_entries=0;

	// count nr of non sparse features
	for (int32_t i=0; i<num_vec; i++)
	{
		num_feat_entries[i]=0;
		for (int32_t j=0; j<num_feat; j++)
		{
			if (src[i*((int64_t) num_feat) + j] != static_cast<T>(0))
				num_feat_entries[i]++;
		}
	}

	num_features=num_feat;
	num_vectors=num_vec;
	sparse_matrix=SG_MALLOC(SGSparseVector<T>,num_vec);

	for (int32_t i=0; i< num_vec; i++)
	{
		sparse_matrix[i]=SGSparseVector<T>(num_feat_entries[i]);
		int32_t sparse_feat_idx=0;

		for (int32_t j=0; j< num_feat; j++)
		{
			int64_t pos= i*num_feat + j;

			if (src[pos] != static_cast<T>(0))
			{
				sparse_matrix[i].features[sparse_feat_idx].entry=src[pos];
				sparse_matrix[i].features[sparse_feat_idx].feat_index=j;
				sparse_feat_idx++;
				num_total_entries++;
			}
		}
	}

	io::info("sparse feature matrix has {} entries (full matrix had {}, sparsity {:2.2f}%%)",
			num_total_entries, int64_t(num_feat)*num_vec, (100.0*num_total_entries)/(int64_t(num_feat)*num_vec));
	SG_FREE(num_feat_entries);
}

template <class T>
bool SGSparseMatrix<T>::operator==(const SGSparseMatrix<T>& other) const
{
	if (num_vectors != other.num_vectors)
		return false;

	if (num_features != other.num_features)
		return false;

	if (sparse_matrix != other.sparse_matrix)
		return false;

	return true;
}

template <class T>
bool SGSparseMatrix<T>::equals(const SGSparseMatrix<T>& other) const
{
	// same instance
	if (*this == other)
		return true;

	// different size
	if (num_vectors != other.num_vectors || num_features != other.num_features)
		return false;

	for (auto i : range(num_vectors))
	{
		if (!sparse_matrix[i].equals(other.sparse_matrix[i]))
			return false;
	}
	return true;
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
template class SGSparseMatrix<complex128_t>;
}
