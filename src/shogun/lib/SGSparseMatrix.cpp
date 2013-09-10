#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/io/File.h>

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
const SGVector<complex64_t> SGSparseMatrix<complex64_t>::operator*(
	SGVector<float64_t> v) const
{
	SGVector<complex64_t> result(num_vectors);
	REQUIRE(v.vlen==num_features,
		"Dimension mismatch! %d vs %d\n",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template <> template <>
const SGVector<complex64_t> SGSparseMatrix<complex64_t>::operator*(
	SGVector<int32_t> v) const
{
	SGVector<complex64_t> result(num_vectors);
	REQUIRE(v.vlen==num_features,
		"Dimension mismatch! %d vs %d\n",
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
	REQUIRE(v.vlen==num_features,
		"Dimension mismatch! %d vs %d\n",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template<class T>
void SGSparseMatrix<T>::load(CFile* loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	loader->get_sparse_matrix(sparse_matrix, num_features, num_vectors);
	SG_RESET_LOCALE;
}

template<>
void SGSparseMatrix<complex64_t>::load(CFile* loader)
{
	SG_SERROR("SGSparseMatrix::load():: Not supported for complex64_t");
}

template<class T>
void SGSparseMatrix<T>::save(CFile* saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_sparse_matrix(sparse_matrix, num_features, num_vectors);
	SG_RESET_LOCALE;
}

template<>
void SGSparseMatrix<complex64_t>::save(CFile* saver)
{
	SG_SERROR("SGSparseMatrix::save():: Not supported for complex64_t");
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
	SG_FREE(sparse_matrix);
	num_vectors = 0;
	num_features = 0;
}

template<class T> CRegressionLabels* SGSparseMatrix<T>::load_svmlight_file(CLibSVMFile* file,
		bool do_sort_features)
{
	ASSERT(file)

	CRegressionLabels* labels=NULL;
	float64_t* raw_labels;
	file->get_sparse_matrix(sparse_matrix, num_features, num_vectors,
					raw_labels, true);

	SGVector<float64_t> labels_data(raw_labels, num_vectors);
	labels=new CRegressionLabels(labels_data);
	SG_REF(labels);

	if (do_sort_features)
		sort_features();

	return labels;
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

	int32_t* index=SG_CALLOC(int32_t, num_vectors);

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

template<class T> void SGSparseMatrix<T>::write_svmlight_file(CLibSVMFile* file,
		CRegressionLabels* labels)
{
	ASSERT(file)
	int32_t num=labels->get_num_labels();
	ASSERT(num>0)
	ASSERT(num==num_vectors)

	SGVector<float64_t> data_labels=labels->get_labels();
	float64_t* raw_labels=data_labels.vector;
	file->set_sparse_matrix(sparse_matrix, num_features, num_vectors,
			raw_labels);
}

template <> void SGSparseMatrix<complex64_t>::write_svmlight_file(CLibSVMFile* file, CRegressionLabels* label) { }

template<> CRegressionLabels* SGSparseMatrix<complex64_t>::load_svmlight_file(CLibSVMFile* file, bool do_sort_features) { return NULL; }

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
template class SGSparseMatrix<complex64_t>;
}
