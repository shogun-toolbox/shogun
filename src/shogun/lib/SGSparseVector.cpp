#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/File.h>

namespace shogun {

template <class T>
SGSparseVector<T>::SGSparseVector() : SGReferencedData()
{
	init_data();
}

template <class T>
SGSparseVector<T>::SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
		bool ref_counting) :
		SGReferencedData(ref_counting),
		num_feat_entries(num_entries), features(feats)
{
}

template <class T>
SGSparseVector<T>::SGSparseVector(index_t num_entries, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_feat_entries(num_entries)
{
	features = SG_MALLOC(SGSparseVectorEntry<T>, num_feat_entries);
}

template <class T>
SGSparseVector<T>::SGSparseVector(const SGSparseVector& orig) :
	SGReferencedData(orig)
{
	copy_data(orig);
}

template <class T>
SGSparseVector<T>::~SGSparseVector()
{
	unref();
}

template <class T>
T SGSparseVector<T>::dense_dot(T alpha, T* vec, int32_t dim, T b)
{
	ASSERT(vec)
	T result=b;

	if (features)
	{
		for (int32_t i=0; i<num_feat_entries; i++)
		{
			result+=alpha*vec[features[i].feat_index]
				*features[i].entry;
		}
	}

	return result;
}

template <class T>
T SGSparseVector<T>::sparse_dot(const SGSparseVector<T>& v)
{
	return sparse_dot(*this, v);
}

template <class T>
T SGSparseVector<T>::sparse_dot(const SGSparseVector<T>& a, const SGSparseVector<T>& b)
{
	if (a.num_feat_entries == 0 || b.num_feat_entries == 0)
		return 0;

	int32_t cmp = cmp_dot_prod_symmetry_fast(a.num_feat_entries, b.num_feat_entries);

	if (cmp == 0) // symmetric
	{
		return dot_prod_symmetric(a, b);
	}
	else if (cmp > 0) // b has more element
	{
		return dot_prod_asymmetric(a, b);
	}
	else // a has more element
	{
		return dot_prod_asymmetric(b, a);
	}
}

template<class T> void SGSparseVector<T>::load(CFile* loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	loader->get_sparse_vector(features, num_feat_entries);
	SG_RESET_LOCALE;
}

template<class T> void SGSparseVector<T>::save(CFile* saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_sparse_vector(features, num_feat_entries);
	SG_RESET_LOCALE;
}
		



template <class T>
void SGSparseVector<T>::copy_data(const SGReferencedData& orig)
{
	num_feat_entries = ((SGSparseVector*)(&orig))->num_feat_entries;
	features = ((SGSparseVector*)(&orig))->features;
}

template <class T>
void SGSparseVector<T>::init_data()
{
	num_feat_entries = 0;
	features = NULL;
}

template <class T>
void SGSparseVector<T>::free_data()
{
	num_feat_entries = 0;
	SG_FREE(features);
}

template <class T>
int32_t SGSparseVector<T>::cmp_dot_prod_symmetry_fast(index_t alen, index_t blen)
{
	if (alen > blen) // no need for floats here
	{
		return (blen * CMath::floor_log(alen) < alen) ? -1 : 0;
	}
	else // alen <= blen
	{
		return (alen * CMath::floor_log(blen) < blen) ? 1 : 0;
	}
}

template <class T>
T SGSparseVector<T>::dot_prod_asymmetric(const SGSparseVector<T>& a, const SGSparseVector<T>& b)
{
	T dot_prod = 0;
	for(index_t b_idx = 0; b_idx < b.num_feat_entries; ++b_idx)
	{
		const T tmp = b.features[b_idx].entry;
		if (a.features[a.num_feat_entries-1].feat_index < b.features[b_idx].feat_index)
			break;
		for (index_t a_idx = 0; a_idx < a.num_feat_entries; ++a_idx)
		{
			if (a.features[a_idx].feat_index == b.features[b_idx].feat_index)
				dot_prod += tmp * a.features[a_idx].entry;
		}
	}
	return dot_prod;
}

template <class T>
T SGSparseVector<T>::dot_prod_symmetric(const SGSparseVector<T>& a, const SGSparseVector<T>& b)
{
	ASSERT(a.num_feat_entries > 0 && b.num_feat_entries > 0)
	T dot_prod = 0;
	index_t a_idx = 0, b_idx = 0;
	while (true)
	{
		if (a.features[a_idx].feat_index == b.features[b_idx].feat_index)
		{
			dot_prod += a.features[a_idx].entry * b.features[b_idx].entry;

			a_idx++;
			if (a.num_feat_entries == a_idx)
				break;
			b_idx++;
			if (b.num_feat_entries == b_idx)
				break;
		}
		else if (a.features[a_idx].feat_index < b.features[b_idx].feat_index)
		{
			a_idx++;
			if (a.num_feat_entries == a_idx)
				break;
		}
		else
		{
			b_idx++;
			if (b.num_feat_entries == b_idx)
				break;
		}
	}
	return dot_prod;
}

template class SGSparseVector<bool>;
template class SGSparseVector<char>;
template class SGSparseVector<int8_t>;
template class SGSparseVector<uint8_t>;
template class SGSparseVector<int16_t>;
template class SGSparseVector<uint16_t>;
template class SGSparseVector<int32_t>;
template class SGSparseVector<uint32_t>;
template class SGSparseVector<int64_t>;
template class SGSparseVector<uint64_t>;
template class SGSparseVector<float32_t>;
template class SGSparseVector<float64_t>;
template class SGSparseVector<floatmax_t>;
}

