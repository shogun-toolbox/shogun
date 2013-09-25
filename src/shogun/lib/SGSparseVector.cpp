#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGVector.h>
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
			if (features[i].feat_index<dim)
				result+=alpha*vec[features[i].feat_index]*features[i].entry;
		}
	}

	return result;
}

template <class T>
template <typename ST>
T SGSparseVector<T>::dense_dot(SGVector<ST> vec)
{
	ASSERT(vec)
	T result(0.0);

	if (features)
	{
		for (int32_t i=0; i<num_feat_entries; i++)
		{
			if (features[i].feat_index<vec.vlen)
				result+=static_cast<T>(vec[features[i].feat_index])
					*features[i].entry;
		}
	}

	return result;
}

template complex128_t SGSparseVector<complex128_t>::dense_dot<float64_t>(SGVector<float64_t>);
template complex128_t SGSparseVector<complex128_t>::dense_dot<int32_t>(SGVector<int32_t> vec);
template float64_t SGSparseVector<float64_t>::dense_dot<int32_t>(SGVector<int32_t> vec);

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

template<class T>
int32_t SGSparseVector<T>::get_num_dimensions()
{
	if (!features)
		return 0;

	int32_t dimensions = -1;
	for (index_t i=0; i<num_feat_entries; i++)
	{
		if (features[i].feat_index > dimensions)
		{
			dimensions = features[i].feat_index;
		}
	}

	return dimensions+1;
}

template<class T>
void SGSparseVector<T>::sort_features(bool stable_pointer)
{
	if (!num_feat_entries)
		return;

	// remember old pointer to enforce quarantee
	const SGSparseVectorEntry<T>* old_features_ptr = features;

	int32_t* feat_idx=SG_MALLOC(int32_t, num_feat_entries);
	for (index_t j=0; j<num_feat_entries; j++)
	{
		feat_idx[j]=features[j].feat_index;
	}

	CMath::qsort_index(feat_idx, features, num_feat_entries);
	SG_FREE(feat_idx);

	for (index_t j=1; j<num_feat_entries; j++)
	{
		REQUIRE(features[j-1].feat_index <= features[j].feat_index,
				"sort_features(): failed sanity check %d <= %d after sorting (comparing indices features[%d] <= features[%d], features=%d)\n",
				features[j-1].feat_index, features[j].feat_index, j-1, j, num_feat_entries);
	}

	// compression: removing zero-entries and merging features with same index
	int32_t last_index = 0;
	for (index_t j=1; j<num_feat_entries; j++)
	{
		// always true, but kept for future changes
		REQUIRE(last_index < j,
			"sort_features(): target index %d must not exceed source index j=%d",
			last_index, j);
		REQUIRE(features[last_index].feat_index <= features[j].feat_index,
			"sort_features(): failed sanity check %d = features[%d].feat_index <= features[%d].feat_index = %d\n",
			features[last_index].feat_index, last_index, j, features[j].feat_index);

		// merging of features with same index
		if (features[last_index].feat_index == features[j].feat_index)
		{
			features[last_index].entry += features[j].entry;
			continue;
		}

		// only skip to next element if current one is not zero
		if (features[last_index].entry != 0.0)
		{
			last_index++;
		}

		features[last_index] = features[j];
	}

	// remove single first element if zero (not caught by loop)
	if (features[last_index].entry == 0.0)
	{
		last_index--;
	}

	int32_t new_feat_count = last_index+1;
	ASSERT(new_feat_count <= num_feat_entries);

	// shrinking vector
	if (!stable_pointer)
	{
		SG_SINFO("shrinking vector from %d to %d\n", num_feat_entries, new_feat_count);
		features = SG_REALLOC(SGSparseVectorEntry<T>, features, num_feat_entries, new_feat_count);
	}
	num_feat_entries = new_feat_count;

	for (index_t j=1; j<num_feat_entries; j++)
	{
		REQUIRE(features[j-1].feat_index < features[j].feat_index,
				"sort_features(): failed sanity check %d < %d after sorting (comparing indices features[%d] < features[%d], features=%d)\n",
				features[j-1].feat_index, features[j].feat_index, j-1, j, num_feat_entries);
	}

	// compare with old pointer to enforce quarantee
	if (stable_pointer) {
		ASSERT(old_features_ptr == features);
	}
	return;
}

template<class T>
T SGSparseVector<T>::get_feature(int32_t index)
{
	T ret = 0;
	if (features)
	{
		for (index_t i=0; i<num_feat_entries; i++)
			if (features[i].feat_index==index)
				ret+=features[i].entry ;
	}

	return ret ;
}

template<class T>
SGVector<T> SGSparseVector<T>::get_dense()
{
	SGVector<T> dense;

	if (features)
	{
		dense.resize_vector(get_num_dimensions());
		dense.zero();

		for (index_t i=0; i<num_feat_entries; i++)
		{
			dense.vector[features[i].feat_index] += features[i].entry;
		}
	}

	return dense;
}

template<class T>
SGVector<T> SGSparseVector<T>::get_dense(int32_t dimension)
{
	SGVector<T> dense(dimension);
	dense.zero();

	if (features)
	{
		REQUIRE(get_num_dimensions() <= dimension, "get_dense(dimension=%d): sparse dimension %d exceeds requested dimension\n",
				dimension, get_num_dimensions());

		for (index_t i=0; i<num_feat_entries; i++)
		{
			dense.vector[features[i].feat_index] += features[i].entry;
		}
	}

	return dense;
}

template<class T>
SGSparseVector<T> SGSparseVector<T>::clone() const
{
	SGSparseVectorEntry <T> * copy = SG_MALLOC(SGSparseVectorEntry <T>, num_feat_entries);
	memcpy(copy, features, num_feat_entries * sizeof(SGSparseVectorEntry <T>));
	return SGSparseVector<T>(copy, num_feat_entries);
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

template <>
void SGSparseVector<complex128_t>::load(CFile* loader)
{
	SG_SERROR("SGSparseVector::load():: Not supported for complex128_t\n");
}

template <>
void SGSparseVector<complex128_t>::save(CFile* saver)
{
	SG_SERROR("SGSparseVector::save():: Not supported for complex128_t\n");
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

template <>
void SGSparseVector<bool>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%d", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry ? 1 : 0);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<char>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%c", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<int8_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%d", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<uint8_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%u", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<int16_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%d", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<uint16_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%u", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<int32_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%d", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<uint32_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%u", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<int64_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%lld", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<uint64_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%llu ", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<float32_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%g", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<float64_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%.18g", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<floatmax_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:%.36Lg", prefix, i==0 ? "" : " ", features[i].feat_index, features[i].entry);
	SG_SPRINT("%s]\n", prefix);
}

template <>
void SGSparseVector<complex128_t>::display_vector(const char* name, const char* prefix)
{
	SG_SPRINT("%s%s=[", prefix, name);
	for (int32_t i=0; i<num_feat_entries; i++)
		SG_SPRINT("%s%s%d:(%.18lg+i%.18lg)", prefix, i==0 ? "" : " ", features[i].feat_index,
				features[i].entry.real(), features[i].entry.imag());
	SG_SPRINT("%s]\n", prefix);
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
template class SGSparseVector<complex128_t>;
}

