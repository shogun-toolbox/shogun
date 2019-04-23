/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Soumyajit De, Soeren Sonnenburg, Heiko Strathmann, 
 *          Weijie Lin, Abinash Panda
 */

#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/File.h>

namespace shogun
{

template <class T>
SGSparseVector<T>::SGSparseVector() : SGReferencedData()
{
	init_data();
}

template <class T>
SGSparseVector<T>::SGSparseVector(pointer feats, size_type num_entries,
                                  bool ref_counting) :
	SGReferencedData(ref_counting),
	num_feat_entries(num_entries), features(feats)
{
}

template <class T>
SGSparseVector<T>::SGSparseVector(size_type num_entries, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_feat_entries(num_entries)
{
	features = SG_MALLOC(value_type, num_feat_entries);
}

template <class T>
SGSparseVector<T>::SGSparseVector(const SGSparseVector &orig) :
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
T SGSparseVector<T>::dense_dot(T alpha, T * vec, int32_t dim, T b)
{
	ASSERT(vec)
	T result = b;

	if (features)
	{
		for (int32_t i = 0; i < num_feat_entries; i++)
		{
			if (features[i].feat_index < dim)
			{
				result += alpha * vec[features[i].feat_index] * features[i].entry;
			}
		}
	}

	return result;
}

template <class T>
void SGSparseVector<T>::add_to_dense(T alpha, T * vec, int32_t dim, bool abs_val)
{
	require(vec, "vec must not be NULL");

	if (abs_val)
	{
		for (int32_t i = 0; i < num_feat_entries; i++)
		{
			vec[features[i].feat_index] += alpha*Math::abs(features[i].entry);
		}
	}
	else
	{
		for (int32_t i = 0; i < num_feat_entries; i++)
		{
			vec[features[i].feat_index] += alpha*features[i].entry;
		}
	}
}

template <class T>
template <typename ST>
T SGSparseVector<T>::dense_dot(SGVector<ST> vec)
{
	ASSERT(vec)
	T result(0.0);

	if (features)
	{
		for (int32_t i = 0; i < num_feat_entries; i++)
		{
			if (features[i].feat_index < vec.vlen)
				result += static_cast<T>(vec[features[i].feat_index])
				          * features[i].entry;
		}
	}

	return result;
}

template complex128_t SGSparseVector<complex128_t>::dense_dot<float64_t>(SGVector<float64_t>);
template complex128_t SGSparseVector<complex128_t>::dense_dot<int32_t>(SGVector<int32_t> vec);
template float64_t SGSparseVector<float64_t>::dense_dot<int32_t>(SGVector<int32_t> vec);

template <class T>
T SGSparseVector<T>::sparse_dot(const SGSparseVector<T> &v)
{
	return sparse_dot(*this, v);
}

template <class T>
T SGSparseVector<T>::sparse_dot(const SGSparseVector<T> &a, const SGSparseVector<T> &b)
{
	if (a.num_feat_entries == 0 || b.num_feat_entries == 0)
	{
		return 0;
	}

	if (!a.is_sorted() || !b.is_sorted())
	{
		return dot_prod_expensive_unsorted(a, b);
	}

	T dot_prod = 0;
	size_type a_idx = 0, b_idx = 0;

	while (a_idx < a.num_feat_entries && b_idx < b.num_feat_entries)
	{
		if (a.features[a_idx].feat_index < b.features[b_idx].feat_index)
		{
			a_idx++;
		}
		else if (a.features[a_idx].feat_index > b.features[b_idx].feat_index)
		{
			b_idx++;
		}
		else
		{
			// a.features[a_idx].feat_index == b.features[b_idx].feat_index
			dot_prod += a.features[a_idx].entry * b.features[b_idx].entry;
			a_idx++;
			b_idx++;
		}
	}

	return dot_prod;
}

template<class T>
int32_t SGSparseVector<T>::get_num_dimensions()
{
	if (!features)
	{
		return 0;
	}

	int32_t dimensions = -1;

	for (size_type i = 0; i < num_feat_entries; i++)
	{
		if (features[i].feat_index > dimensions)
		{
			dimensions = features[i].feat_index;
		}
	}

	return dimensions + 1;
}

template<class T>
void SGSparseVector<T>::sort_features(bool stable_pointer)
{
	if (!num_feat_entries)
	{
		return;
	}

	// remember old pointer to enforce quarantee
	const pointer old_features_ptr = features;

	int32_t * feat_idx = SG_MALLOC(int32_t, num_feat_entries);

	for (size_type j = 0; j < num_feat_entries; j++)
	{
		feat_idx[j] = features[j].feat_index;
	}

	Math::qsort_index(feat_idx, features, num_feat_entries);
	SG_FREE(feat_idx);

	for (size_type j = 1; j < num_feat_entries; j++)
	{
		require(features[j - 1].feat_index <= features[j].feat_index,
		        "sort_features(): failed sanity check {} <= {} after sorting (comparing indices features[{}] <= features[{}], features={})",
		        features[j - 1].feat_index, features[j].feat_index, j - 1, j, num_feat_entries);
	}

	// compression: removing zero-entries and merging features with same index
	int32_t last_index = 0;

	for (size_type j = 1; j < num_feat_entries; j++)
	{
		// always true, but kept for future changes
		require(last_index < j,
		        "sort_features(): target index {} must not exceed source index j={}",
		        last_index, j);
		require(features[last_index].feat_index <= features[j].feat_index,
		        "sort_features(): failed sanity check {} = features[{}].feat_index <= features[{}].feat_index = {}",
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

	int32_t new_feat_count = last_index + 1;
	ASSERT(new_feat_count <= num_feat_entries);

	// shrinking vector
	if (!stable_pointer)
	{
		io::info("shrinking vector from {} to {}", num_feat_entries, new_feat_count);
		features = SG_REALLOC(value_type, features, num_feat_entries, new_feat_count);
	}

	num_feat_entries = new_feat_count;

	for (size_type j = 1; j < num_feat_entries; j++)
	{
		require(features[j - 1].feat_index < features[j].feat_index,
		        "sort_features(): failed sanity check {} < {} after sorting (comparing indices features[{}] < features[{}], features={})",
		        features[j - 1].feat_index, features[j].feat_index, j - 1, j, num_feat_entries);
	}

	// compare with old pointer to enforce quarantee
	if (stable_pointer)
	{
		ASSERT(old_features_ptr == features);
	}

	return;
}

template<class T>
bool SGSparseVector<T>::is_sorted() const
{
	if (num_feat_entries == 0 || num_feat_entries == 1)
	{
		return true;
	}

	for (size_type j = 1; j < num_feat_entries; j++)
	{
		if (features[j - 1].feat_index >= features[j].feat_index)
		{
			return false;
		}
	}

	return true;
}

template<class T>
T SGSparseVector<T>::get_feature(int32_t index)
{
	T ret = 0;

	if (features)
	{
		for (size_type i = 0; i < num_feat_entries; i++)
			if (features[i].feat_index == index)
			{
				ret += features[i].entry ;
			}
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

		for (size_type i = 0; i < num_feat_entries; i++)
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
		require(get_num_dimensions() <= dimension, "get_dense(dimension={}): sparse dimension {} exceeds requested dimension",
		        dimension, get_num_dimensions());

		for (size_type i = 0; i < num_feat_entries; i++)
		{
			dense.vector[features[i].feat_index] += features[i].entry;
		}
	}

	return dense;
}

template<class T>
SGSparseVector<T> SGSparseVector<T>::clone() const
{
	pointer copy = SG_MALLOC(value_type, num_feat_entries);
	sg_memcpy(copy, features, num_feat_entries * sizeof(value_type));
	return SGSparseVector<T>(copy, num_feat_entries);
}

template <class T>
inline bool SGSparseVector<T>::operator==(const SGSparseVector<T>& other) const
{
	if (num_feat_entries != other.num_feat_entries)
		return false;

	if (features != other.features)
		return false;

	return true;
}

template <class T>
bool SGSparseVector<T>::equals(const SGSparseVector<T>& other) const
{
	/* same instance */
	if (*this == other)
		return true;

	// both empty
	if (!(num_feat_entries || other.num_feat_entries))
		return true;

	// only one empty
	if (!num_feat_entries || !other.num_feat_entries)
		return false;

	// different size
	if (num_feat_entries != other.num_feat_entries)
		return false;

	// content
	return std::equal(features, features + num_feat_entries, other.features);
}

template<class T> void SGSparseVector<T>::load(File * loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	loader->get_sparse_vector(features, num_feat_entries);
	SG_RESET_LOCALE;
}

template<class T> void SGSparseVector<T>::save(File * saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_sparse_vector(features, num_feat_entries);
	SG_RESET_LOCALE;
}

template <>
void SGSparseVector<complex128_t>::load(File * loader)
{
	error("SGSparseVector::load():: Not supported for complex128_t");
}

template <>
void SGSparseVector<complex128_t>::save(File * saver)
{
	error("SGSparseVector::save():: Not supported for complex128_t");
}

template <class T>
void SGSparseVector<T>::copy_data(const SGReferencedData &orig)
{
	num_feat_entries = ((SGSparseVector *)(&orig))->num_feat_entries;
	features = ((SGSparseVector *)(&orig))->features;
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
T SGSparseVector<T>::dot_prod_expensive_unsorted(const SGSparseVector<T> &a, const SGSparseVector<T> &b)
{
	io::warn("Computing sparse_dot(a,b) on unsorted vectors is very expensive: O(n^2)");
	io::warn("Using fallback to give correct results because upstream code does not sort.");

	T dot_prod = 0;

	for (size_type b_idx = 0; b_idx < b.num_feat_entries; ++b_idx)
	{
		const T tmp = b.features[b_idx].entry;

		for (size_type a_idx = 0; a_idx < a.num_feat_entries; ++a_idx)
		{
			if (a.features[a_idx].feat_index == b.features[b_idx].feat_index)
			{
				dot_prod += tmp * a.features[a_idx].entry;
			}
		}
	}

	return dot_prod;
}

template <class T>
std::string SGSparseVector<T>::to_string() const
{
	std::stringstream ss;
	ss << "[";
	if (num_feat_entries > 0)
	{
		pointer begin = features;
		pointer end = features+num_feat_entries;
		ss << std::boolalpha << begin->feat_index << ":" << begin->entry;
		std::for_each(++begin, end, [&ss](const_reference _v)
				{ ss << " " << _v.feat_index << ":" << _v.entry; });
	}
	ss << "]";
	return ss.str();
}

template <class T>
void SGSparseVector<T>::display_vector(const char * name, const char * prefix)
{
	io::print("{}{}={}\n", prefix, name, to_string().c_str());
}

template <class T>
bool SGSparseVectorEntry<T>::
operator==(const SGSparseVectorEntry<T>& other) const
{
	if (feat_index != other.feat_index)
		return false;

	return entry == other.entry;
}

#ifndef REAL_SPARSE_EQUALS
#define REAL_SPARSE_EQUALS(real_t)                                             \
	template <>                                                                \
	bool SGSparseVectorEntry<real_t>::operator==(                              \
	    const SGSparseVectorEntry<real_t>& other) const                        \
	{                                                                          \
		if (feat_index != other.feat_index)                                    \
			return false;                                                      \
                                                                               \
		return Math::fequals<real_t>(                                         \
		    entry, other.entry, std::numeric_limits<real_t>::epsilon());       \
	}

REAL_SPARSE_EQUALS(float32_t)
REAL_SPARSE_EQUALS(float64_t)
REAL_SPARSE_EQUALS(floatmax_t)
#undef REAL_SPARSE_EQUALS
#endif // REAL_SPARSE_EQUALS

template <>
bool SGSparseVectorEntry<complex128_t>::
operator==(const SGSparseVectorEntry<complex128_t>& other) const
{
	if (feat_index != other.feat_index)
		return false;

	return Math::fequals<float64_t>(
		       entry.real(), other.entry.real(), LDBL_EPSILON) &&
		   Math::fequals<float64_t>(
		       entry.imag(), other.entry.imag(), LDBL_EPSILON);
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
