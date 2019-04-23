/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/features/hashed/HashedSparseFeatures.h>
#include <shogun/features/hashed/HashedDenseFeatures.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Hash.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Math.h>
#include <string.h>
#include <iostream>

namespace shogun {

template <class ST>
HashedSparseFeatures<ST>::HashedSparseFeatures(int32_t size, bool use_quadr,
	bool keep_lin_terms) : DotFeatures(size)
{
	init(NULL, 0, use_quadr, keep_lin_terms);
}

template <class ST>
HashedSparseFeatures<ST>::HashedSparseFeatures(std::shared_ptr<SparseFeatures<ST>> feats, int32_t d,
	bool use_quadr, bool keep_lin_terms) : DotFeatures()
{
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
HashedSparseFeatures<ST>::HashedSparseFeatures(SGSparseMatrix<ST> matrix, int32_t d,
	bool use_quadr, bool keep_lin_terms) : DotFeatures()
{
	auto feats = std::make_shared<SparseFeatures<ST>>(matrix);
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
HashedSparseFeatures<ST>::HashedSparseFeatures(std::shared_ptr<File> loader, int32_t d, bool use_quadr,
	bool keep_lin_terms) : DotFeatures(loader)
{
	auto feats = std::make_shared<SparseFeatures<ST>>();
	feats->load(loader);
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
void HashedSparseFeatures<ST>::init(std::shared_ptr<SparseFeatures<ST>> feats, int32_t d, bool use_quadr,
	bool keep_lin_terms)
{
	dim = d;
	use_quadratic = use_quadr;
	keep_linear_terms = keep_lin_terms;
	sparse_feats = feats;


	SG_ADD(&use_quadratic, "use_quadratic", "Whether to use quadratic features");
	SG_ADD(&keep_linear_terms, "keep_linear_terms", "Whether to keep the linear terms or not");
	SG_ADD(&dim, "dim", "Dimension of new feature space");
	SG_ADD((std::shared_ptr<SGObject>* ) &sparse_feats, "sparse_feats", "Sparse features to work on");

	set_generic<ST>();
}

template <class ST>
HashedSparseFeatures<ST>::HashedSparseFeatures(const HashedSparseFeatures& orig)
: DotFeatures(orig)
{
	init(orig.sparse_feats, orig.dim, orig.use_quadratic, orig.keep_linear_terms);
}

template <class ST>
HashedSparseFeatures<ST>::~HashedSparseFeatures()
{

}

template <class ST>
std::shared_ptr<Features> HashedSparseFeatures<ST>::duplicate() const
{
	return std::make_shared<HashedSparseFeatures>(*this);
}

template <class ST>
int32_t HashedSparseFeatures<ST>::get_dim_feature_space() const
{
	return dim;
}

template <class ST>
SGSparseVector<ST> HashedSparseFeatures<ST>::get_hashed_feature_vector(
	int32_t vec_idx) const
{
	return HashedSparseFeatures<ST>::hash_vector(sparse_feats->get_sparse_feature_vector(vec_idx),
		dim, use_quadratic, keep_linear_terms);
}

template <class ST>
SGSparseVector<ST> HashedSparseFeatures<ST>::hash_vector(SGVector<ST> vec, int32_t dim,
	bool use_quadratic, bool keep_linear_terms)
{
	return HashedDenseFeatures<ST>::hash_vector(vec, dim, use_quadratic, keep_linear_terms);
}

template <class ST>
SGSparseVector<ST> HashedSparseFeatures<ST>::hash_vector(SGSparseVector<ST> vec, int32_t dim,
	bool use_quadratic, bool keep_linear_terms)
{
	SGVector<ST> h_vec(dim);
	SGVector<ST>::fill_vector(h_vec, dim, 0);

	int32_t hash_cache_size = use_quadratic ? vec.num_feat_entries : 0;
	SGVector<uint32_t> hash_cache(hash_cache_size);

	for (index_t i=0; i<vec.num_feat_entries; i++)
	{
		uint32_t hash = Hash::MurmurHash3((uint8_t* ) &vec.features[i].feat_index, sizeof (index_t),
						vec.features[i].feat_index);

		if (use_quadratic)
			hash_cache[i] = hash;

		if ( (!use_quadratic) || keep_linear_terms )
			h_vec[hash % dim] += vec.features[i].entry;
	}

	if (use_quadratic)
	{
		for (index_t i=0; i<vec.num_feat_entries; i++)
		{
			index_t n_idx = vec.features[i].feat_index + vec.features[i].feat_index;
			index_t idx = Hash::MurmurHash3((uint8_t* ) &n_idx, sizeof(index_t),
					vec.features[i].feat_index) % dim;

			h_vec[idx] += vec.features[i].entry * vec.features[i].entry;

			for (index_t j=i+1; j<vec.num_feat_entries; j++)
			{
				idx = (hash_cache[i] ^ hash_cache[j]) % dim;
				h_vec[idx] += vec.features[i].entry * vec.features[j].entry;
			}
		}
	}

	int32_t num_nnz_features = 0;
	for (index_t i=0; i<dim; i++)
	{
		if (h_vec[i]!=0)
			num_nnz_features++;
	}

	SGSparseVector<ST> sv(num_nnz_features);

	int32_t sparse_index = 0;
	for (index_t i=0; i<dim; i++)
	{
		if (h_vec[i]!=0)
		{
			sv.features[sparse_index].entry = h_vec[i];
			sv.features[sparse_index++].feat_index = i;
		}
	}

	return sv;
}

template <class ST>
float64_t HashedSparseFeatures<ST>::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df,
	int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	ASSERT(strcmp(df->get_name(), get_name())==0)

	auto feats = std::dynamic_pointer_cast<HashedSparseFeatures<ST>>(df);
	SGSparseVector<ST> vec_1 = get_hashed_feature_vector(vec_idx1);
	SGSparseVector<ST> vec_2 = feats->get_hashed_feature_vector(vec_idx2);

	float64_t result = vec_1.sparse_dot(vec_2);
	return result;
}

template <class ST>
float64_t HashedSparseFeatures<ST>::dense_dot(int32_t vec_idx1, const float64_t* vec2,
	int32_t vec2_len) const
{
	ASSERT(vec2_len == dim)

	SGSparseVector<ST> vec = sparse_feats->get_sparse_feature_vector(vec_idx1);

	int32_t hash_cache_size = use_quadratic ? vec.num_feat_entries : 0;
	SGVector<uint32_t> hash_cache(hash_cache_size);

	float64_t result = 0;
	for (index_t i=0; i<vec.num_feat_entries; i++)
	{
		uint32_t hash = Hash::MurmurHash3((uint8_t* ) &vec.features[i].feat_index, sizeof (index_t),
					   vec.features[i].feat_index);

		if (use_quadratic)
			hash_cache[i] = hash;

		if ( (!use_quadratic) || keep_linear_terms)
			result += vec2[hash % dim] * vec.features[i].entry;
	}

	if (use_quadratic)
	{
		for (index_t i=0; i<vec.num_feat_entries; i++)
		{
			index_t n_idx = vec.features[i].feat_index + vec.features[i].feat_index;
			index_t idx = Hash::MurmurHash3((uint8_t* ) &n_idx, sizeof (index_t),
						vec.features[i].feat_index) % dim;

			result += vec2[idx] * vec.features[i].entry * vec.features[i].entry;

			for (index_t j=i+1; j<vec.num_feat_entries; j++)
			{
				idx = (hash_cache[i] ^ hash_cache[j]) % dim;
				result += vec2[idx] * vec.features[i].entry * vec.features[j].entry;
			}
		}
	}

	sparse_feats ->free_feature_vector(vec_idx1);
	return result;
}

template <class ST>
void HashedSparseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
	float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	float64_t val = abs_val ? Math::abs(alpha) : alpha;
	ASSERT(vec2_len == dim)

	SGSparseVector<ST> vec = sparse_feats->get_sparse_feature_vector(vec_idx1);

	int32_t hash_cache_size = use_quadratic ? vec.num_feat_entries : 0;
	SGVector<uint32_t> hash_cache(hash_cache_size);

	for (index_t i=0; i<vec.num_feat_entries; i++)
	{
		uint32_t hash = Hash::MurmurHash3((uint8_t* ) &vec.features[i].feat_index, sizeof (index_t),
					   vec.features[i].feat_index);
		if (use_quadratic)
			hash_cache[i] = hash;

		if ( (!use_quadratic) || keep_linear_terms)
			vec2[hash % dim] += val * vec.features[i].entry;
	}

	if (use_quadratic)
	{
		for (index_t i=0; i<vec.num_feat_entries; i++)
		{
			index_t n_idx = vec.features[i].feat_index + vec.features[i].feat_index;
			index_t idx = Hash::MurmurHash3((uint8_t* ) &n_idx, sizeof (index_t),
						vec.features[i].feat_index) % dim;

			vec2[idx] += val * vec.features[i].entry * vec.features[i].entry;

			for (index_t j=i+1; j<vec.num_feat_entries; j++)
			{
				idx = (hash_cache[i] ^ hash_cache[j]) % dim;
				vec2[idx] += val * vec.features[i].entry * vec.features[j].entry;
			}
		}
	}
	sparse_feats ->free_feature_vector(vec_idx1);
}

template <class ST>
int32_t HashedSparseFeatures<ST>::get_nnz_features_for_vector(int32_t num) const
{
	return dim;
}

template <class ST>
void* HashedSparseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}
template <class ST>
bool HashedSparseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value,
	void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}
template <class ST>
void HashedSparseFeatures<ST>::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED;
}

template <class ST>
const char* HashedSparseFeatures<ST>::get_name() const
{
	return "HashedSparseFeatures";
}

template <class ST>
EFeatureType HashedSparseFeatures<ST>::get_feature_type() const
{
	return F_UINT;
}

template <class ST>
EFeatureClass HashedSparseFeatures<ST>::get_feature_class() const
{
	return C_SPARSE;
}

template <class ST>
int32_t HashedSparseFeatures<ST>::get_num_vectors() const
{
	return sparse_feats ->get_num_vectors();
}

template class HashedSparseFeatures <bool>;
template class HashedSparseFeatures <char>;
template class HashedSparseFeatures <int8_t>;
template class HashedSparseFeatures <uint8_t>;
template class HashedSparseFeatures <int16_t>;
template class HashedSparseFeatures <uint16_t>;
template class HashedSparseFeatures <int32_t>;
template class HashedSparseFeatures <uint32_t>;
template class HashedSparseFeatures <int64_t>;
template class HashedSparseFeatures <uint64_t>;
template class HashedSparseFeatures <float32_t>;
template class HashedSparseFeatures <float64_t>;
template class HashedSparseFeatures <floatmax_t>;
}
