/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Saurabh Mahindre, Soumyajit De, Heiko Strathmann,
 *          Sergey Lisitsyn, Sanuj Sharma, Chiyuan Zhang, Viktor Gal,
 *          Michele Mazzoni, Vladislav Horbatiuk, Kevin Hughes, Weijie Lin,
 *          Fernando Iglesias, Bjoern Esser, Evgeniy Andreev,
 *          Christopher Goldsworthy
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <algorithm>
#include <string.h>

#define ASSERT_FLOATING_POINT                                                  \
	switch (get_feature_type())                                                \
	{                                                                          \
	case F_SHORTREAL:                                                          \
	case F_DREAL:                                                              \
	case F_LONGREAL:                                                           \
		break;                                                                 \
	default:                                                                   \
		error(                                                                 \
		    "Only defined for {} with real type, not for {}.",	           \
		    get_name(), demangled_type<ST>().c_str());                         \
	}

namespace shogun {

template<class ST> DenseFeatures<ST>::DenseFeatures(int32_t size) : DotFeatures(size)
{
	init();
}

template<class ST> DenseFeatures<ST>::DenseFeatures(const DenseFeatures & orig) :
		DotFeatures(orig)
{
	init();
	set_feature_matrix(orig.feature_matrix);
	initialize_cache();

	if (orig.m_subset_stack != NULL)
	{

		m_subset_stack=std::make_shared<SubsetStack>(*orig.m_subset_stack);

	}
}

template<class ST> DenseFeatures<ST>::DenseFeatures(SGMatrix<ST> matrix) :
		DotFeatures()
{
	init();
	set_feature_matrix(matrix);
}

template<class ST> DenseFeatures<ST>::DenseFeatures(ST* src, int32_t num_feat, int32_t num_vec) :
		DotFeatures()
{
	init();
	set_feature_matrix(SGMatrix<ST>(src, num_feat, num_vec));
}
template<class ST> DenseFeatures<ST>::DenseFeatures(std::shared_ptr<File> loader) :
		DotFeatures()
{
	init();
	load(loader);
}

template<class ST> DenseFeatures<ST>::DenseFeatures(std::shared_ptr<DotFeatures> features) :
		DotFeatures()
{
	init();

	auto num_feat = features->get_dim_feature_space();
	auto num_vec = features->get_num_vectors();

	ASSERT(num_feat>0 && num_vec>0)
	feature_matrix = SGMatrix<ST>(num_feat, num_vec);
	for (auto i = 0; i < num_vec; i++)
	{
		SGVector<float64_t> v = features->get_computed_dot_feature_vector(i);
		ASSERT(num_feat==v.vlen)

		for (auto j = 0; j < num_feat; j++)
			feature_matrix.matrix[i * int64_t(num_feat) + j] = (ST) v.vector[j];
	}
	num_features = num_feat;
	num_vectors = num_vec;
}

template<class ST> std::shared_ptr<Features> DenseFeatures<ST>::duplicate() const
{
	return std::make_shared<DenseFeatures>(*this);
}

template<class ST> DenseFeatures<ST>::~DenseFeatures()
{
	free_features();
}

template<class ST> void DenseFeatures<ST>::free_features()
{
	free_feature_matrix();

}

template<class ST> void DenseFeatures<ST>::free_feature_matrix()
{
	m_subset_stack->remove_all_subsets();
	feature_matrix=SGMatrix<ST>();
	num_vectors = 0;
	num_features = 0;
}

template<class ST> ST* DenseFeatures<ST>::get_feature_vector(int32_t num, int32_t& len, bool& dofree) const
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	len = num_features;

	ST* feat = NULL;
	dofree = false;

	if (feature_matrix.matrix)
	{
		feat = &feature_matrix.matrix[real_num * int64_t(num_features)];
	}
	else
	{
		if (feature_cache)
		{
			feat = feature_cache->lock_entry(real_num);

			if (!feat)
				feat = feature_cache->set_entry(real_num);
		}

		if (!feat)
		{
			dofree = true;
			feat = compute_feature_vector(num, len, feat);
		}
	}

	if (get_num_preprocessors())
	{
		SGVector<ST> feat_vec(feat, len, false);

		for (auto i = 0; i < get_num_preprocessors(); i++)
		{
			auto preprocessor =
				get_preprocessor(i)->template as<DensePreprocessor<ST>>();
			// temporary hack
			SGVector<ST> applied =
				preprocessor->apply_to_feature_vector(feat_vec);

			if (i == 0)
				free_feature_vector(feat_vec.vector, num, dofree);
			feat_vec = applied;
		}

		feat = SG_MALLOC(ST, feat_vec.vlen);
		sg_memcpy(feat, feat_vec.vector, feat_vec.vlen * sizeof(ST));
		dofree = true;
		len = feat_vec.vlen;
	}
	return feat;
}

template<class ST> SGVector<ST> DenseFeatures<ST>::get_feature_vector(int32_t num) const
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (num >= get_num_vectors())
	{
		error("Index out of bounds (number of vectors {}, you "
		      "requested {})", get_num_vectors(), real_num);
	}

	int32_t vlen;
	bool do_free;
	ST* vector= get_feature_vector(num, vlen, do_free);
	return SGVector<ST>(vector, vlen, do_free);
}

template<class ST> void DenseFeatures<ST>::free_feature_vector(ST* feat_vec, int32_t num, bool dofree) const
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	if (dofree)
		SG_FREE(feat_vec);
}

template<class ST> void DenseFeatures<ST>::free_feature_vector(SGVector<ST> vec, int32_t num) const
{
	free_feature_vector(vec.vector, num, false);
	vec=SGVector<ST>();
}

template<class ST> void DenseFeatures<ST>::vector_subset(int32_t* idx, int32_t idx_len)
{
	if (m_subset_stack->has_subsets())
		error("A subset is set, cannot call vector_subset");

	ASSERT(feature_matrix.matrix)
	ASSERT(idx_len<=num_vectors)

	int32_t num_vec = num_vectors;
	num_vectors = idx_len;

	int32_t old_ii = -1;

	for (int32_t i = 0; i < idx_len; i++)
	{
		int32_t ii = idx[i];
		ASSERT(old_ii<ii)

		if (ii < 0 || ii >= num_vec)
			error("Index out of range: should be 0<{}<{}", ii, num_vec);

		if (i == ii)
			continue;

		sg_memcpy(&feature_matrix.matrix[int64_t(num_features) * i],
				&feature_matrix.matrix[int64_t(num_features) * ii],
				num_features * sizeof(ST));
		old_ii = ii;
	}
}

template<class ST> void DenseFeatures<ST>::feature_subset(int32_t* idx, int32_t idx_len)
{
	if (m_subset_stack->has_subsets())
		error("A subset is set, cannot call feature_subset");

	ASSERT(feature_matrix.matrix)
	ASSERT(idx_len<=num_features)
	int32_t num_feat = num_features;
	num_features = idx_len;

	for (int32_t i = 0; i < num_vectors; i++)
	{
		ST* src = &feature_matrix.matrix[int64_t(num_feat) * i];
		ST* dst = &feature_matrix.matrix[int64_t(num_features) * i];

		int32_t old_jj = -1;
		for (int32_t j = 0; j < idx_len; j++)
		{
			int32_t jj = idx[j];
			ASSERT(old_jj<jj)
			if (jj < 0 || jj >= num_feat)
				error("Index out of range: should be 0<{}<{}", jj, num_feat);

			dst[j] = src[jj];
			old_jj = jj;
		}
	}
}

template <class ST>
SGMatrix<ST> DenseFeatures<ST>::get_feature_matrix() const
{
	if (!m_subset_stack->has_subsets())
		return feature_matrix;

	SGMatrix<ST> target(num_features, get_num_vectors());
	copy_feature_matrix(target);
	return target;
}

template <class ST>
void DenseFeatures<ST>::copy_feature_matrix(SGMatrix<ST> target, index_t column_offset) const
{
	require(column_offset>=0, "Column offset ({}) cannot be negative!", column_offset);
	require(!target.equals(feature_matrix), "Source and target feature matrices cannot be the same");

	index_t num_vecs=get_num_vectors();
	index_t num_cols=num_vecs+column_offset;

	require(target.matrix!=nullptr, "Provided matrix is not allocated!");
	require(target.num_rows==num_features,
			"Number of rows of given matrix ({}) should be equal to the number of features ({})!",
			target.num_rows, num_features);
	require(target.num_cols>=num_cols,
			"Number of cols of given matrix ({}) should be at least {}!",
			target.num_cols, num_cols);

	if (!m_subset_stack->has_subsets())
	{
		auto src=feature_matrix.matrix;
		auto dest=target.matrix+int64_t(num_features)*column_offset;
		sg_memcpy(dest, src, feature_matrix.size()*sizeof(ST));
	}
	else
	{
		for (int32_t i=0; i<num_vecs; ++i)
		{
			auto real_i=m_subset_stack->subset_idx_conversion(i);
			auto src=feature_matrix.matrix+real_i*int64_t(num_features);
			auto dest=target.matrix+int64_t(num_features)*(column_offset+i);
			sg_memcpy(dest, src, num_features*sizeof(ST));
		}
	}
}

template<class ST> void DenseFeatures<ST>::set_feature_matrix(SGMatrix<ST> matrix)
{
	m_subset_stack->remove_all_subsets();
	free_feature_matrix();
	feature_matrix = matrix;
	num_features = matrix.num_rows;
	num_vectors = matrix.num_cols;
}

template <class ST>
ST* DenseFeatures<ST>::get_feature_matrix(
	int32_t& num_feat, int32_t& num_vec) const
{
	num_feat = num_features;
	num_vec = num_vectors;
	return feature_matrix.matrix;
}

template<class ST> std::shared_ptr<DenseFeatures<ST>> DenseFeatures<ST>::get_transposed()
{
	int32_t num_feat;
	int32_t num_vec;
	auto fm = get_transposed(num_feat, num_vec);

	return std::make_shared<DenseFeatures>(fm, num_feat, num_vec);
}

template<class ST> ST* DenseFeatures<ST>::get_transposed(int32_t &num_feat, int32_t &num_vec)
{
	num_feat = get_num_vectors();
	num_vec = num_features;

	int32_t old_num_vec=get_num_vectors();

	ST* fm = SG_MALLOC(ST, int64_t(num_feat) * num_vec);

	for (int32_t i=0; i<old_num_vec; i++)
	{
		SGVector<ST> vec=get_feature_vector(i);

		for (int32_t j=0; j<vec.vlen; j++)
			fm[j*int64_t(old_num_vec)+i]=vec.vector[j];

		free_feature_vector(vec, i);
	}

	return fm;
}

template<class ST> int32_t DenseFeatures<ST>::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : num_vectors;
}

template<class ST> int32_t DenseFeatures<ST>::get_num_features() const { return num_features; }

template<class ST> void DenseFeatures<ST>::set_num_features(int32_t num)
{
	num_features = num;
	initialize_cache();
}

template<class ST> void DenseFeatures<ST>::set_num_vectors(int32_t num)
{
	if (m_subset_stack->has_subsets())
		error("A subset is set, cannot call set_num_vectors");

	num_vectors = num;
	initialize_cache();
}

template<class ST> void DenseFeatures<ST>::initialize_cache()
{
	if (m_subset_stack->has_subsets())
		error("A subset is set, cannot call initialize_cache");

	if (num_features && num_vectors)
	{

		feature_cache = std::make_shared<Cache<ST>>(get_cache_size(), num_features,
				num_vectors);

	}
}

template<class ST> EFeatureClass DenseFeatures<ST>::get_feature_class() const  { return C_DENSE; }

template<class ST> int32_t DenseFeatures<ST>::get_dim_feature_space() const { return num_features; }

template<class ST> float64_t DenseFeatures<ST>::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df,
		int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	auto sf = std::static_pointer_cast<DenseFeatures<ST>>(df);

	int32_t len1, len2;
	bool free1, free2;

	ST* vec1 = get_feature_vector(vec_idx1, len1, free1);
	ST* vec2 = sf->get_feature_vector(vec_idx2, len2, free2);
	SGVector<ST> sg_vec1(vec1, len1, false);
	SGVector<ST> sg_vec2(vec2, len2, false);

	float64_t result = linalg::dot(sg_vec1, sg_vec2);

	free_feature_vector(vec1, vec_idx1, free1);
	sf->free_feature_vector(vec2, vec_idx2, free2);

	return result;
}

template<class ST> void DenseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
		float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	ST* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)

	if (abs_val)
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * Math::abs(vec1[i]);
	}
	else
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * vec1[i];
	}

	free_feature_vector(vec1, vec_idx1, vfree);
}

template<>
void DenseFeatures<float64_t>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
		float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	float64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)

	if (abs_val)
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * Math::abs(vec1[i]);
	}
	else
	{
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(vec2, alpha, vec1, num_features);
	}

	free_feature_vector(vec1, vec_idx1, vfree);
}

template<class ST> int32_t DenseFeatures<ST>::get_nnz_features_for_vector(int32_t num) const
{
	return num_features;
}

template<class ST> void* DenseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=get_num_vectors())
	{
		error("Index out of bounds (number of vectors {}, you "
		      "requested {})", get_num_vectors(), vector_index);
	}

	dense_feature_iterator* iterator = SG_MALLOC(dense_feature_iterator, 1);
	iterator->vec = get_feature_vector(vector_index, iterator->vlen,
			iterator->vfree);
	iterator->vidx = vector_index;
	iterator->index = 0;
	return iterator;
}

template<class ST> bool DenseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value,
		void* iterator)
{
	dense_feature_iterator* it = (dense_feature_iterator*) iterator;
	if (!it || it->index >= it->vlen)
		return false;

	index = it->index++;
	value = (float64_t) it->vec[index];

	return true;
}

template<class ST> void DenseFeatures<ST>::free_feature_iterator(void* iterator)
{
	if (!iterator)
		return;

	dense_feature_iterator* it = (dense_feature_iterator*) iterator;
	free_feature_vector(it->vec, it->vidx, it->vfree);
	SG_FREE(it);
}

template<class ST> std::shared_ptr<Features> DenseFeatures<ST>::copy_subset(SGVector<index_t> indices) const
{
	SGMatrix<ST> feature_matrix_copy(num_features, indices.vlen);

	for (index_t i=0; i<indices.vlen; ++i)
	{
		index_t real_idx=m_subset_stack->subset_idx_conversion(indices.vector[i]);
		sg_memcpy(&feature_matrix_copy.matrix[i*num_features],
				&feature_matrix.matrix[real_idx*num_features],
				num_features*sizeof(ST));
	}

	return std::make_shared<DenseFeatures>(feature_matrix_copy);
}

template<class ST>
std::shared_ptr<Features> DenseFeatures<ST>::copy_dimension_subset(SGVector<index_t> dims) const
{
	SG_TRACE("Entering!");

	// sanity checks
	index_t max=Math::max(dims.vector, dims.vlen);
	index_t min=Math::min(dims.vector, dims.vlen);
	require(max<num_features && min>=0,
			"Provided dimensions is in the range [{}, {}] but they "
			"have to be within [0, {}]! But it ", min, max, num_features);

	SGMatrix<ST> feature_matrix_copy(dims.vlen, get_num_vectors());

	for (index_t i=0; i<dims.vlen; ++i)
	{
		for (index_t j=0; j<get_num_vectors(); ++j)
		{
			index_t real_idx=m_subset_stack->subset_idx_conversion(j);
			feature_matrix_copy(i, j)=feature_matrix(dims[i], real_idx);
		}
	}

	SG_TRACE("Leaving!");
	return std::make_shared<DenseFeatures>(feature_matrix_copy);
}

template<class ST>
std::shared_ptr<Features> DenseFeatures<ST>::shallow_subset_copy()
{
	std::shared_ptr<Features> shallow_copy_features=NULL;

	SG_DEBUG("Using underlying feature matrix with {} dimensions and {} feature vectors!", num_features, num_vectors);
	SGMatrix<ST> shallow_copy_matrix(feature_matrix);
	shallow_copy_features=std::make_shared<DenseFeatures>(shallow_copy_matrix);

	if (m_subset_stack->has_subsets())
		shallow_copy_features->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_features;
}

template<class ST> ST* DenseFeatures<ST>::compute_feature_vector(int32_t num, int32_t& len,
		ST* target) const
{
	not_implemented(SOURCE_LOCATION);
	len = 0;
	return NULL;
}

template<class ST> void DenseFeatures<ST>::init()
{
	num_vectors = 0;
	num_features = 0;

	feature_matrix = SGMatrix<ST>();
	feature_cache = NULL;

	set_generic<ST>();

	/* not store number of vectors in subset */
	SG_ADD(&num_vectors, "num_vectors", "Number of vectors.");
	SG_ADD(&num_features, "num_features", "Number of features.");
	SG_ADD(&feature_matrix, "feature_matrix",
			"Matrix of feature vectors / 1 vector per column.");
}

#define GET_FEATURE_TYPE(f_type, sg_type)	\
template<> EFeatureType DenseFeatures<sg_type>::get_feature_type() const \
{																			\
	return f_type;															\
}

GET_FEATURE_TYPE(F_BOOL, bool)
GET_FEATURE_TYPE(F_CHAR, char)
GET_FEATURE_TYPE(F_BYTE, uint8_t)
GET_FEATURE_TYPE(F_BYTE, int8_t)
GET_FEATURE_TYPE(F_SHORT, int16_t)
GET_FEATURE_TYPE(F_WORD, uint16_t)
GET_FEATURE_TYPE(F_INT, int32_t)
GET_FEATURE_TYPE(F_UINT, uint32_t)
GET_FEATURE_TYPE(F_LONG, int64_t)
GET_FEATURE_TYPE(F_ULONG, uint64_t)
GET_FEATURE_TYPE(F_SHORTREAL, float32_t)
GET_FEATURE_TYPE(F_DREAL, float64_t)
GET_FEATURE_TYPE(F_LONGREAL, floatmax_t)
#undef GET_FEATURE_TYPE

template <typename ST>
float64_t
DenseFeatures<ST>::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	SGVector<ST> vec1 = get_feature_vector(vec_idx1);
	float64_t result = linalg::dot(vec2, vec1, linalg::allow_cast{});
	free_feature_vector(vec1, vec_idx1);
	return result;
}

template<class ST> bool DenseFeatures<ST>::is_equal(std::shared_ptr<DenseFeatures> rhs)
{
	if ( num_features != rhs->num_features || num_vectors != rhs->num_vectors )
		return false;

	ST* vec1;
	ST* vec2;
	int32_t v1len, v2len;
	bool v1free, v2free, stop = false;

	for (int32_t i = 0; i < num_vectors; i++)
	{
		vec1 = get_feature_vector(i, v1len, v1free);
		vec2 = rhs->get_feature_vector(i, v2len, v2free);

		if (v1len!=v2len)
			stop = true;

		for (int32_t j=0; j<v1len; j++)
		{
			if (vec1[j]!=vec2[j])
				stop = true;
		}

		free_feature_vector(vec1, i, v1free);
		free_feature_vector(vec2, i, v2free);

		if (stop)
			return false;
	}

	return true;
}

template <class ST>
std::shared_ptr<Features> DenseFeatures<ST>::create_merged_copy(const std::vector<std::shared_ptr<Features>>& others) const
{
	SG_TRACE("Entering.");

	require(others.size() > 0, "The list of other feature instances is not initialized!");
	auto total_num_vectors=get_num_vectors();

	for (auto current: others)
	{
		auto casted = current->as<DenseFeatures<ST>>();

		require(casted!=nullptr, "Provided object's type ({}) must match own type ({})!",
				current->get_name(), get_name());
		require(num_features==casted->num_features,
				"Provided feature object has different dimension ({}) than this one ({})!",
				casted->num_features, num_features);

		total_num_vectors+=casted->get_num_vectors();
	}

	SGMatrix<ST> data(num_features, total_num_vectors);
	index_t num_copied=0;
	copy_feature_matrix(data, num_copied);
	num_copied+=get_num_vectors();

	for (auto current: others)
	{
		auto casted = current->as<DenseFeatures<ST>>();
		casted->copy_feature_matrix(data, num_copied);
		num_copied+=casted->get_num_vectors();
	}

	SG_TRACE("Leaving.");
	return std::make_shared<DenseFeatures>(data);
}

template <class ST>
std::shared_ptr<Features> DenseFeatures<ST>::create_merged_copy(std::shared_ptr<Features> other) const
{
	std::vector<std::shared_ptr<Features>> v {other};
	return create_merged_copy(v);
}

template<class ST>
void DenseFeatures<ST>::load(std::shared_ptr<File> loader)
{
	SGMatrix<ST> matrix;
	matrix.load(loader);
	set_feature_matrix(matrix);
}

template<class ST>
void DenseFeatures<ST>::save(std::shared_ptr<File> writer)
{
	feature_matrix.save(writer);
}

template< class ST > std::shared_ptr<DenseFeatures< ST >> DenseFeatures< ST >::obtain_from_generic(std::shared_ptr<Features> base_features)
{
	require(base_features->get_feature_class() == C_DENSE,
			"base_features must be of dynamic type DenseFeatures");

	return std::static_pointer_cast<DenseFeatures< ST >>(base_features);
}

template <typename ST>
SGVector<ST> DenseFeatures<ST>::sum() const
{
	// TODO optimize non batch mode, but get_feature_vector is non const :(
	SGVector<ST> result = linalg::rowwise_sum(get_feature_matrix());
	return result;
}

template <typename ST>
SGVector<ST> DenseFeatures<ST>::mean() const
{
	ASSERT_FLOATING_POINT

	auto result = sum();
	ST scale = ((ST)1.0) / get_num_vectors();
	linalg::scale(result, result, scale);
	return result;
}

template <typename ST>
SGVector<float64_t > DenseFeatures<ST>::std(bool colwise) const
{
	ASSERT_FLOATING_POINT

	auto mat = get_feature_matrix();
	return linalg::std_deviation(mat, colwise);
}

template <typename ST>
SGMatrix<ST> DenseFeatures<ST>::cov() const
{
	// TODO optimize non batch mode, but get_feature_vector is non const :(
	auto mat = get_feature_matrix();
	return linalg::matrix_prod(mat, mat, false, true);
}

template <typename ST>
SGMatrix<ST> DenseFeatures<ST>::gram() const
{
	// TODO optimize non batch mode, but get_feature_vector is non const :(
	auto mat = get_feature_matrix();
	return linalg::matrix_prod(mat, mat, true, false);
}

template <typename ST>
SGVector<ST> DenseFeatures<ST>::dot(const SGVector<ST>& other) const
{
	require(
		get_num_vectors() == other.size(), "Number of feature vectors ({}) "
		                                   "must match provided vector's size "
		                                   "({}).",
		get_num_features(), other.size());
	// TODO optimize non batch mode, but get_feature_vector is non const :(
	return linalg::matrix_prod(get_feature_matrix(), other, false);
}

template class DenseFeatures<bool>;
template class DenseFeatures<char>;
template class DenseFeatures<int8_t>;
template class DenseFeatures<uint8_t>;
template class DenseFeatures<int16_t>;
template class DenseFeatures<uint16_t>;
template class DenseFeatures<int32_t>;
template class DenseFeatures<uint32_t>;
template class DenseFeatures<int64_t>;
template class DenseFeatures<uint64_t>;
template class DenseFeatures<float32_t>;
template class DenseFeatures<float64_t>;
template class DenseFeatures<floatmax_t>;
}
