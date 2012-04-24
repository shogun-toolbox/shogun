#include <shogun/features/SimpleFeatures.h>
#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>

#include <string.h>

namespace shogun {

template<class ST> CSimpleFeatures<ST>::CSimpleFeatures(int32_t size) : CDotFeatures(size)
{
	init();
}

template<class ST> CSimpleFeatures<ST>::CSimpleFeatures(const CSimpleFeatures & orig) :
		CDotFeatures(orig)
{
	copy_feature_matrix(SGMatrix<ST>(orig.feature_matrix,
									 orig.num_features,
									 orig.num_vectors));
	initialize_cache();
	init();

	m_subset_stack=orig.m_subset_stack;
	SG_REF(m_subset_stack);
}

template<class ST> CSimpleFeatures<ST>::CSimpleFeatures(SGMatrix<ST> matrix) :
		CDotFeatures()
{
	init();
	set_feature_matrix(matrix);
}
template<class ST> CSimpleFeatures<ST>::CSimpleFeatures(ST* src, int32_t num_feat, int32_t num_vec) :
		CDotFeatures()
{
	init();
	set_feature_matrix(src, num_feat, num_vec);
}
template<class ST> CSimpleFeatures<ST>::CSimpleFeatures(CFile* loader) :
		CDotFeatures(loader)
{
	init();
	load(loader);
}
template<class ST> CFeatures* CSimpleFeatures<ST>::duplicate() const
{
	return new CSimpleFeatures<ST>(*this);
}

template<class ST> CSimpleFeatures<ST>::~CSimpleFeatures()
{
	free_features();
}

template<class ST> void CSimpleFeatures<ST>::free_features()
{
	m_subset_stack->remove_all_subsets();
	free_feature_matrix();
	SG_UNREF(feature_cache);
}

template<class ST> void CSimpleFeatures<ST>::free_feature_matrix()
{
	m_subset_stack->remove_all_subsets();
	SG_FREE(feature_matrix);
	feature_matrix = NULL;
	feature_matrix_num_features = num_features;
	feature_matrix_num_vectors = num_vectors;
	num_vectors = 0;
	num_features = 0;
}

template<class ST> ST* CSimpleFeatures<ST>::get_feature_vector(int32_t num, int32_t& len, bool& dofree)
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	len = num_features;

	if (feature_matrix)
	{
		dofree = false;
		return &feature_matrix[real_num * int64_t(num_features)];
	}

	ST* feat = NULL;
	dofree = false;

	if (feature_cache)
	{
		feat = feature_cache->lock_entry(num);

		if (feat)
			return feat;
		else
			feat = feature_cache->set_entry(real_num);
	}

	if (!feat)
		dofree = true;
	feat = compute_feature_vector(num, len, feat);

	if (get_num_preprocessors())
	{
		int32_t tmp_len = len;
		ST* tmp_feat_before = feat;
		ST* tmp_feat_after = NULL;

		for (int32_t i = 0; i < get_num_preprocessors(); i++)
		{
			CSimplePreprocessor<ST>* p =
					(CSimplePreprocessor<ST>*) get_preprocessor(i);
			// temporary hack
			SGVector<ST> applied = p->apply_to_feature_vector(
					SGVector<ST>(tmp_feat_before, tmp_len));
			tmp_feat_after = applied.vector;
			SG_UNREF(p);

			if (i != 0) // delete feature vector, except for the the first one, i.e., feat
				SG_FREE(tmp_feat_before);
			tmp_feat_before = tmp_feat_after;
		}

		memcpy(feat, tmp_feat_after, sizeof(ST) * tmp_len);
		SG_FREE(tmp_feat_after);

		len = tmp_len;
	}
	return feat;
}

template<class ST> void CSimpleFeatures<ST>::set_feature_vector(const SGVector<ST>& vector, int32_t num)
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (num>=get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
		"requested %d)\n", get_num_vectors(), num);
	}

	if (!feature_matrix)
		SG_ERROR("Requires a in-memory feature matrix\n");

	if (vector.vlen != num_features)
		SG_ERROR(
				"Vector not of length %d (has %d)\n", num_features, vector.vlen);

	memcpy(&feature_matrix[real_num * int64_t(num_features)], vector.vector,
			int64_t(num_features) * sizeof(ST));
}

template<class ST> SGVector<ST> CSimpleFeatures<ST>::get_feature_vector(int32_t num)
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (num >= get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
		"requested %d)\n", get_num_vectors(), real_num);
	}

	SGVector<ST> vec;
	vec.vector = get_feature_vector(num, vec.vlen, vec.do_free);
	return vec;
}

template<class ST> void CSimpleFeatures<ST>::free_feature_vector(ST* feat_vec, int32_t num, bool dofree)
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	if (dofree)
		SG_FREE(feat_vec);
}

template<class ST> void CSimpleFeatures<ST>::free_feature_vector(const SGVector<ST>& vec, int32_t num)
{
	free_feature_vector(vec.vector, num, vec.do_free);
}

template<class ST> void CSimpleFeatures<ST>::vector_subset(int32_t* idx, int32_t idx_len)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call vector_subset\n");

	ASSERT(feature_matrix);
	ASSERT(idx_len<=num_vectors);

	int32_t num_vec = num_vectors;
	num_vectors = idx_len;

	int32_t old_ii = -1;

	for (int32_t i = 0; i < idx_len; i++)
	{
		int32_t ii = idx[i];
		ASSERT(old_ii<ii);

		if (ii < 0 || ii >= num_vec)
			SG_ERROR( "Index out of range: should be 0<%d<%d\n", ii, num_vec);

		if (i == ii)
			continue;

		memcpy(&feature_matrix[int64_t(num_features) * i],
				&feature_matrix[int64_t(num_features) * ii],
				num_features * sizeof(ST));
		old_ii = ii;
	}
}

template<class ST> void CSimpleFeatures<ST>::feature_subset(int32_t* idx, int32_t idx_len)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call feature_subset\n");

	ASSERT(feature_matrix);
	ASSERT(idx_len<=num_features);
	int32_t num_feat = num_features;
	num_features = idx_len;

	for (int32_t i = 0; i < num_vectors; i++)
	{
		ST* src = &feature_matrix[int64_t(num_feat) * i];
		ST* dst = &feature_matrix[int64_t(num_features) * i];

		int32_t old_jj = -1;
		for (int32_t j = 0; j < idx_len; j++)
		{
			int32_t jj = idx[j];
			ASSERT(old_jj<jj);
			if (jj < 0 || jj >= num_feat)
				SG_ERROR(
						"Index out of range: should be 0<%d<%d\n", jj, num_feat);

			dst[j] = src[jj];
			old_jj = jj;
		}
	}
}

template<class ST> void CSimpleFeatures<ST>::get_feature_matrix(ST** dst, int32_t* num_feat, int32_t* num_vec)
{
	ASSERT(feature_matrix);

	int64_t num = int64_t(num_features) * get_num_vectors();
	*num_feat = num_features;
	*num_vec = get_num_vectors();
	*dst = SG_MALLOC(ST, num);

	/* copying depends on whether a subset is used */
	if (m_subset_stack->has_subsets())
	{
		/* copy vector wise */
		for (int32_t i = 0; i < *num_vec; ++i)
		{
			int32_t real_i = m_subset_stack->subset_idx_conversion(i);
			memcpy(*dst, &feature_matrix[real_i * int64_t(num_features)],
					num_features * sizeof(ST));
		}
	}
	else
	{
		/* copy complete matrix */
		memcpy(*dst, feature_matrix, num * sizeof(ST));
	}
}

template<class ST> SGMatrix<ST> CSimpleFeatures<ST>::get_feature_matrix()
{
	return SGMatrix<ST>(feature_matrix, num_features, num_vectors);
}

template<class ST> SGMatrix<ST> CSimpleFeatures<ST>::steal_feature_matrix()
{
	SGMatrix<ST> st_feature_matrix(feature_matrix, num_features, num_vectors);
	m_subset_stack->remove_all_subsets();
	SG_UNREF(feature_cache);
	clean_preprocessors();

	feature_matrix = NULL;
	feature_matrix_num_vectors = 0;
	feature_matrix_num_features = 0;
	num_features = 0;
	num_vectors = 0;
	return st_feature_matrix;
}

template<class ST> void CSimpleFeatures<ST>::set_feature_matrix(SGMatrix<ST> matrix)
{
	m_subset_stack->remove_all_subsets();
	free_feature_matrix();
	feature_matrix = matrix.matrix;
	num_features = matrix.num_rows;
	num_vectors = matrix.num_cols;
	feature_matrix_num_vectors = num_vectors;
	feature_matrix_num_features = num_features;
}

template<class ST> ST* CSimpleFeatures<ST>::get_feature_matrix(int32_t &num_feat, int32_t &num_vec)
{
	num_feat = num_features;
	num_vec = num_vectors;
	return feature_matrix;
}

template<class ST> CSimpleFeatures<ST>* CSimpleFeatures<ST>::get_transposed()
{
	int32_t num_feat;
	int32_t num_vec;
	ST* fm = get_transposed(num_feat, num_vec);

	return new CSimpleFeatures<ST>(fm, num_feat, num_vec);
}

template<class ST> ST* CSimpleFeatures<ST>::get_transposed(int32_t &num_feat, int32_t &num_vec)
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

template<class ST> void CSimpleFeatures<ST>::set_feature_matrix(ST* fm, int32_t num_feat, int32_t num_vec)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call set_feature_matrix\n");

	free_feature_matrix();
	feature_matrix = fm;
	feature_matrix_num_features = num_feat;
	feature_matrix_num_vectors = num_vec;

	num_features = num_feat;
	num_vectors = num_vec;
	initialize_cache();
}

template<class ST> void CSimpleFeatures<ST>::copy_feature_matrix(SGMatrix<ST> src)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call copy_feature_matrix\n");

	free_feature_matrix();
	int32_t num_feat = src.num_rows;
	int32_t num_vec = src.num_cols;
	feature_matrix = SG_MALLOC(ST, ((int64_t) num_feat) * num_vec);
	feature_matrix_num_features = num_feat;
	feature_matrix_num_vectors = num_vec;

	memcpy(feature_matrix, src.matrix,
			(sizeof(ST) * ((int64_t) num_feat) * num_vec));

	num_features = num_feat;
	num_vectors = num_vec;
	initialize_cache();
}

template<class ST> void CSimpleFeatures<ST>::obtain_from_dot(CDotFeatures* df)
{
	m_subset_stack->remove_all_subsets();

	int32_t num_feat = df->get_dim_feature_space();
	int32_t num_vec = df->get_num_vectors();

	ASSERT(num_feat>0 && num_vec>0);

	free_feature_matrix();
	feature_matrix = SG_MALLOC(ST, ((int64_t) num_feat) * num_vec);
	feature_matrix_num_features = num_feat;
	feature_matrix_num_vectors = num_vec;

	for (int32_t i = 0; i < num_vec; i++)
	{
		SGVector<float64_t> v = df->get_computed_dot_feature_vector(i);
		ASSERT(num_feat==v.vlen);

		for (int32_t j = 0; j < num_feat; j++)
			feature_matrix[i * int64_t(num_feat) + j] = (ST) v.vector[j];

		v.free_vector();
	}
	num_features = num_feat;
	num_vectors = num_vec;
}

template<class ST> bool CSimpleFeatures<ST>::apply_preprocessor(bool force_preprocessing)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call apply_preproc\n");

	SG_DEBUG( "force: %d\n", force_preprocessing);

	if (feature_matrix && get_num_preprocessors())
	{
		for (int32_t i = 0; i < get_num_preprocessors(); i++)
		{
			if ((!is_preprocessed(i) || force_preprocessing))
			{
				set_preprocessed(i);
				CSimplePreprocessor<ST>* p =
						(CSimplePreprocessor<ST>*) get_preprocessor(i);
				SG_INFO( "preprocessing using preproc %s\n", p->get_name());

				if (p->apply_to_feature_matrix(this).matrix == NULL)
				{
					SG_UNREF(p);
					return false;
				}SG_UNREF(p);

			}
		}

		return true;
	}
	else
	{
		if (!feature_matrix)
			SG_ERROR( "no feature matrix\n");

		if (!get_num_preprocessors())
			SG_ERROR( "no preprocessors available\n");

		return false;
	}
}

template<class ST> int32_t CSimpleFeatures<ST>::get_size() { return sizeof(ST); }

template<class ST> int32_t CSimpleFeatures<ST>::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : num_vectors;
}

template<class ST> int32_t CSimpleFeatures<ST>::get_num_features() { return num_features; }

template<class ST> void CSimpleFeatures<ST>::set_num_features(int32_t num)
{
	num_features = num;
	initialize_cache();
}

template<class ST> void CSimpleFeatures<ST>::set_num_vectors(int32_t num)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call set_num_vectors\n");

	num_vectors = num;
	initialize_cache();
}

template<class ST> void CSimpleFeatures<ST>::initialize_cache()
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call initialize_cache\n");

	if (num_features && num_vectors)
	{
		SG_UNREF(feature_cache);
		feature_cache = new CCache<ST>(get_cache_size(), num_features,
				num_vectors);
		SG_REF(feature_cache);
	}
}

template<class ST> EFeatureClass CSimpleFeatures<ST>::get_feature_class() { return C_SIMPLE; }

template<class ST> bool CSimpleFeatures<ST>::reshape(int32_t p_num_features, int32_t p_num_vectors)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call reshape\n");

	if (p_num_features * p_num_vectors
			== this->num_features * this->num_vectors)
	{
		num_features = p_num_features;
		num_vectors = p_num_vectors;
		return true;
	} else
		return false;
}

template<class ST> int32_t CSimpleFeatures<ST>::get_dim_feature_space() const { return num_features; }

template<class ST> float64_t CSimpleFeatures<ST>::dot(int32_t vec_idx1, CDotFeatures* df,
		int32_t vec_idx2)
{
	ASSERT(df);
	ASSERT(df->get_feature_type() == get_feature_type());
	ASSERT(df->get_feature_class() == get_feature_class());
	CSimpleFeatures<ST>* sf = (CSimpleFeatures<ST>*) df;

	int32_t len1, len2;
	bool free1, free2;

	ST* vec1 = get_feature_vector(vec_idx1, len1, free1);
	ST* vec2 = sf->get_feature_vector(vec_idx2, len2, free2);

	float64_t result = CMath::dot(vec1, vec2, len1);

	free_feature_vector(vec1, vec_idx1, free1);
	sf->free_feature_vector(vec2, vec_idx2, free2);

	return result;
}

template<class ST> void CSimpleFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
		float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	ST* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);

	if (abs_val)
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * CMath::abs(vec1[i]);
	}
	else
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * vec1[i];
	}

	free_feature_vector(vec1, vec_idx1, vfree);
}

template<class ST> int32_t CSimpleFeatures<ST>::get_nnz_features_for_vector(int32_t num)
{
	return num_features;
}

template<class ST> bool CSimpleFeatures<ST>::Align_char_features(CStringFeatures<char>* cf,
		CStringFeatures<char>* Ref, float64_t gapCost)
{
	return false;
}

template<class ST> void* CSimpleFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
		"requested %d)\n", get_num_vectors(), vector_index);
	}

	simple_feature_iterator* iterator = SG_MALLOC(simple_feature_iterator, 1);
	iterator->vec = get_feature_vector(vector_index, iterator->vlen,
			iterator->vfree);
	iterator->vidx = vector_index;
	iterator->index = 0;
	return iterator;
}

template<class ST> bool CSimpleFeatures<ST>::get_next_feature(int32_t& index, float64_t& value,
		void* iterator)
{
	simple_feature_iterator* it = (simple_feature_iterator*) iterator;
	if (!it || it->index >= it->vlen)
		return false;

	index = it->index++;
	value = (float64_t) it->vec[index];

	return true;
}

template<class ST> void CSimpleFeatures<ST>::free_feature_iterator(void* iterator)
{
	if (!iterator)
		return;

	simple_feature_iterator* it = (simple_feature_iterator*) iterator;
	free_feature_vector(it->vec, it->vidx, it->vfree);
	SG_FREE(it);
}

template<class ST> CFeatures* CSimpleFeatures<ST>::copy_subset(const SGVector<index_t>& indices)
{
	SGMatrix<ST> feature_matrix_copy(num_features, indices.vlen);

	for (index_t i=0; i<indices.vlen; ++i)
	{
		index_t real_idx=m_subset_stack->subset_idx_conversion(indices.vector[i]);
		memcpy(&feature_matrix_copy.matrix[i*num_features],
				&feature_matrix[real_idx*num_features],
				num_features*sizeof(ST));
	}

	return new CSimpleFeatures(feature_matrix_copy);
}

template<class ST> ST* CSimpleFeatures<ST>::compute_feature_vector(int32_t num, int32_t& len,
		ST* target)
{
	SG_NOTIMPLEMENTED;
	len = 0;
	return NULL;
}

template<class ST> void CSimpleFeatures<ST>::init()
{
	num_vectors = 0;
	num_features = 0;

	feature_matrix = NULL;
	feature_matrix_num_vectors = 0;
	feature_matrix_num_features = 0;

	feature_cache = NULL;

	set_generic<ST>();
	/* not store number of vectors in subset */
	m_parameters->add(&num_vectors, "num_vectors",
			"Number of vectors.");
	m_parameters->add(&num_features, "num_features", "Number of features.");
	m_parameters->add_matrix(&feature_matrix, &feature_matrix_num_features,
			&feature_matrix_num_vectors, "feature_matrix",
			"Matrix of feature vectors / 1 vector per column.");
}

#define GET_FEATURE_TYPE(f_type, sg_type)	\
template<> EFeatureType CSimpleFeatures<sg_type>::get_feature_type() \
{ 																			\
	return f_type; 															\
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

/** align strings and compute emperical kernel map based on alignment scores
 *
 * non functional code - needs updating
 *
 * @param cf strings to be aligned to reference
 * @param Ref reference strings to be aligned to
 * @param gapCost costs for a gap
 */
template<> bool CSimpleFeatures<float64_t>::Align_char_features(
		CStringFeatures<char>* cf, CStringFeatures<char>* Ref,
		float64_t gapCost)
{
	ASSERT(cf);
	/*num_vectors=cf->get_num_vectors();
	 num_features=Ref->get_num_vectors();

	 int64_t len=((int64_t) num_vectors)*num_features;
	 free_feature_matrix();
	 feature_matrix=SG_MALLOC(float64_t, len);
	 int32_t num_cf_feat=0;
	 int32_t num_cf_vec=0;
	 int32_t num_ref_feat=0;
	 int32_t num_ref_vec=0;
	 char* fm_cf=NULL; //cf->get_feature_matrix(num_cf_feat, num_cf_vec);
	 char* fm_ref=NULL; //Ref->get_feature_matrix(num_ref_feat, num_ref_vec);

	 ASSERT(num_cf_vec==num_vectors);
	 ASSERT(num_ref_vec==num_features);

	 SG_INFO( "computing aligments of %i vectors to %i reference vectors: ", num_cf_vec, num_ref_vec) ;
	 for (int32_t i=0; i< num_ref_vec; i++)
	 {
	 SG_PROGRESS(i, num_ref_vec) ;
	 for (int32_t j=0; j<num_cf_vec; j++)
	 feature_matrix[i+j*num_features] = CMath::Align(&fm_cf[j*num_cf_feat], &fm_ref[i*num_ref_feat], num_cf_feat, num_ref_feat, gapCost);
	 } ;

	 SG_INFO( "created %i x %i matrix (0x%p)\n", num_features, num_vectors, feature_matrix) ;*/
	return true;
}

template<> float64_t CSimpleFeatures<bool>::dense_dot(int32_t vec_idx1,
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	bool* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] ? vec2[i] : 0;

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<char>::dense_dot(int32_t vec_idx1,
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	char* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<int8_t>::dense_dot(int32_t vec_idx1,
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	int8_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<uint8_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint8_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<int16_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	int16_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<uint16_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint16_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<int32_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	int32_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<uint32_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint32_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<int64_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	int64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<uint64_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	uint64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<float32_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	float32_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<float64_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	float64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = CMath::dot(vec1, vec2, num_features);

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CSimpleFeatures<floatmax_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features);

	int32_t vlen;
	bool vfree;
	floatmax_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features);
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<class ST> bool CSimpleFeatures<ST>::is_equal(CSimpleFeatures* rhs)
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

		if ( v1len != v2len )
			stop = true;

		for (int32_t j = 0; j < v1len; j++)
		{
			if ( vec1[j] != vec2[j] )
				stop = true;
		}

		free_feature_vector(vec1, i, v1free);
		free_feature_vector(vec2, i, v2free);

		if ( stop )
			return false;
	}

	return true;
}

#define LOAD(f_load, sg_type)												\
template<> void CSimpleFeatures<sg_type>::load(CFile* loader)		\
{ 																			\
	SG_SET_LOCALE_C;													\
	ASSERT(loader);															\
	sg_type* matrix;														\
	int32_t num_feat;														\
	int32_t num_vec;														\
	loader->f_load(matrix, num_feat, num_vec);								\
	set_feature_matrix(matrix, num_feat, num_vec);							\
	SG_RESET_LOCALE;													\
}

LOAD(get_matrix, bool)
LOAD(get_matrix, char)
LOAD(get_int8_matrix, int8_t)
LOAD(get_matrix, uint8_t)
LOAD(get_matrix, int16_t)
LOAD(get_matrix, uint16_t)
LOAD(get_matrix, int32_t)
LOAD(get_uint_matrix, uint32_t)
LOAD(get_long_matrix, int64_t)
LOAD(get_ulong_matrix, uint64_t)
LOAD(get_matrix, float32_t)
LOAD(get_matrix, float64_t)
LOAD(get_longreal_matrix, floatmax_t)
#undef LOAD

#define SAVE(f_write, sg_type)												\
template<> void CSimpleFeatures<sg_type>::save(CFile* writer)		\
{ 																			\
	SG_SET_LOCALE_C;													\
	ASSERT(writer);															\
	writer->f_write(feature_matrix, num_features, num_vectors);				\
	SG_RESET_LOCALE;													\
}

SAVE(set_matrix, bool)
SAVE(set_matrix, char)
SAVE(set_int8_matrix, int8_t)
SAVE(set_matrix, uint8_t)
SAVE(set_matrix, int16_t)
SAVE(set_matrix, uint16_t)
SAVE(set_matrix, int32_t)
SAVE(set_uint_matrix, uint32_t)
SAVE(set_long_matrix, int64_t)
SAVE(set_ulong_matrix, uint64_t)
SAVE(set_matrix, float32_t)
SAVE(set_matrix, float64_t)
SAVE(set_longreal_matrix, floatmax_t)
#undef SAVE

template class CSimpleFeatures<bool>;
template class CSimpleFeatures<char>;
template class CSimpleFeatures<int8_t>;
template class CSimpleFeatures<uint8_t>;
template class CSimpleFeatures<int16_t>;
template class CSimpleFeatures<uint16_t>;
template class CSimpleFeatures<int32_t>;
template class CSimpleFeatures<uint32_t>;
template class CSimpleFeatures<int64_t>;
template class CSimpleFeatures<uint64_t>;
template class CSimpleFeatures<float32_t>;
template class CSimpleFeatures<float64_t>;
template class CSimpleFeatures<floatmax_t>;
}
