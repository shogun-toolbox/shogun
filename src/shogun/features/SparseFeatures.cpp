#include <shogun/lib/memory.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/preprocessor/SparsePreprocessor.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/DataType.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/io/SGIO.h>

#include <string.h>
#include <stdlib.h>

namespace shogun
{

template<class ST> CSparseFeatures<ST>::CSparseFeatures(int32_t size)
: CDotFeatures(size), feature_cache(NULL)
{
	init();
}

template<class ST> CSparseFeatures<ST>::CSparseFeatures(SGSparseMatrix<ST> sparse)
: CDotFeatures(0), feature_cache(NULL)
{
	init();

	set_sparse_feature_matrix(sparse);
}

template<class ST> CSparseFeatures<ST>::CSparseFeatures(SGMatrix<ST> dense)
: CDotFeatures(0), feature_cache(NULL)
{
	init();

	set_full_feature_matrix(dense);
}

template<class ST> CSparseFeatures<ST>::CSparseFeatures(const CSparseFeatures & orig)
: CDotFeatures(orig), sparse_feature_matrix(orig.sparse_feature_matrix),
	feature_cache(orig.feature_cache)
{
	init();

	m_subset_stack=orig.m_subset_stack;
	SG_REF(m_subset_stack);
}
template<class ST> CSparseFeatures<ST>::CSparseFeatures(CFile* loader)
: CDotFeatures(loader), feature_cache(NULL)
{
	init();

	load(loader);
}

template<class ST> CSparseFeatures<ST>::~CSparseFeatures()
{
	SG_UNREF(feature_cache);
}

template<class ST> CFeatures* CSparseFeatures<ST>::duplicate() const
{
	return new CSparseFeatures<ST>(*this);
}

template<class ST> ST CSparseFeatures<ST>::get_feature(int32_t num, int32_t index)
{
	ASSERT(num>=0 && num<get_num_vectors())
	SGSparseVector<ST> sv=get_sparse_feature_vector(num);

	ASSERT(index>=0 && index<get_num_features());
	ST ret = sv.get_feature(index);

	free_sparse_feature_vector(num);

	return ret ;
}

template<class ST> SGVector<ST> CSparseFeatures<ST>::get_full_feature_vector(int32_t num)
{
	if (num>=get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
				"requested %d)\n", get_num_vectors(), num);
	}

	SGSparseVector<ST> sv=get_sparse_feature_vector(num);

	SGVector<ST> dense;

	if (sv.features)
	{
		dense=SGVector<ST>(get_num_features());
		dense.zero();

		for (int32_t i=0; i<sv.num_feat_entries; i++)
			dense.vector[sv.features[i].feat_index] = sv.features[i].entry;
	}

	free_sparse_feature_vector(num);

	return dense;
}

template<class ST> int32_t CSparseFeatures<ST>::get_nnz_features_for_vector(int32_t num)
{
	SGSparseVector<ST> sv = get_sparse_feature_vector(num);
	int32_t len=sv.num_feat_entries;
	free_sparse_feature_vector(num);
	return len;
}

template<class ST> SGSparseVector<ST> CSparseFeatures<ST>::get_sparse_feature_vector(int32_t num)
{
	ASSERT(num<get_num_vectors())

	index_t real_num=m_subset_stack->subset_idx_conversion(num);

	SGSparseVector<ST> result;

	if (sparse_feature_matrix.sparse_matrix)
	{
		return sparse_feature_matrix[real_num];
	}
	else
	{
		if (feature_cache)
		{
			result.features=feature_cache->lock_entry(num);

			if (result.features)
				return result;
			else
			{
				result.features=feature_cache->set_entry(num);
			}
		}

		//if (!result.features)
		//	result.do_free=true;

		result.features=compute_sparse_feature_vector(num,
			result.num_feat_entries, result.features);


		if (get_num_preprocessors())
		{
			int32_t tmp_len=result.num_feat_entries;
			SGSparseVectorEntry<ST>* tmp_feat_before=result.features;
			SGSparseVectorEntry<ST>* tmp_feat_after = NULL;

			for (int32_t i=0; i<get_num_preprocessors(); i++)
			{
				//tmp_feat_after=((CSparsePreprocessor<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

				if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
					SG_FREE(tmp_feat_before);
				tmp_feat_before=tmp_feat_after;
			}

			memcpy(result.features, tmp_feat_after,
					sizeof(SGSparseVectorEntry<ST>)*tmp_len);

			SG_FREE(tmp_feat_after);
			result.num_feat_entries=tmp_len ;
			SG_DEBUG("len: %d len2: %d\n", result.num_feat_entries, get_num_features())
		}
		return result ;
	}
}

template<class ST> ST CSparseFeatures<ST>::dense_dot(ST alpha, int32_t num, ST* vec, int32_t dim, ST b)
{
	SGSparseVector<ST> sv=get_sparse_feature_vector(num);

	ST result = sv.dense_dot(alpha,vec,dim,b);

	free_sparse_feature_vector(num);
	return result;
}

template<class ST> void CSparseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t num, float64_t* vec, int32_t dim, bool abs_val)
{
	ASSERT(vec)
	if (dim!=get_num_features())
	{
		SG_ERROR("dimension of vec (=%d) does not match number of features (=%d)\n",
				dim, get_num_features());
	}

	SGSparseVector<ST> sv=get_sparse_feature_vector(num);

	if (sv.features)
	{
		if (abs_val)
		{
			for (int32_t i=0; i<sv.num_feat_entries; i++)
			{
				vec[sv.features[i].feat_index]+=alpha
					*CMath::abs(sv.features[i].entry);
			}
		}
		else
		{
			for (int32_t i=0; i<sv.num_feat_entries; i++)
			{
				vec[sv.features[i].feat_index]+=alpha
						*sv.features[i].entry;
			}
		}
	}

	free_sparse_feature_vector(num);
}

template<>
void CSparseFeatures<complex64_t>::add_to_dense_vec(float64_t alpha,
	int32_t num, float64_t* vec, int32_t dim, bool abs_val)
{
	SG_NOTIMPLEMENTED;
}

template<class ST> void CSparseFeatures<ST>::free_sparse_feature_vector(int32_t num)
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	//vec.free_vector();
}

template<class ST> SGSparseMatrix<ST> CSparseFeatures<ST>::get_sparse_feature_matrix()
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("Not allowed with subset\n");

	return sparse_feature_matrix;
}

template<class ST> CSparseFeatures<ST>* CSparseFeatures<ST>::get_transposed()
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("Not allowed with subset\n");

	return new CSparseFeatures<ST>(sparse_feature_matrix.get_transposed());
}

template<class ST> void CSparseFeatures<ST>::set_sparse_feature_matrix(SGSparseMatrix<ST> sm)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("Not allowed with subset\n");

	sparse_feature_matrix=sm;
}

template<class ST> SGMatrix<ST> CSparseFeatures<ST>::get_full_feature_matrix()
{
	SGMatrix<ST> full(get_num_features(), get_num_vectors());
	full.zero();

	SG_INFO("converting sparse features to full feature matrix of %d x %d"
			" entries\n", sparse_feature_matrix.num_vectors, get_num_features())

	for (int32_t v=0; v<full.num_cols; v++)
	{
		int32_t idx=m_subset_stack->subset_idx_conversion(v);
		SGSparseVector<ST> current=sparse_feature_matrix[idx];

		for (int32_t f=0; f<current.num_feat_entries; f++)
		{
			int64_t offs=(v*get_num_features())
					+current.features[f].feat_index;

			full.matrix[offs]=current.features[f].entry;
		}
	}

	return full;
}

template<class ST> void CSparseFeatures<ST>::free_sparse_features()
{
	free_sparse_feature_matrix();
	SG_UNREF(feature_cache);
}

template<class ST> void CSparseFeatures<ST>::free_sparse_feature_matrix()
{
	sparse_feature_matrix=SGSparseMatrix<ST>();
}

template<class ST> bool CSparseFeatures<ST>::set_full_feature_matrix(SGMatrix<ST> full)
{
	remove_all_subsets();

	ST* src=full.matrix;
	int32_t num_feat=full.num_rows;
	int32_t num_vec=full.num_cols;

	free_sparse_feature_matrix();
	bool result=true;

	SG_INFO("converting dense feature matrix to sparse one\n")
	int32_t* num_feat_entries=SG_MALLOC(int, num_vec);

	if (num_feat_entries)
	{
		int64_t num_total_entries=0;

		// count nr of non sparse features
		for (int32_t i=0; i<num_vec; i++)
		{
			num_feat_entries[i]=0;
			for (int32_t j=0; j<num_feat; j++)
			{
				if (src[i*((int64_t) num_feat) + j] != static_cast<ST>(0))
					num_feat_entries[i]++;
			}
		}

		if (num_vec>0)
		{
			sparse_feature_matrix=SGSparseMatrix<ST>(num_feat, num_vec);
			for (int32_t i=0; i< num_vec; i++)
			{
				sparse_feature_matrix[i] =SGSparseVector<ST>(num_feat_entries[i]);
				int32_t sparse_feat_idx=0;

				for (int32_t j=0; j< num_feat; j++)
				{
					int64_t pos= i*num_feat + j;

					if (src[pos] != static_cast<ST>(0))
					{
						sparse_feature_matrix[i].features[sparse_feat_idx].entry=src[pos];
						sparse_feature_matrix[i].features[sparse_feat_idx].feat_index=j;
						sparse_feat_idx++;
						num_total_entries++;
					}
				}
			}

			SG_INFO("sparse feature matrix has %ld entries (full matrix had %ld, sparsity %2.2f%%)\n",
					num_total_entries, int64_t(num_feat)*num_vec, (100.0*num_total_entries)/(int64_t(num_feat)*num_vec));
		}
		else
		{
			SG_ERROR("huh ? zero size matrix given ?\n")
			result=false;
		}
	}
	SG_FREE(num_feat_entries);
	return result;
}

template<class ST> bool CSparseFeatures<ST>::apply_preprocessor(bool force_preprocessing)
{
	SG_INFO("force: %d\n", force_preprocessing)

	if ( sparse_feature_matrix.sparse_matrix && get_num_preprocessors() )
	{
		for (int32_t i=0; i<get_num_preprocessors(); i++)
		{
			if ( (!is_preprocessed(i) || force_preprocessing) )
			{
				set_preprocessed(i);
				SG_INFO("preprocessing using preproc %s\n", get_preprocessor(i)->get_name())
				if (((CSparsePreprocessor<ST>*) get_preprocessor(i))->apply_to_sparse_feature_matrix(this) == NULL)
					return false;
			}
			return true;
		}
		return true;
	}
	else
	{
		SG_WARNING("no sparse feature matrix available or features already preprocessed - skipping.\n")
		return false;
	}
}

template<class ST> bool CSparseFeatures<ST>::obtain_from_simple(CDenseFeatures<ST>* sf)
{
	SGMatrix<ST> fm=sf->get_feature_matrix();
	ASSERT(fm.matrix && fm.num_cols>0 && fm.num_rows>0)

	return set_full_feature_matrix(fm);
}

template<> bool CSparseFeatures<complex64_t>::obtain_from_simple(CDenseFeatures<complex64_t>* sf)
{
	SG_NOTIMPLEMENTED;
	return false;
}

template<class ST> int32_t  CSparseFeatures<ST>::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : sparse_feature_matrix.num_vectors;
}

template<class ST> int32_t  CSparseFeatures<ST>::get_num_features() const
{
	return sparse_feature_matrix.num_features;
}

template<class ST> int32_t CSparseFeatures<ST>::set_num_features(int32_t num)
{
	int32_t n=get_num_features();
	ASSERT(n<=num)
	sparse_feature_matrix.num_features=num;
	return sparse_feature_matrix.num_features;
}

template<class ST> EFeatureClass CSparseFeatures<ST>::get_feature_class() const
{
	return C_SPARSE;
}

template<class ST> void CSparseFeatures<ST>::free_feature_vector(int32_t num)
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	//vec.free_vector();
}

template<class ST> int64_t CSparseFeatures<ST>::get_num_nonzero_entries()
{
	int64_t num=0;
	index_t num_vec=get_num_vectors();
	for (int32_t i=0; i<num_vec; i++)
		num+=sparse_feature_matrix[m_subset_stack->subset_idx_conversion(i)].num_feat_entries;

	return num;
}

template<class ST> float64_t* CSparseFeatures<ST>::compute_squared(float64_t* sq)
{
	ASSERT(sq)

	index_t num_vec=get_num_vectors();
	for (int32_t i=0; i<num_vec; i++)
	{
		sq[i]=0;
		SGSparseVector<ST> vec=get_sparse_feature_vector(i);

		for (int32_t j=0; j<vec.num_feat_entries; j++)
			sq[i]+=vec.features[j].entry*vec.features[j].entry;

		free_feature_vector(i);
	}

	return sq;
}

template<> float64_t* CSparseFeatures<complex64_t>::compute_squared(float64_t* sq)
{
	SG_NOTIMPLEMENTED;
	return sq;
}

template<class ST> float64_t CSparseFeatures<ST>::compute_squared_norm(
		CSparseFeatures<float64_t>* lhs, float64_t* sq_lhs, int32_t idx_a,
		CSparseFeatures<float64_t>* rhs, float64_t* sq_rhs, int32_t idx_b)
{
	int32_t i,j;
	ASSERT(lhs)
	ASSERT(rhs)

	SGSparseVector<float64_t> avec=lhs->get_sparse_feature_vector(idx_a);
	SGSparseVector<float64_t> bvec=rhs->get_sparse_feature_vector(idx_b);
	ASSERT(avec.features)
	ASSERT(bvec.features)

	float64_t result=sq_lhs[idx_a]+sq_rhs[idx_b];

	if (avec.num_feat_entries<=bvec.num_feat_entries)
	{
		j=0;
		for (i=0; i<avec.num_feat_entries; i++)
		{
			int32_t a_feat_idx=avec.features[i].feat_index;

			while ((j<bvec.num_feat_entries)
					&&(bvec.features[j].feat_index<a_feat_idx))
				j++;

			if ((j<bvec.num_feat_entries)
					&&(bvec.features[j].feat_index==a_feat_idx))
			{
				result-=2*(avec.features[i].entry*bvec.features[j].entry);
				j++;
			}
		}
	}
	else
	{
		j=0;
		for (i=0; i<bvec.num_feat_entries; i++)
		{
			int32_t b_feat_idx=bvec.features[i].feat_index;

			while ((j<avec.num_feat_entries)
					&&(avec.features[j].feat_index<b_feat_idx))
				j++;

			if ((j<avec.num_feat_entries)
					&&(avec.features[j].feat_index==b_feat_idx))
			{
				result-=2*(bvec.features[i].entry*avec.features[j].entry);
				j++;
			}
		}
	}

	((CSparseFeatures<float64_t>*) lhs)->free_feature_vector(idx_a);
	((CSparseFeatures<float64_t>*) rhs)->free_feature_vector(idx_b);

	return CMath::abs(result);
}

template<class ST> int32_t CSparseFeatures<ST>::get_dim_feature_space() const
{
	return get_num_features();
}

template<class ST> float64_t CSparseFeatures<ST>::dot(int32_t vec_idx1,
		CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CSparseFeatures<ST>* sf = (CSparseFeatures<ST>*) df;

	SGSparseVector<ST> avec=get_sparse_feature_vector(vec_idx1);
	SGSparseVector<ST> bvec=sf->get_sparse_feature_vector(vec_idx2);

	float64_t result = SGSparseVector<ST>::sparse_dot(avec, bvec);
	free_sparse_feature_vector(vec_idx1);
	sf->free_sparse_feature_vector(vec_idx2);

	return result;
}

template<> float64_t CSparseFeatures<complex64_t>::dot(int32_t vec_idx1,
		CDotFeatures* df, int32_t vec_idx2)
{
	SG_NOTIMPLEMENTED;
	return 0.0;
}

template<class ST> float64_t CSparseFeatures<ST>::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2)
	if (vec2_len!=get_num_features())
	{
		SG_ERROR("dimension of vec2 (=%d) does not match number of features (=%d)\n",
				vec2_len, get_num_features());
	}
	float64_t result=0;

	SGSparseVector<ST> sv=get_sparse_feature_vector(vec_idx1);

	if (sv.features)
	{
		for (int32_t i=0; i<sv.num_feat_entries; i++)
			result+=vec2[sv.features[i].feat_index]*sv.features[i].entry;
	}

	free_sparse_feature_vector(vec_idx1);

	return result;
}

template<> float64_t CSparseFeatures<complex64_t>::dense_dot(int32_t vec_idx1,
	const float64_t* vec2, int32_t vec2_len)
{
	SG_NOTIMPLEMENTED;
	return 0.0;
}

template<class ST> void* CSparseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
				"requested %d)\n", get_num_vectors(), vector_index);
	}

	if (!sparse_feature_matrix.sparse_matrix)
		SG_ERROR("Requires a in-memory feature matrix\n")

	sparse_feature_iterator* it=new sparse_feature_iterator();
	it->sv=get_sparse_feature_vector(vector_index);
	it->index=0;
	it->vector_index=vector_index;

	return it;
}

template<class ST> bool CSparseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	sparse_feature_iterator* it=(sparse_feature_iterator*) iterator;
	if (!it || it->index>=it->sv.num_feat_entries)
		return false;

	int32_t i=it->index++;

	index=it->sv.features[i].feat_index;
	value=(float64_t) it->sv.features[i].entry;

	return true;
}

template<> bool CSparseFeatures<complex64_t>::get_next_feature(int32_t& index,
	float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}

template<class ST> void CSparseFeatures<ST>::free_feature_iterator(void* iterator)
{
	if (!iterator)
		return;

	delete ((sparse_feature_iterator*) iterator);
}

template<class ST> CFeatures* CSparseFeatures<ST>::copy_subset(SGVector<index_t> indices)
{
	SGSparseMatrix<ST> matrix_copy=SGSparseMatrix<ST>(get_dim_feature_space(),
			indices.vlen);

	for (index_t i=0; i<indices.vlen; ++i)
	{
		/* index to copy */
		index_t index=indices.vector[i];
		index_t real_index=m_subset_stack->subset_idx_conversion(index);

		/* copy sparse vector */
		SGSparseVector<ST> current=get_sparse_feature_vector(real_index);
		matrix_copy.sparse_matrix[i]=current;

		free_sparse_feature_vector(index);
	}

	CFeatures* result=new CSparseFeatures<ST>(matrix_copy);
	return result;
}

template<class ST> SGSparseVectorEntry<ST>* CSparseFeatures<ST>::compute_sparse_feature_vector(int32_t num,
	int32_t& len, SGSparseVectorEntry<ST>* target)
{
	SG_NOTIMPLEMENTED

	len=0;
	return NULL;
}

template<class ST> void CSparseFeatures<ST>::sort_features()
{
	sparse_feature_matrix.sort_features();
}

template<class ST> CRegressionLabels* CSparseFeatures<ST>::load_svmlight_file(char* fname, bool do_sort_features) 
{
	return sparse_feature_matrix.load_svmlight_file(fname, do_sort_features);
}

template<class ST> bool CSparseFeatures<ST>::write_svmlight_file(char* fname, CRegressionLabels* label)
{
	return sparse_feature_matrix.write_svmlight_file(fname, label);
}

template<class ST> void CSparseFeatures<ST>::init()
{
	set_generic<ST>();

	m_parameters->add_vector(&sparse_feature_matrix.sparse_matrix, &sparse_feature_matrix.num_vectors,
			"sparse_feature_matrix",
			"Array of sparse vectors.");
	m_parameters->add(&sparse_feature_matrix.num_features, "sparse_feature_matrix.num_features",
			"Total number of features.");
}

#define GET_FEATURE_TYPE(sg_type, f_type)									\
template<> EFeatureType CSparseFeatures<sg_type>::get_feature_type() const	\
{																			\
	return f_type;															\
}
GET_FEATURE_TYPE(bool, F_BOOL)
GET_FEATURE_TYPE(char, F_CHAR)
GET_FEATURE_TYPE(uint8_t, F_BYTE)
GET_FEATURE_TYPE(int8_t, F_BYTE)
GET_FEATURE_TYPE(int16_t, F_SHORT)
GET_FEATURE_TYPE(uint16_t, F_WORD)
GET_FEATURE_TYPE(int32_t, F_INT)
GET_FEATURE_TYPE(uint32_t, F_UINT)
GET_FEATURE_TYPE(int64_t, F_LONG)
GET_FEATURE_TYPE(uint64_t, F_ULONG)
GET_FEATURE_TYPE(float32_t, F_SHORTREAL)
GET_FEATURE_TYPE(float64_t, F_DREAL)
GET_FEATURE_TYPE(floatmax_t, F_LONGREAL)
GET_FEATURE_TYPE(complex64_t, F_ANY)
#undef GET_FEATURE_TYPE

#define LOAD(fname, sg_type)											\
template<> void CSparseFeatures<sg_type>::load(CFile* loader)			\
{																		\
	remove_all_subsets();												\
	SG_SET_LOCALE_C;													\
	ASSERT(loader)														\
	SGSparseVector<sg_type>* matrix=NULL;								\
	int32_t num_feat=0;													\
	int32_t num_vec=0;													\
	loader->fname(matrix, num_feat, num_vec);							\
	set_sparse_feature_matrix(SGSparseMatrix<sg_type>(matrix, num_feat, num_vec));				\
	SG_RESET_LOCALE;													\
}
LOAD(get_sparse_matrix, bool)
LOAD(get_sparse_matrix, char)
LOAD(get_sparse_matrix, uint8_t)
LOAD(get_sparse_matrix, int8_t)
LOAD(get_sparse_matrix, int16_t)
LOAD(get_sparse_matrix, uint16_t)
LOAD(get_sparse_matrix, int32_t)
LOAD(get_sparse_matrix, uint32_t)
LOAD(get_sparse_matrix, int64_t)
LOAD(get_sparse_matrix, uint64_t)
LOAD(get_sparse_matrix, float32_t)
LOAD(get_sparse_matrix, float64_t)
LOAD(get_sparse_matrix, floatmax_t)
#undef LOAD

#define WRITE(fname, sg_type)											\
template<> void CSparseFeatures<sg_type>::save(CFile* writer)			\
{																		\
	if (m_subset_stack->has_subsets())									\
	SG_ERROR("Not allowed with subset\n");								\
	SG_SET_LOCALE_C;													\
	ASSERT(writer)														\
	writer->fname(sparse_feature_matrix.sparse_matrix, sparse_feature_matrix.num_features, sparse_feature_matrix.num_vectors);	\
	SG_RESET_LOCALE;													\
}
WRITE(set_sparse_matrix, bool)
WRITE(set_sparse_matrix, char)
WRITE(set_sparse_matrix, uint8_t)
WRITE(set_sparse_matrix, int8_t)
WRITE(set_sparse_matrix, int16_t)
WRITE(set_sparse_matrix, uint16_t)
WRITE(set_sparse_matrix, int32_t)
WRITE(set_sparse_matrix, uint32_t)
WRITE(set_sparse_matrix, int64_t)
WRITE(set_sparse_matrix, uint64_t)
WRITE(set_sparse_matrix, float32_t)
WRITE(set_sparse_matrix, float64_t)
WRITE(set_sparse_matrix, floatmax_t)
#undef WRITE

#define UNDEFINED(fname, type)	\
template<> void CSparseFeatures<type>::fname(CFile* f)	\
{	\
	SG_NOTIMPLEMENTED;	\
}
UNDEFINED(load, complex64_t)
UNDEFINED(save, complex64_t)
#undef UNDEFINED

template class CSparseFeatures<bool>;
template class CSparseFeatures<char>;
template class CSparseFeatures<int8_t>;
template class CSparseFeatures<uint8_t>;
template class CSparseFeatures<int16_t>;
template class CSparseFeatures<uint16_t>;
template class CSparseFeatures<int32_t>;
template class CSparseFeatures<uint32_t>;
template class CSparseFeatures<int64_t>;
template class CSparseFeatures<uint64_t>;
template class CSparseFeatures<float32_t>;
template class CSparseFeatures<float64_t>;
template class CSparseFeatures<floatmax_t>;
template class CSparseFeatures<complex64_t>;
}
