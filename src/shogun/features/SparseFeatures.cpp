#include <shogun/lib/common.h>
#include <shogun/lib/memory.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/preprocessor/SparsePreprocessor.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

#include <string.h>
#include <stdlib.h>

namespace shogun
{

template <class ST> class SparsePreprocessor;

template<class ST> SparseFeatures<ST>::SparseFeatures(int32_t size)
: DotFeatures(size), feature_cache(NULL)
{
	init();
}

template<class ST> SparseFeatures<ST>::SparseFeatures(SGSparseMatrix<ST> sparse)
: SparseFeatures(0)
{
	set_sparse_feature_matrix(sparse);
}

template<class ST> SparseFeatures<ST>::SparseFeatures(SGMatrix<ST> dense)
: SparseFeatures(0)
{
	set_full_feature_matrix(dense);
}

template<class ST> SparseFeatures<ST>::SparseFeatures(const SparseFeatures & orig)
: DotFeatures(orig), sparse_feature_matrix(orig.sparse_feature_matrix),
	feature_cache(orig.feature_cache)
{
	init();

	m_subset_stack=orig.m_subset_stack;

}

template <class ST>
SparseFeatures<ST>::SparseFeatures(std::shared_ptr<DenseFeatures<ST>> dense)
	: SparseFeatures(0)
{
	SGMatrix<ST> fm=dense->get_feature_matrix();
	ASSERT(fm.matrix && fm.num_cols>0 && fm.num_rows>0)
	set_full_feature_matrix(fm);
}

template<> SparseFeatures<complex128_t>::SparseFeatures(std::shared_ptr<DenseFeatures<complex128_t>> dense)
{
	not_implemented(SOURCE_LOCATION);;
}

template<class ST> SparseFeatures<ST>::SparseFeatures(const std::shared_ptr<File>& loader)
: DotFeatures(), feature_cache(NULL)
{
	init();

	load(loader);
}

template<class ST> SparseFeatures<ST>::~SparseFeatures()
{

}

template<class ST> std::shared_ptr<Features> SparseFeatures<ST>::duplicate() const
{
	return std::make_shared<SparseFeatures>(*this);
}

template<class ST> ST SparseFeatures<ST>::get_feature(int32_t num, int32_t index) const
{
	require(index>=0 && index<get_num_features(),
		"get_feature(num={},index={}): index exceeds [0;{}]",
		num, index, get_num_features()-1);

	SGSparseVector<ST> sv=get_sparse_feature_vector(num);
	ST ret = sv.get_feature(index);

	free_sparse_feature_vector(num);
	return ret;
}

template<class ST> SGVector<ST> SparseFeatures<ST>::get_full_feature_vector(int32_t num)
{
	SGSparseVector<ST> sv=get_sparse_feature_vector(num);
	SGVector<ST> dense = sv.get_dense(get_num_features());
	free_sparse_feature_vector(num);
	return dense;
}

template<class ST> int32_t SparseFeatures<ST>::get_nnz_features_for_vector(int32_t num) const
{
	SGSparseVector<ST> sv = get_sparse_feature_vector(num);
	int32_t len=sv.num_feat_entries;
	free_sparse_feature_vector(num);
	return len;
}

template<class ST> SGSparseVector<ST> SparseFeatures<ST>::get_sparse_feature_vector(int32_t num) const
{
	require(num>=0 && num<get_num_vectors(),
		"get_sparse_feature_vector(num={}): num exceeds [0;{}]",
		num, get_num_vectors()-1);
	index_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (sparse_feature_matrix.sparse_matrix)
	{
		return sparse_feature_matrix[real_num];
	}
	else
	{
		SGSparseVector<ST> result;
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
			auto tmp_feat_before=result.features;
			typename SGSparseVector<ST>::pointer tmp_feat_after = nullptr;

			for (int32_t i=0; i<get_num_preprocessors(); i++)
			{
				//tmp_feat_after=((SparsePreprocessor<ST>*) get_preproc(i))->apply_to_feature_vector(tmp_feat_before, tmp_len);

				if (i!=0)	// delete feature vector, except for the the first one, i.e., feat
					SG_FREE(tmp_feat_before);
				tmp_feat_before=tmp_feat_after;
			}

			if (tmp_feat_after)
			{
				sg_memcpy(result.features, tmp_feat_after,
						sizeof(SGSparseVectorEntry<ST>)*tmp_len);

				SG_FREE(tmp_feat_after);
				result.num_feat_entries=tmp_len;
			}
			SG_DEBUG("len: {} len2: {}", result.num_feat_entries, get_num_features())
		}
		return result ;
	}
}

template<class ST> ST SparseFeatures<ST>::dense_dot(ST alpha, int32_t num, ST* vec, int32_t dim, ST b) const
{
	SGSparseVector<ST> sv=get_sparse_feature_vector(num);
	ST result = sv.dense_dot(alpha,vec,dim,b);
	free_sparse_feature_vector(num);
	return result;
}

template<class ST> void SparseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t num, float64_t* vec, int32_t dim, bool abs_val) const
{
	require(vec, "add_to_dense_vec(num={},dim={}): vec must not be NULL",
		num, dim);
	require(dim>=get_num_features(),
		"add_to_dense_vec(num={},dim={}): dim should contain number of features {}",
		num, dim, get_num_features());

	SGSparseVector<ST> sv=get_sparse_feature_vector(num);

	if (sv.features)
	{
		if (abs_val)
		{
			for (int32_t i=0; i<sv.num_feat_entries; i++)
			{
				vec[sv.features[i].feat_index]+=alpha
					*Math::abs(sv.features[i].entry);
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
void SparseFeatures<complex128_t>::add_to_dense_vec(float64_t alpha,
	int32_t num, float64_t* vec, int32_t dim, bool abs_val) const
{
	not_implemented(SOURCE_LOCATION);;
}

template<class ST> void SparseFeatures<ST>::free_sparse_feature_vector(int32_t num) const
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	//vec.free_vector();
}

template<class ST> SGSparseMatrix<ST> SparseFeatures<ST>::get_sparse_feature_matrix()
{
	if (m_subset_stack->has_subsets())
		error("Not allowed with subset");

	return sparse_feature_matrix;
}

template<class ST> std::shared_ptr<SparseFeatures<ST>> SparseFeatures<ST>::get_transposed()
{
	if (m_subset_stack->has_subsets())
		error("Not allowed with subset");

	return std::make_shared<SparseFeatures>(sparse_feature_matrix.get_transposed());
}

template<class ST> void SparseFeatures<ST>::set_sparse_feature_matrix(SGSparseMatrix<ST> sm)
{
	if (m_subset_stack->has_subsets())
		error("Not allowed with subset");

	sparse_feature_matrix=sm;

	// TODO: check should be implemented in sparse matrix class
	for (int32_t j=0; j<get_num_vectors(); j++) {
		SGSparseVector<ST> sv=get_sparse_feature_vector(j);
		require(get_num_features() >= sv.get_num_dimensions(),
			"sparse_matrix[{}] check failed (matrix features {} >= vector dimension {})",
			j, get_num_features(), sv.get_num_dimensions());
	}
}

template<class ST> SGMatrix<ST> SparseFeatures<ST>::get_full_feature_matrix()
{
	SGMatrix<ST> full(get_num_features(), get_num_vectors());
	full.zero();

	io::info("converting sparse features to full feature matrix of {} x {}"
			" entries", sparse_feature_matrix.num_vectors, get_num_features());

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

template<class ST> void SparseFeatures<ST>::free_sparse_features()
{
	free_sparse_feature_matrix();

}

template<class ST> void SparseFeatures<ST>::free_sparse_feature_matrix()
{
	sparse_feature_matrix=SGSparseMatrix<ST>();
}

template<class ST> void SparseFeatures<ST>::set_full_feature_matrix(SGMatrix<ST> full)
{
	remove_all_subsets();
	free_sparse_feature_matrix();
	sparse_feature_matrix.from_dense(full);
}

template<class ST> int32_t  SparseFeatures<ST>::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : sparse_feature_matrix.num_vectors;
}

template<class ST> int32_t  SparseFeatures<ST>::get_num_features() const
{
	return sparse_feature_matrix.num_features;
}

template<class ST> int32_t SparseFeatures<ST>::set_num_features(int32_t num)
{
	int32_t n=get_num_features();
	ASSERT(n<=num)
	sparse_feature_matrix.num_features=num;
	return sparse_feature_matrix.num_features;
}

template<class ST> EFeatureClass SparseFeatures<ST>::get_feature_class() const
{
	return C_SPARSE;
}

template<class ST> void SparseFeatures<ST>::free_feature_vector(int32_t num) const
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	//vec.free_vector();
}

template<class ST> int64_t SparseFeatures<ST>::get_num_nonzero_entries()
{
	int64_t num=0;
	index_t num_vec=get_num_vectors();
	for (int32_t i=0; i<num_vec; i++)
		num+=sparse_feature_matrix[m_subset_stack->subset_idx_conversion(i)].num_feat_entries;

	return num;
}

template<class ST> float64_t* SparseFeatures<ST>::compute_squared(float64_t* sq)
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

template<> float64_t* SparseFeatures<complex128_t>::compute_squared(float64_t* sq)
{
	not_implemented(SOURCE_LOCATION);;
	return sq;
}

template<class ST> float64_t SparseFeatures<ST>::compute_squared_norm(
		const std::shared_ptr<SparseFeatures<float64_t>>& lhs, float64_t* sq_lhs, int32_t idx_a,
		const std::shared_ptr<SparseFeatures<float64_t>>& rhs, float64_t* sq_rhs, int32_t idx_b)
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

	lhs->free_feature_vector(idx_a);
	rhs->free_feature_vector(idx_b);

	return Math::abs(result);
}

template<class ST> int32_t SparseFeatures<ST>::get_dim_feature_space() const
{
	return get_num_features();
}

template<class ST> float64_t SparseFeatures<ST>::dot(int32_t vec_idx1,
		std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	auto sf = std::dynamic_pointer_cast<SparseFeatures<ST>>(df);

	SGSparseVector<ST> avec=get_sparse_feature_vector(vec_idx1);
	SGSparseVector<ST> bvec=sf->get_sparse_feature_vector(vec_idx2);

	float64_t result = SGSparseVector<ST>::sparse_dot(avec, bvec);
	free_sparse_feature_vector(vec_idx1);
	sf->free_sparse_feature_vector(vec_idx2);

	return result;
}

template<> float64_t SparseFeatures<complex128_t>::dot(int32_t vec_idx1,
		std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	not_implemented(SOURCE_LOCATION);;
	return 0.0;
}

template <class ST>
float64_t
SparseFeatures<ST>::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	require(
		vec2.size() >= get_num_features(),
		"dot(vec_idx1={},vec2_len={}): vec2_len should contain number of "
		"features {} {}",
		vec_idx1, vec2.size(), get_num_features());

	float64_t result=0;
	SGSparseVector<ST> sv=get_sparse_feature_vector(vec_idx1);

	if (sv.features)
	{
		require(get_num_features() >= sv.get_num_dimensions(),
			"sparse_matrix[{}] check failed (matrix features {} >= vector dimension {})",
			vec_idx1, get_num_features(), sv.get_num_dimensions());

		require(
			vec2.size() >= sv.get_num_dimensions(),
			"sparse_matrix[{}] check failed (dense vector dimension {} >= "
			"vector dimension {})",
			vec_idx1, vec2.size(), sv.get_num_dimensions());

		for (int32_t i=0; i<sv.num_feat_entries; i++)
			result+=vec2[sv.features[i].feat_index]*sv.features[i].entry;
	}

	free_sparse_feature_vector(vec_idx1);

	return result;
}

template <>
float64_t SparseFeatures<complex128_t>::dot(
	int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	not_implemented(SOURCE_LOCATION);;
	return 0.0;
}

template<class ST> void* SparseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=get_num_vectors())
	{
		error("Index out of bounds (number of vectors {}, you "
				"requested {})", get_num_vectors(), vector_index);
	}

	if (!sparse_feature_matrix.sparse_matrix)
		error("Requires a in-memory feature matrix");

	sparse_feature_iterator* it=new sparse_feature_iterator();
	it->sv=get_sparse_feature_vector(vector_index);
	it->index=0;
	it->vector_index=vector_index;

	return it;
}

template<class ST> bool SparseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	sparse_feature_iterator* it=(sparse_feature_iterator*) iterator;
	if (!it || it->index>=it->sv.num_feat_entries)
		return false;

	int32_t i=it->index++;

	index=it->sv.features[i].feat_index;
	value=(float64_t) it->sv.features[i].entry;

	return true;
}

template<> bool SparseFeatures<complex128_t>::get_next_feature(int32_t& index,
	float64_t& value, void* iterator)
{
	not_implemented(SOURCE_LOCATION);;
	return false;
}

template<class ST> void SparseFeatures<ST>::free_feature_iterator(void* iterator)
{
	if (!iterator)
		return;

	delete ((sparse_feature_iterator*) iterator);
}

template<class ST> std::shared_ptr<Features> SparseFeatures<ST>::copy_subset(SGVector<index_t> indices ) const
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

	return std::make_shared<SparseFeatures>(matrix_copy);
}

template<class ST> SGSparseVectorEntry<ST>* SparseFeatures<ST>::compute_sparse_feature_vector(int32_t num,
	int32_t& len, SGSparseVectorEntry<ST>* target) const
{
	not_implemented(SOURCE_LOCATION);

	len=0;
	return NULL;
}

template<class ST> void SparseFeatures<ST>::sort_features()
{
	sparse_feature_matrix.sort_features();
}

template<class ST> void SparseFeatures<ST>::init()
{
	set_generic<ST>();

	/*m_parameters->add_vector(&sparse_feature_matrix.sparse_matrix, &sparse_feature_matrix.num_vectors,
			"sparse_feature_matrix",
			"Array of sparse vectors.");*/
	watch_param(
		"sparse_feature_matrix", &sparse_feature_matrix.sparse_matrix,
		&sparse_feature_matrix.num_vectors);
	watch_param("sparse_feature_matrix.num_features",  &sparse_feature_matrix.num_features);

	/*m_parameters->add(&sparse_feature_matrix.num_features, "sparse_feature_matrix.num_features",
			"Total number of features.");*/
}

#define GET_FEATURE_TYPE(sg_type, f_type)									\
template<> EFeatureType SparseFeatures<sg_type>::get_feature_type() const	\
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
GET_FEATURE_TYPE(complex128_t, F_ANY)
#undef GET_FEATURE_TYPE

template<class ST> void SparseFeatures<ST>::load(std::shared_ptr<File> loader)
{
	remove_all_subsets();
	ASSERT(loader)
	free_sparse_feature_matrix();
	sparse_feature_matrix.load(loader);
}

template<class ST> SGVector<float64_t> SparseFeatures<ST>::load_with_labels(const std::shared_ptr<File>& loader)
{
	remove_all_subsets();
	ASSERT(loader)
	free_sparse_feature_matrix();
	return sparse_feature_matrix.load_with_labels(loader);
}

template<class ST> void SparseFeatures<ST>::save(std::shared_ptr<File> writer)
{
	if (m_subset_stack->has_subsets())
		error("Not allowed with subset");
	ASSERT(writer)
	sparse_feature_matrix.save(writer);
}

template<class ST> void SparseFeatures<ST>::save_with_labels(const std::shared_ptr<File>& writer, SGVector<float64_t> labels)
{
	if (m_subset_stack->has_subsets())
		error("Not allowed with subset");
	ASSERT(writer)
	sparse_feature_matrix.save_with_labels(writer, labels);
}

template class SparseFeatures<bool>;
template class SparseFeatures<char>;
template class SparseFeatures<int8_t>;
template class SparseFeatures<uint8_t>;
template class SparseFeatures<int16_t>;
template class SparseFeatures<uint16_t>;
template class SparseFeatures<int32_t>;
template class SparseFeatures<uint32_t>;
template class SparseFeatures<int64_t>;
template class SparseFeatures<uint64_t>;
template class SparseFeatures<float32_t>;
template class SparseFeatures<float64_t>;
template class SparseFeatures<floatmax_t>;
template class SparseFeatures<complex128_t>;
}
