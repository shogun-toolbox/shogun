#include <shogun/features/SimpleFeatures.h>

namespace shogun {
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
