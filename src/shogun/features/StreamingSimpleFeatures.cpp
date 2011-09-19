#include <shogun/mathematics/Math.h>
#include <shogun/features/StreamingSimpleFeatures.h>
#include <shogun/io/StreamingFileFromSimpleFeatures.h>

namespace shogun
{
template <class T> CStreamingSimpleFeatures<T>::CStreamingSimpleFeatures() : CStreamingDotFeatures()
{
	set_read_functions();
	init();
	parser.set_free_vector_after_release(false);
}

template <class T> CStreamingSimpleFeatures<T>::CStreamingSimpleFeatures(CStreamingFile* file,
			 bool is_labelled,
			 int32_t size)
	: CStreamingDotFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
}

template <class T> CStreamingSimpleFeatures<T>::CStreamingSimpleFeatures(CSimpleFeatures<T>* simple_features,
			 float64_t* lab)
	: CStreamingDotFeatures()
{
	CStreamingFileFromSimpleFeatures<T>* file;
	bool is_labelled;
	int32_t size = 1024;

	if (lab)
	{
		is_labelled = true;
		file = new CStreamingFileFromSimpleFeatures<T>(simple_features, lab);
	}
	else
	{
		is_labelled = false;
		file = new CStreamingFileFromSimpleFeatures<T>(simple_features);
	}

	SG_REF(file);

	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
	parser.set_free_vectors_on_destruct(false);
	seekable=true;
}

template <class T> CStreamingSimpleFeatures<T>::~CStreamingSimpleFeatures()
{
	parser.end_parser();
}

template <class T> void CStreamingSimpleFeatures<T>::reset_stream()
{
	if (seekable)
	{
		((CStreamingFileFromSimpleFeatures<T>*) working_file)->reset_stream();
		parser.exit_parser();
		parser.init(working_file, has_labels, 1);
		parser.set_free_vector_after_release(false);
		parser.start_parser();
	}
}

template <class T> float32_t CStreamingSimpleFeatures<T>::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_length);
	float32_t result=0;

	for (int32_t i=0; i<current_length; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template <class T> float64_t CStreamingSimpleFeatures<T>::dense_dot(const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_length);
	float64_t result=0;

	for (int32_t i=0; i<current_length; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template <class T> void CStreamingSimpleFeatures<T>::add_to_dense_vec(float32_t alpha, float32_t* vec2, int32_t vec2_len , bool abs_val)
{
	ASSERT(vec2_len==current_length);

	if (abs_val)
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template <class T> void CStreamingSimpleFeatures<T>::add_to_dense_vec(float64_t alpha, float64_t* vec2, int32_t vec2_len , bool abs_val)
{
	ASSERT(vec2_len==current_length);

	if (abs_val)
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template <class T> int32_t CStreamingSimpleFeatures<T>::get_nnz_features_for_vector()
{
	return current_length;
}

template <class T> CFeatures* CStreamingSimpleFeatures<T>::duplicate() const
{
	return new CStreamingSimpleFeatures<T>(*this);
}

template <class T> int32_t CStreamingSimpleFeatures<T>::get_num_vectors() const
{
	if (current_vector)
		return 1;
	return 0;
}

template <class T> int32_t CStreamingSimpleFeatures<T>::get_size()
{
	return sizeof(T);
}

template <class T>
void CStreamingSimpleFeatures<T>::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_vector);
}

template <class T>
void CStreamingSimpleFeatures<T>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_vector_and_label);
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> EFeatureType CStreamingSimpleFeatures<sg_type>::get_feature_type() \
{									\
	return f_type;							\
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


template <class T>
void CStreamingSimpleFeatures<T>::init()
{
	working_file=NULL;
	current_vector=NULL;
	seekable=false;
	current_length=-1;
}

template <class T>
void CStreamingSimpleFeatures<T>::init(CStreamingFile* file,
				    bool is_labelled,
				    int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
	seekable=false;
}

template <class T>
void CStreamingSimpleFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class T>
void CStreamingSimpleFeatures<T>::end_parser()
{
	parser.end_parser();
}

template <class T>
bool CStreamingSimpleFeatures<T>::get_next_example()
{
	bool ret_value;
	ret_value = (bool) parser.get_next_example(current_vector,
						   current_length,
						   current_label);

	return ret_value;
}

template <class T>
SGVector<T> CStreamingSimpleFeatures<T>::get_vector()
{
	current_sgvector.vector=current_vector;
	current_sgvector.vlen=current_length;

	return current_sgvector;
}

template <class T>
float64_t CStreamingSimpleFeatures<T>::get_label()
{
	ASSERT(has_labels);

	return current_label;
}

template <class T>
void CStreamingSimpleFeatures<T>::release_example()
{
	parser.finalize_example();
}

template <class T>
int32_t CStreamingSimpleFeatures<T>::get_dim_feature_space() const
{
	return current_length;
}

template <class T>
	float32_t CStreamingSimpleFeatures<T>::dot(CStreamingDotFeatures* df)
{
	ASSERT(df);
	ASSERT(df->get_feature_type() == get_feature_type());
	ASSERT(df->get_feature_class() == get_feature_class());
	CStreamingSimpleFeatures<T>* sf = (CStreamingSimpleFeatures<T>*) df;

	SGVector<T> other_vector=sf->get_vector();

	float32_t result = CMath::dot(current_vector, other_vector.vector, current_length);

	return result;
}

template <class T>
float32_t CStreamingSimpleFeatures<T>::dot(SGVector<T> sgvec1)
{
	int32_t len1;
	len1=sgvec1.vlen;

	if (len1 != current_length)
		SG_ERROR("Lengths %d and %d not equal while computing dot product!\n", len1, current_length);

	float32_t result=CMath::dot(current_vector, sgvec1.vector, len1);
	return result;
}

template <class T>
int32_t CStreamingSimpleFeatures<T>::get_num_features()
{
	return current_length;
}

template <class T>
EFeatureClass CStreamingSimpleFeatures<T>::get_feature_class()
{
	return C_STREAMING_SIMPLE;
}

template class CStreamingSimpleFeatures<bool>;
template class CStreamingSimpleFeatures<char>;
template class CStreamingSimpleFeatures<int8_t>;
template class CStreamingSimpleFeatures<uint8_t>;
template class CStreamingSimpleFeatures<int16_t>;
template class CStreamingSimpleFeatures<uint16_t>;
template class CStreamingSimpleFeatures<int32_t>;
template class CStreamingSimpleFeatures<uint32_t>;
template class CStreamingSimpleFeatures<int64_t>;
template class CStreamingSimpleFeatures<uint64_t>;
template class CStreamingSimpleFeatures<float32_t>;
template class CStreamingSimpleFeatures<float64_t>;
template class CStreamingSimpleFeatures<floatmax_t>;
}
