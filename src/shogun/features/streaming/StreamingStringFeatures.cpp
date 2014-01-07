#include <features/streaming/StreamingStringFeatures.h>

namespace shogun
{


template <class T>
CStreamingStringFeatures<T>::CStreamingStringFeatures() : CStreamingFeatures()
{
	init();
	set_read_functions();
	remap_to_bin=false;
}

template <class T>
CStreamingStringFeatures<T>::CStreamingStringFeatures(CStreamingFile* file,
			 bool is_labelled,
			 int32_t size)
	: CStreamingFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
	remap_to_bin=false;
}

template <class T>
CStreamingStringFeatures<T>::~CStreamingStringFeatures()
{
	if (parser.is_running())
		parser.end_parser();
	SG_UNREF(alphabet);
}

template <class T>
void CStreamingStringFeatures<T>::use_alphabet(EAlphabet alpha)
{
	SG_UNREF(alphabet);

	alphabet=new CAlphabet(alpha);
	SG_REF(alphabet);
	num_symbols=alphabet->get_num_symbols();
}

template <class T>
void CStreamingStringFeatures<T>::use_alphabet(CAlphabet* alpha)
{
	SG_UNREF(alphabet);

	alphabet=new CAlphabet(alpha);
	SG_REF(alphabet);
	num_symbols=alphabet->get_num_symbols();
}

template <class T>
void CStreamingStringFeatures<T>::set_remap(CAlphabet* ascii_alphabet, CAlphabet* binary_alphabet)
{
	remap_to_bin=true;
	alpha_ascii=new CAlphabet(ascii_alphabet);
	alpha_bin=new CAlphabet(binary_alphabet);
}

template <class T>
void CStreamingStringFeatures<T>::set_remap(EAlphabet ascii_alphabet, EAlphabet binary_alphabet)
{
	remap_to_bin=true;
	alpha_ascii=new CAlphabet(ascii_alphabet);
	alpha_bin=new CAlphabet(binary_alphabet);
}

template <class T>
CAlphabet* CStreamingStringFeatures<T>::get_alphabet()
{
	SG_REF(alphabet);
	return alphabet;
}

template <class T>
floatmax_t CStreamingStringFeatures<T>::get_num_symbols()
{
	return num_symbols;
}

template <class T>
CFeatures* CStreamingStringFeatures<T>::duplicate() const
{
	return new CStreamingStringFeatures<T>(*this);
}

template <class T>
int32_t CStreamingStringFeatures<T>::get_num_vectors() const
{
	if (current_string)
		return 1;
	return 0;
}

template <class T>
int32_t CStreamingStringFeatures<T>::get_num_features()
{
	return current_length;
}

template <class T> void CStreamingStringFeatures<T>::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_string);
}

template <class T> void CStreamingStringFeatures<T>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label
		(&CStreamingFile::get_string_and_label);
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> EFeatureType CStreamingStringFeatures<sg_type>::get_feature_type() const \
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
void CStreamingStringFeatures<T>::init()
{
	working_file=NULL;
	alphabet=new CAlphabet();

	current_string=NULL;
	current_length=-1;
	current_sgstring.string=current_string;
	current_sgstring.slen=current_length;

	set_generic<T>();
}

template <class T>
void CStreamingStringFeatures<T>::init(CStreamingFile* file,
				       bool is_labelled,
				       int32_t size)
{
	init();
	has_labels=is_labelled;
	working_file=file;
	parser.init(file, is_labelled, size);
	parser.set_free_vector_after_release(false);
	parser.set_free_vectors_on_destruct(false);
}

template <class T>
void CStreamingStringFeatures<T>::start_parser()
{
	if (!remap_to_bin)
		alpha_ascii=alphabet;

	if (!parser.is_running())
		parser.start_parser();
}

template <class T>
void CStreamingStringFeatures<T>::end_parser()
{
	parser.end_parser();
}

template <class T>
bool CStreamingStringFeatures<T>::get_next_example()
{
	bool ret_value;

	ret_value = (bool) parser.get_next_example(current_string,
						   current_length,
						   current_label);

	if (!ret_value)
		return false;

	int32_t i;
	if (remap_to_bin)
	{
		alpha_ascii->add_string_to_histogram(current_string, current_length);

		for (i=0; i<current_length; i++)
			current_string[i]=alpha_ascii->remap_to_bin(current_string[i]);
		alpha_bin->add_string_to_histogram(current_string, current_length);
	}
	else
	{
		alpha_ascii->add_string_to_histogram(current_string, current_length);
	}

	/* Check the input using src alphabet, alpha_ascii */
	if ( !(alpha_ascii->check_alphabet_size() && alpha_ascii->check_alphabet()) )
	{
		SG_ERROR("StreamingStringFeatures: The given input was found to be incompatible with the alphabet!\n")
		return 0;
	}

	//SG_UNREF(alphabet);

	if (remap_to_bin)
		alphabet=alpha_bin;
	else
		alphabet=alpha_ascii;

	//SG_REF(alphabet);
	num_symbols=alphabet->get_num_symbols();

	return ret_value;
}

template <class T>
SGString<T> CStreamingStringFeatures<T>::get_vector()
{
	current_sgstring.string=current_string;
	current_sgstring.slen=current_length;

	return current_sgstring;
}

template <class T>
float64_t CStreamingStringFeatures<T>::get_label()
{
	ASSERT(has_labels)

	return current_label;
}

template <class T>
void CStreamingStringFeatures<T>::release_example()
{
	parser.finalize_example();
}

template <class T>
int32_t CStreamingStringFeatures<T>::get_vector_length()
{
	return current_length;
}

template <class T>
EFeatureClass CStreamingStringFeatures<T>::get_feature_class() const
{
	return C_STREAMING_STRING;
}

template class CStreamingStringFeatures<bool>;
template class CStreamingStringFeatures<char>;
template class CStreamingStringFeatures<int8_t>;
template class CStreamingStringFeatures<uint8_t>;
template class CStreamingStringFeatures<int16_t>;
template class CStreamingStringFeatures<uint16_t>;
template class CStreamingStringFeatures<int32_t>;
template class CStreamingStringFeatures<uint32_t>;
template class CStreamingStringFeatures<int64_t>;
template class CStreamingStringFeatures<uint64_t>;
template class CStreamingStringFeatures<float32_t>;
template class CStreamingStringFeatures<float64_t>;
template class CStreamingStringFeatures<floatmax_t>;

}
