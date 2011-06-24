/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _STREAMING_STRINGFEATURES__H__
#define _STREAMING_STRINGFEATURES__H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "base/Parameter.h"
#include "lib/DataType.h"
#include "lib/InputParser.h"

#include "features/StreamingFeatures.h"
#include "features/Alphabet.h"

namespace shogun
{
/** @brief This class implements streaming features as strings.
 *
 */
template <class T> class CStreamingStringFeatures : public CStreamingFeatures
{
public:

	CStreamingStringFeatures()
		: CStreamingFeatures()
	{
		init();
		set_read_functions();
		remap_to_bin=false;
	}

	CStreamingStringFeatures(CStreamingFile* file,
				 bool is_labelled,
				 int32_t size)
		: CStreamingFeatures()
	{
		init(file, is_labelled, size);
		set_read_functions();
		remap_to_bin=false;
	}
	
	~CStreamingStringFeatures()
	{
		parser.end_parser();
		delete[] current_string;
	}

	/** 
	 * Sets the read function (in case the examples are
	 * unlabelled) to get_*_vector() from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 * 
	 * The parser uses the function set by this while reading
	 * unlabelled examples.
	 */
	virtual void set_vector_reader();

	/** 
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 * 
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	virtual void set_vector_and_label_reader();

	void use_alphabet(EAlphabet alpha)
	{
		alphabet=new CAlphabet(alpha);
		SG_REF(alphabet);
		num_symbols=alphabet->get_num_symbols();
	}

	void use_alphabet(CAlphabet* alpha)
	{
		alphabet=new CAlphabet(alpha);
		SG_REF(alphabet);
		num_symbols=alphabet->get_num_symbols();
	}

	void set_remap(CAlphabet* ascii_alphabet, CAlphabet* binary_alphabet)
	{
		remap_to_bin=true;
		alpha_ascii=new CAlphabet(ascii_alphabet);
		alpha_bin=new CAlphabet(binary_alphabet);
	}

	void set_remap(EAlphabet ascii_alphabet=DNA, EAlphabet binary_alphabet=RAWDNA)
	{
		remap_to_bin=true;
		alpha_ascii=new CAlphabet(ascii_alphabet);
		alpha_bin=new CAlphabet(binary_alphabet);
	}
	
	CAlphabet* get_alphabet()
	{
		SG_REF(alphabet);
		return alphabet;
	}

	floatmax_t get_num_symbols()
	{
		return num_symbols;
	}

	virtual void start_parser();

	virtual void end_parser();

	virtual bool get_next_example();

	SGString<T> get_vector();

	virtual float64_t get_label();

	virtual void release_example();

	virtual int32_t get_vector_length();

	virtual inline EFeatureType get_feature_type();
	
	virtual EFeatureClass get_feature_class();

	virtual CFeatures* duplicate() const
	{
		return new CStreamingStringFeatures<T>(*this);
	}

	inline virtual const char* get_name() const { return "StreamingStringFeatures"; }

	inline virtual int32_t get_num_vectors()
	{
		if (current_string)
			return 1;
		return 0;
	}

	virtual int32_t get_size() { return sizeof(T); }

	virtual int32_t get_num_features() { return current_length; }

private:
	void init();
	
	void init(CStreamingFile *file, bool is_labelled, int32_t size);

protected:
		
	CInputParser<T> parser;

	CAlphabet* alphabet;

	CAlphabet* alpha_ascii;
	
	CAlphabet* alpha_bin;
	
	CStreamingFile* working_file;
	
	SGString<T> current_sgstring;
	
	T* current_string;
	
	int32_t current_length;

	float64_t current_label;

	bool has_labels;

	bool remap_to_bin;

	int32_t num_symbols;
};

#define SET_VECTOR_READER(sg_type, sg_function)				\
template <> void CStreamingStringFeatures<sg_type>::set_vector_reader() \
{									\
	parser.set_read_vector(&CStreamingFile::sg_function);		\
}

SET_VECTOR_READER(bool, get_bool_string);
SET_VECTOR_READER(char, get_char_string);
SET_VECTOR_READER(int8_t, get_int8_string);
SET_VECTOR_READER(uint8_t, get_byte_string);
SET_VECTOR_READER(int16_t, get_short_string);
SET_VECTOR_READER(uint16_t, get_word_string);
SET_VECTOR_READER(int32_t, get_int_string);
SET_VECTOR_READER(uint32_t, get_uint_string);
SET_VECTOR_READER(int64_t, get_long_string);
SET_VECTOR_READER(uint64_t, get_ulong_string);
SET_VECTOR_READER(float32_t, get_shortreal_string);
SET_VECTOR_READER(float64_t, get_real_string);
SET_VECTOR_READER(floatmax_t, get_longreal_string);
	
#undef SET_VECTOR_READER

#define SET_VECTOR_AND_LABEL_READER(sg_type, sg_function)		\
template <> void CStreamingStringFeatures<sg_type>::set_vector_and_label_reader() \
{									\
	parser.set_read_vector_and_label(&CStreamingFile::sg_function); \
}

SET_VECTOR_AND_LABEL_READER(bool, get_bool_string_and_label);
SET_VECTOR_AND_LABEL_READER(char, get_char_string_and_label);
SET_VECTOR_AND_LABEL_READER(int8_t, get_int8_string_and_label);
SET_VECTOR_AND_LABEL_READER(uint8_t, get_byte_string_and_label);
SET_VECTOR_AND_LABEL_READER(int16_t, get_short_string_and_label);
SET_VECTOR_AND_LABEL_READER(uint16_t, get_word_string_and_label);
SET_VECTOR_AND_LABEL_READER(int32_t, get_int_string_and_label);
SET_VECTOR_AND_LABEL_READER(uint32_t, get_uint_string_and_label);
SET_VECTOR_AND_LABEL_READER(int64_t, get_long_string_and_label);
SET_VECTOR_AND_LABEL_READER(uint64_t, get_ulong_string_and_label);
SET_VECTOR_AND_LABEL_READER(float32_t, get_shortreal_string_and_label);
SET_VECTOR_AND_LABEL_READER(float64_t, get_real_string_and_label);
SET_VECTOR_AND_LABEL_READER(floatmax_t, get_longreal_string_and_label);
	
#undef SET_VECTOR_AND_LABEL_READER		

	
#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> inline EFeatureType CStreamingStringFeatures<sg_type>::get_feature_type() \
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
	current_sgstring.length=current_length;
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
		SG_ERROR("StreamingStringFeatures: The given input was found to be incompatible with the alphabet!\n");
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
	current_sgstring.length=current_length;

	return current_sgstring;
}

template <class T>
float64_t CStreamingStringFeatures<T>::get_label()
{
	ASSERT(has_labels);

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
EFeatureClass CStreamingStringFeatures<T>::get_feature_class()
{
	return C_STREAMING_STRING;
}

}
#endif // _STREAMING_STRINGFEATURES__H__
