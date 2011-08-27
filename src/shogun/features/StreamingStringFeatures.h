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

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/DataType.h>
#include <shogun/io/InputParser.h>

#include <shogun/features/StreamingFeatures.h>
#include <shogun/features/Alphabet.h>

namespace shogun
{
/** @brief This class implements streaming features as strings.
 *
 */
template <class T> class CStreamingStringFeatures : public CStreamingFeatures
{
public:

	/** 
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CStreamingFile::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	CStreamingStringFeatures()
		: CStreamingFeatures()
	{
		init();
		set_read_functions();
		remap_to_bin=false;
	}

	/** 
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 * 
	 * @param file StreamingFile object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	CStreamingStringFeatures(CStreamingFile* file,
				 bool is_labelled,
				 int32_t size)
		: CStreamingFeatures()
	{
		init(file, is_labelled, size);
		set_read_functions();
		remap_to_bin=false;
	}

	/** 
	 * Destructor.
	 * 
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	virtual ~CStreamingStringFeatures()
	{
		parser.end_parser();
		SG_UNREF(alphabet);
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

	/** 
	 * Set the alphabet to be used.
	 * Call before parsing.
	 * 
	 * @param alpha alphabet as an EAlphabet enum.
	 */
	void use_alphabet(EAlphabet alpha)
	{
		SG_UNREF(alphabet);

		alphabet=new CAlphabet(alpha);
		SG_REF(alphabet);
		num_symbols=alphabet->get_num_symbols();
	}

	/** 
	 * Set the alphabet to be used.
	 * Call before parsing.
	 * 
	 * @param alpha alphabet as a pointer to a CAlphabet object.
	 */
	void use_alphabet(CAlphabet* alpha)
	{
		SG_UNREF(alphabet);

		alphabet=new CAlphabet(alpha);
		SG_REF(alphabet);
		num_symbols=alphabet->get_num_symbols();
	}

	/** 
	 * Set whether remapping to another alphabet is required.
	 *
	 * Call before parsing.
	 * @param ascii_alphabet the alphabet to convert from, CAlphabet*
	 * @param binary_alphabet the alphabet to convert to, CAlphabet*
	 */
	void set_remap(CAlphabet* ascii_alphabet, CAlphabet* binary_alphabet)
	{
		remap_to_bin=true;
		alpha_ascii=new CAlphabet(ascii_alphabet);
		alpha_bin=new CAlphabet(binary_alphabet);
	}

	/** 
	 * Set whether remapping to another alphabet is required.
	 *
	 * Call before parsing.
	 * @param ascii_alphabet the alphabet to convert from, EAlphabet
	 * @param binary_alphabet the alphabet to convert to, EAlphabet
	 */
	void set_remap(EAlphabet ascii_alphabet=DNA, EAlphabet binary_alphabet=RAWDNA)
	{
		remap_to_bin=true;
		alpha_ascii=new CAlphabet(ascii_alphabet);
		alpha_bin=new CAlphabet(binary_alphabet);
	}

	/** 
	 * Return the alphabet being used as a CAlphabet*
	 * @return 
	 */
	CAlphabet* get_alphabet()
	{
		SG_REF(alphabet);
		return alphabet;
	}
	
	/** get number of symbols
	 *
	 * Note: floatmax_t sounds weird, but LONG is not long enough
	 *
	 * @return number of symbols
	 */
	floatmax_t get_num_symbols()
	{
		return num_symbols;
	}

	/** 
	 * Starts the parsing thread.
	 *
	 * To be called before trying to use any feature vectors from this object.
	 */
	virtual void start_parser();

	/** 
	 * Ends the parsing thread.
	 *
	 * Waits for the thread to join.
	 */
	virtual void end_parser();

	/** 
	 * Instructs the parser to return the next example.
	 * 
	 * This example is stored as the current_example in this object.
	 * 
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	virtual bool get_next_example();

	/** 
	 * Return the current feature vector as an SGString<T>.
	 * 
	 * @return The vector as SGString<T>
	 */
	SGString<T> get_vector();

	/** 
	 * Return the label of the current example as a float.
	 * 
	 * Examples must be labelled, otherwise an error occurs.
	 * 
	 * @return The label as a float64_t.
	 */
	virtual float64_t get_label();

	/** 
	 * Release the current example, indicating to the parser that
	 * it has been processed by the learning algorithm.
	 *
	 * The parser is then free to throw away that example.
	 */
	virtual void release_example();

	/** 
	 * Return the length of the current vector.
	 * 
	 * @return current vector length as int32_t
	 */
	virtual int32_t get_vector_length();

	/** 
	 * Return the feature type, depending on T.
	 * 
	 * @return Feature type as EFeatureType
	 */
	virtual inline EFeatureType get_feature_type();

	/** 
	 * Return the feature class
	 * 
	 * @return C_STREAMING_STRING
	 */
	virtual EFeatureClass get_feature_class();

	/** 
	 * Duplicate the object.
	 * 
	 * @return a duplicate object as CFeatures*
	 */
	virtual CFeatures* duplicate() const
	{
		return new CStreamingStringFeatures<T>(*this);
	}

	/** 
	 * Return the name.
	 * 
	 * @return StreamingSparseFeatures
	 */
	inline virtual const char* get_name() const { return "StreamingStringFeatures"; }

	/** 
	 * Return the number of vectors stored in this object.
	 * 
	 * @return 1 if current_vector exists, else 0.
	 */
	inline virtual int32_t get_num_vectors() const
	{
		if (current_string)
			return 1;
		return 0;
	}

	/** 
	 * Return the size of one T object.
	 * 
	 * @return Size of T.
	 */
	virtual int32_t get_size() { return sizeof(T); }

	/** 
	 * Return the number of features in the current vector.
	 * 
	 * @return length of the vector
	 */
	virtual int32_t get_num_features() { return current_length; }

private:

	/** 
	 * Initializes members to null values.
	 * current_length is set to -1.
	 */
	void init();
	
	/** 
	 * Calls init, and also initializes the parser with the given args.
	 * 
	 * @param file StreamingFile to read from
	 * @param is_labelled whether labelled or not
	 * @param size number of examples in the parser's ring
	 */
	void init(CStreamingFile *file, bool is_labelled, int32_t size);

protected:

	/// The parser object, which reads from input and returns parsed example objects.
	CInputParser<T> parser;

	/// Alphabet to use
	CAlphabet* alphabet;

	/// If remapping is enabled, this is the source alphabet
	CAlphabet* alpha_ascii;

	/// If remapping is enabled, this is the target alphabet
	CAlphabet* alpha_bin;

	/// The StreamingFile object to read from.
	CStreamingFile* working_file;

	/// The current example's string as an SGString<T>
	SGString<T> current_sgstring;

	/// The current example's string as a T*
	T* current_string;

	/// The length of the current string
	int32_t current_length;

	/// The label of the current example, if applicable
	float64_t current_label;

	/// Whether examples are labelled or not
	bool has_labels;

	/// Whether remapping must be done
	bool remap_to_bin;

	/// Number of symbols
	int32_t num_symbols;
};

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
	current_sgstring.slen=current_length;
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
	current_sgstring.slen=current_length;

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
