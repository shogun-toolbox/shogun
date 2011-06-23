/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _STREAMING_SIMPLEFEATURES__H__
#define _STREAMING_SIMPLEFEATURES__H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "features/StreamingDotFeatures.h"
#include "lib/DataType.h"
#include "lib/InputParser.h"

namespace shogun
{
/** @brief This class implements streaming features with dense feature vectors.
 *
 * The current example is stored as a combination of current_vector
 * and current_label.
 */
template <class T> class CStreamingSimpleFeatures : public CStreamingDotFeatures
{
public:

	/** 
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CStreamingFile::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	CStreamingSimpleFeatures()
	{
		set_vector_reader();
		set_vector_and_label_reader();
		init();
	}

	/** 
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 * 
	 * @param file StreamingFile object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	CStreamingSimpleFeatures(CStreamingFile* file,
				 bool is_labelled,
				 int32_t size)
	{
		set_vector_reader();
		set_vector_and_label_reader();
		init(file, is_labelled, size);
	}

	/** 
	 * Destructor.
	 * 
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	~CStreamingSimpleFeatures()
	{
		parser.end_parser();
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
	void set_vector_reader();

	/** 
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 * 
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	void set_vector_and_label_reader();

	/** 
	 * Starts the parsing thread.
	 *
	 * To be called before trying to use any feature vectors from this object.
	 */
	void start_parser();

	/** 
	 * Ends the parsing thread.
	 *
	 * Waits for the thread to join.
	 */
	void end_parser();

	/** 
	 * Instructs the parser to return the next example.
	 * 
	 * This example is stored as the current_example in this object.
	 * 
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	bool get_next_example();

	/** 
	 * Return the current feature vector as an SGVector<T>.
	 * 
	 * @return The vector as SGVector<T>
	 */
	SGVector<T> get_vector();

	/** 
	 * Return the label of the current example as a float.
	 * 
	 * Examples must be labelled, otherwise an error occurs.
	 * 
	 * @return The label as a float64_t.
	 */
	float64_t get_label();

	/** 
	 * Release the current example, indicating to the parser that
	 * it has been processed by the learning algorithm.
	 *
	 * The parser is then free to throw away that example.
	 */
	void release_example();

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space();

	/** 
	 * Dot product using the current vector and another vector, passed as arg.
	 * 
	 * @param vec The vector with which to calculate the dot product.
	 * 
	 * @return Dot product as a float64_t
	 */
	virtual float64_t dot(SGVector<T> &vec);

	/** 
	 * Dot product taken with another StreamingDotFeatures object.
	 *
	 * Currently only works if it is a CStreamingSimpleFeatures object.
	 * It takes the dot product of the current_vectors of both objects.
	 * 
	 * @param df CStreamingDotFeatures object.
	 * 
	 * @return Dot product.
	 */
	virtual float64_t dot(CStreamingDotFeatures *df);

	/** 
	 * Dot product with another dense vector.
	 * 
	 * @param sgvec1 The dense vector with which to take the dot product.
	 * 
	 * @return Dot product as a float64_t.
	 */
	virtual float64_t dense_dot(SGVector<float64_t> &sgvec1)
	{
		int32_t len1=sgvec1.vlen;
		
		ASSERT(len1==current_length);
		float64_t result=0;
		
		for (int32_t i=0; i<current_length; i++)
			result+=current_vector[i]*sgvec1.vector[i];
		
		return result;
	}

	/** 
	 * Add alpha*current_vector to another dense vector.
	 * Takes the absolute value of current_vector if specified.
	 * 
	 * @param alpha alpha
	 * @param vec vector to add to
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float64_t alpha, SGVector<float64_t> &vec, bool abs_val=false)
	{
		ASSERT(vec.vlen==current_length);
		
		if (abs_val)
		{
			for (int32_t i=0; i<current_length; i++)
				vec.vector[i]+=alpha*CMath::abs(current_vector[i]);
		}
		else
		{
			for (int32_t i=0; i<current_length; i++)
				vec.vector[i]+=alpha*current_vector[i];
		}
	}

	/** 
	 * Return the number of features in the current example.
	 * 
	 * @return number of features as int
	 */
	int32_t get_num_features();
	
	/** 
	 * Return the feature type, depending on T.
	 * 
	 * @return Feature type as EFeatureType
	 */
	virtual inline EFeatureType get_feature_type();

	/** 
	 * Return the feature class
	 * 
	 * @return C_STREAMING_SIMPLE
	 */
	EFeatureClass get_feature_class();

	/** 
	 * Duplicate the object.
	 * 
	 * @return a duplicate object as CFeatures*
	 */
	virtual CFeatures* duplicate() const
	{
		return new CStreamingSimpleFeatures<T>(*this);
	}

	/** 
	 * Return the name.
	 * 
	 * @return StreamingSimpleFeatures
	 */
	inline virtual const char* get_name() const { return "StreamingSimpleFeatures"; }

	/** 
	 * Return the number of vectors stored in this object.
	 * 
	 * @return 1 if current_vector exists, else 0.
	 */
	inline virtual int32_t get_num_vectors()
	{
		if (current_vector)
			return 1;
		return 0;
	}

	/** 
	 * Return the size of one T object.
	 * 
	 * @return Size of T.
	 */
	virtual int32_t get_size() { return sizeof(T); }

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
		
	/// feature weighting in combined dot features
	float64_t combined_weight;

	/// The parser object, which reads from input and returns parsed example objects.
	CInputParser<T> parser;

	/// The StreamingFile object to read from.
	CStreamingFile* working_file;

	/// The current example's feature vector as an SGVector<T>
	SGVector<T> current_sgvector;

	/// The current example's feature vector as a T*.
	T* current_vector;

	/// The current example's label.
	float64_t current_label;

	/// Number of features in current example.
	int32_t current_length;

	/// Whether examples are labelled or not.
	bool has_labels;
};

#define SET_VECTOR_READER(sg_type, sg_function)				\
template <> void CStreamingSimpleFeatures<sg_type>::set_vector_reader() \
{									\
	parser.set_read_vector(&CStreamingFile::sg_function);		\
}

SET_VECTOR_READER(bool, get_bool_vector);
SET_VECTOR_READER(char, get_char_vector);
SET_VECTOR_READER(int8_t, get_int8_vector);
SET_VECTOR_READER(uint8_t, get_byte_vector);
SET_VECTOR_READER(int16_t, get_short_vector);
SET_VECTOR_READER(uint16_t, get_word_vector);
SET_VECTOR_READER(int32_t, get_int_vector);
SET_VECTOR_READER(uint32_t, get_uint_vector);
SET_VECTOR_READER(int64_t, get_long_vector);
SET_VECTOR_READER(uint64_t, get_ulong_vector);
SET_VECTOR_READER(float32_t, get_shortreal_vector);
SET_VECTOR_READER(float64_t, get_real_vector);
SET_VECTOR_READER(floatmax_t, get_longreal_vector);
	
#undef SET_VECTOR_READER

#define SET_VECTOR_AND_LABEL_READER(sg_type, sg_function)		\
template <> void CStreamingSimpleFeatures<sg_type>::set_vector_and_label_reader() \
{									\
	parser.set_read_vector_and_label(&CStreamingFile::sg_function); \
}

SET_VECTOR_AND_LABEL_READER(bool, get_bool_vector_and_label);
SET_VECTOR_AND_LABEL_READER(char, get_char_vector_and_label);
SET_VECTOR_AND_LABEL_READER(int8_t, get_int8_vector_and_label);
SET_VECTOR_AND_LABEL_READER(uint8_t, get_byte_vector_and_label);
SET_VECTOR_AND_LABEL_READER(int16_t, get_short_vector_and_label);
SET_VECTOR_AND_LABEL_READER(uint16_t, get_word_vector_and_label);
SET_VECTOR_AND_LABEL_READER(int32_t, get_int_vector_and_label);
SET_VECTOR_AND_LABEL_READER(uint32_t, get_uint_vector_and_label);
SET_VECTOR_AND_LABEL_READER(int64_t, get_long_vector_and_label);
SET_VECTOR_AND_LABEL_READER(uint64_t, get_ulong_vector_and_label);
SET_VECTOR_AND_LABEL_READER(float32_t, get_shortreal_vector_and_label);
SET_VECTOR_AND_LABEL_READER(float64_t, get_real_vector_and_label);
SET_VECTOR_AND_LABEL_READER(floatmax_t, get_longreal_vector_and_label);
	
#undef SET_VECTOR_AND_LABEL_READER		
	
#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> inline EFeatureType CStreamingSimpleFeatures<sg_type>::get_feature_type() \
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
int32_t CStreamingSimpleFeatures<T>::get_dim_feature_space()
{
	return current_length;
}

template <class T>
	float64_t CStreamingSimpleFeatures<T>::dot(CStreamingDotFeatures* df)
{
	SG_NOTIMPLEMENTED;
	return -1;
}

template <class T>
float64_t CStreamingSimpleFeatures<T>::dot(SGVector<T> &sgvec1)
{
	int32_t len1;
	len1=sgvec1.vlen;
				
	if (len1 != current_length)
		SG_ERROR("Lengths %d and %d not equal while computing dot product!\n", len1, current_length);

	float64_t result=CMath::dot(current_vector, sgvec1.vector, len1);
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

}
#endif // _STREAMING_SIMPLEFEATURES__H__
