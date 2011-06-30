/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _STREAMING_SPARSEFEATURES__H__
#define _STREAMING_SPARSEFEATURES__H__

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
template <class T> class CStreamingSparseFeatures : public CStreamingDotFeatures
{
public:

	/** 
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CFeatureStream::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	CStreamingSparseFeatures()
		: CStreamingDotFeatures()
	{
		set_read_functions();
		init();
	}

	/** 
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 * 
	 * @param file FeatureStream object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	CStreamingSparseFeatures(CFeatureStream* file,
				 bool is_labelled,
				 int32_t size)
		: CStreamingDotFeatures()
	{
		set_read_functions();
		init(file, is_labelled, size);
	}

	/** 
	 * Destructor.
	 * 
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	~CStreamingSparseFeatures()
	{
		parser.end_parser();
	}

	/** 
	 * Sets the read function (in case the examples are
	 * unlabelled) to get_*_vector() from CFeatureStream.
	 *
	 * The exact function depends on type T.
	 * 
	 * The parser uses the function set by this while reading
	 * unlabelled examples.
	 */
	virtual void set_vector_reader();

	/** 
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CFeatureStream.
	 *
	 * The exact function depends on type T.
	 * 
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	virtual void set_vector_and_label_reader();

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

	/** get a single feature
	 *
	 * @param index index of feature in this vector
	 *
	 * @return sum of features that match dimension index and 0 if none is found
	 */
	T get_feature(int32_t index)
	{
		ASSERT(index>=0 && index<current_num_features);

		T ret=0;
		
		if (current_vector)
		{
			for (int32_t i=0; i<current_num_features; i++)
				if (current_vector[i].feat_index==index)
					ret += current_vector[i].entry;
		}

		return ret;
	}

	/** 
	 * Return the current feature vector as an SGSparseVector<T>.
	 * 
	 * @return The vector as SGSparseVector<T>
	 */
	SGSparseVector<T> get_vector();

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
	
	/** set number of features
	 *
	 * Sometimes when loading sparse features not all possible dimensions
	 * are used. This may pose a problem to classifiers when being applied
	 * to higher dimensional test-data. This function allows to
	 * artificially explode the feature space
	 *
	 * @param num the number of features, must be larger
	 *        than the current number of features
	 * @return previous number of features
	 */
	inline int32_t set_num_features(int32_t num)
	{
		int32_t n=current_num_features;
		ASSERT(n<=num);
		current_num_features=num;
		return n;
	}
	
	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space();

	/** 
	 * Dot product taken with another StreamingDotFeatures object.
	 *
	 * Currently only works if it is a CStreamingSparseFeatures object.
	 * It takes the dot product of the current_vectors of both objects.
	 * 
	 * @param df CStreamingDotFeatures object.
	 * 
	 * @return Dot product.
	 */
	virtual float64_t dot(CStreamingDotFeatures *df);
	
	/** compute the dot product between two sparse feature vectors
	 * alpha * vec^T * vec
	 *
	 * @param alpha scalar to multiply with
	 * @param avec first sparse feature vector
	 * @param alen avec's length
	 * @param bvec second sparse feature vector
	 * @param blen bvec's length
	 * @return dot product between the two sparse feature vectors
	 */
	static T sparse_dot(T alpha, SGSparseVectorEntry<T>* avec, int32_t alen, SGSparseVectorEntry<T>* bvec, int32_t blen)
	{
		T result=0;

		//result remains zero when one of the vectors is non existent
		if (avec && bvec)
		{
			if (alen<=blen)
			{
				int32_t j=0;
				for (int32_t i=0; i<alen; i++)
				{
					int32_t a_feat_idx=avec[i].feat_index;

					while ( (j<blen) && (bvec[j].feat_index < a_feat_idx) )
						j++;

					if ( (j<blen) && (bvec[j].feat_index == a_feat_idx) )
					{
						result+= avec[i].entry * bvec[j].entry;
						j++;
					}
				}
			}
			else
			{
				int32_t j=0;
				for (int32_t i=0; i<blen; i++)
				{
					int32_t b_feat_idx=bvec[i].feat_index;

					while ( (j<alen) && (avec[j].feat_index < b_feat_idx) )
						j++;

					if ( (j<alen) && (avec[j].feat_index == b_feat_idx) )
					{
						result+= bvec[i].entry * avec[j].entry;
						j++;
					}
				}
			}

			result*=alpha;
		}

		return result;
	}
	
	/** compute the dot product between dense weights and a sparse feature vector
	 * alpha * sparse^T * w + b
	 *
	 * @param alpha scalar to multiply with
	 * @param vec dense vector to compute dot product with
	 * @param dim length of the dense vector
	 * @param b bias
	 * @return dot product between dense weights and a sparse feature vector
	 */
	T dense_dot(T alpha, T* vec, int32_t dim, T b)
	{
		ASSERT(vec);
		ASSERT(dim==current_num_features);
		T result=b;

		int32_t num_feat=current_length;
		SGSparseVectorEntry<T>* sv=current_vector;

		if (sv)
		{
			for (int32_t i=0; i<num_feat; i++)
				result+=alpha*vec[sv[i].feat_index]*sv[i].entry;
		}

		return result;
	}

	/**
	 * Dot product with another dense vector.
	 * 
	 * @param sgvec2 The dense vector with which to take the dot product.
	 * 
	 * @return Dot product as a float64_t.
	 */
	virtual float64_t dense_dot(SGVector<float64_t> &sgvec2)
	{
		float64_t* vec2=sgvec2.vector;
		int32_t vec2_len=sgvec2.vlen;
		
		ASSERT(vec2);
		if (vec2_len!=current_num_features)
		{
			SG_ERROR("dimension of vec2 (=%d) does not match number of features (=%d)\n",
				 vec2_len, current_num_features);
		}
		
		float64_t result=0;
		if (current_vector)
		{
			for (int32_t i=0; i<current_length; i++)
				result+=vec2[current_vector[i].feat_index]*current_vector[i].entry;
		}

		return result;
	}

	/** 
	 * Add alpha*current_vector to another dense vector.
	 * Takes the absolute value of current_vector if specified.
	 * 
	 * @param alpha alpha
	 * @param sgvec2 vector to add to
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float64_t alpha, SGVector<float64_t> &sgvec2, bool abs_val=false)
	{
		float64_t* vec2=sgvec2.vector;
		int32_t vec2_len=sgvec2.vlen;
		
		ASSERT(vec2);
		if (vec2_len!=current_num_features)
		{
			SG_ERROR("dimension of vec (=%d) does not match number of features (=%d)\n",
				 vec2_len, current_num_features);
		}

		SGSparseVectorEntry<T>* sv=current_vector;
		int32_t num_feat=current_length;
		
		if (sv)
		{
			if (abs_val)
			{
				for (int32_t i=0; i<num_feat; i++)
					vec2[sv[i].feat_index]+= alpha*CMath::abs(sv[i].entry);
			}
			else
			{
				for (int32_t i=0; i<num_feat; i++)
					vec2[sv[i].feat_index]+= alpha*sv[i].entry;
			}
		}
	}


	/** 
	 * Get number of non-zero entries in current sparse vector
	 * 
	 * @return number of features explicity set in the sparse vector
	 */
	int64_t get_num_nonzero_entries()
	{
		return current_length;
	}

	/** 
	 * Compute sum of squares of features on current vector.
	 * 
	 * @return sum of squares for current vector
	 */
	float64_t compute_squared()
	{
		ASSERT(current_vector);

		float64_t sq=0;
		
		for (int32_t i=0; i<current_length; i++)
			sq += current_vector[i].entry * current_vector[i].entry;

		return sq;
	}

	/** 
	 * Ensure features of the current vector are in ascending order.
	 */
	void sort_features()
	{
		ASSERT(current_vector);

		SGSparseVectorEntry<T>* sf_orig=current_vector;
		int32_t len=current_length;

		int32_t* feat_idx=new int32_t[len];
		int32_t* orig_idx=new int32_t[len];
		
		for (int32_t i=0; i<len; i++)
		{
			feat_idx[i]=sf_orig[i].feat_index;
			orig_idx[i]=i;
		}

		CMath::qsort_index(feat_idx, orig_idx, len);

		SGSparseVectorEntry<T>* sf_new=new SGSparseVectorEntry<T>[len];

		for (int32_t i=0; i<len; i++)
			sf_new[i]=sf_orig[orig_idx[i]];

		current_vector=sf_new;

		// sanity check
		for (int32_t i=0; i<len-1; i++)
			ASSERT(sf_new[i].feat_index<sf_new[i+1].feat_index);

		delete[] orig_idx;
		delete[] feat_idx;
		delete[] sf_orig;
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
	 * @return C_STREAMING_SPARSE
	 */
	virtual EFeatureClass get_feature_class();

	/** 
	 * Duplicate the object.
	 * 
	 * @return a duplicate object as CFeatures*
	 */
	virtual CFeatures* duplicate() const
	{
		return new CStreamingSparseFeatures<T>(*this);
	}

	/** 
	 * Return the name.
	 * 
	 * @return StreamingSparseFeatures
	 */
	inline virtual const char* get_name() const { return "StreamingSparseFeatures"; }

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
	 * @param file FeatureStream to read from
	 * @param is_labelled whether labelled or not
	 * @param size number of examples in the parser's ring
	 */
	void init(CFeatureStream *file, bool is_labelled, int32_t size);

protected:
		
	/// feature weighting in combined dot features
	float64_t combined_weight;

	/// The parser object, which reads from input and returns parsed example objects.
	CInputParser< SGSparseVectorEntry<T> > parser;

	/// The FeatureStream object to read from.
	CFeatureStream* working_file;

	/// The current example's feature vector as an SGVector<T>
	SGSparseVector<T> current_sgvector;

	/// The current example's feature vector as an SGSparseVectorEntry<T>*.
	SGSparseVectorEntry<T>* current_vector;

	/// The current vector index
	index_t current_vec_index;
	
	/// The current example's label.
	float64_t current_label;

	/// Number of set indices in current example.
	int32_t current_length;

	/// Number of features in current vector (as seen so far upto the current vector)
	int32_t current_num_features;

	/// Whether examples are labelled or not.
	bool has_labels;
};

#define SET_VECTOR_READER(sg_type, sg_function)				\
template <> void CStreamingSparseFeatures<sg_type>::set_vector_reader() \
{									\
	parser.set_read_vector(&CFeatureStream::sg_function);		\
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
template <> void CStreamingSparseFeatures<sg_type>::set_vector_and_label_reader() \
{									\
	parser.set_read_vector_and_label(&CFeatureStream::sg_function); \
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
template<> inline EFeatureType CStreamingSparseFeatures<sg_type>::get_feature_type() \
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
void CStreamingSparseFeatures<T>::init()
{
	working_file=NULL;
	current_vector=NULL;
	current_length=-1;
	current_vec_index=0;
}

template <class T>
void CStreamingSparseFeatures<T>::init(CFeatureStream* file,
				    bool is_labelled,
				    int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
}
	
template <class T>
void CStreamingSparseFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class T>
void CStreamingSparseFeatures<T>::end_parser()
{
	parser.end_parser();
}

template <class T>
bool CStreamingSparseFeatures<T>::get_next_example()
{
	bool ret_value;
	ret_value = (bool) parser.get_next_example(current_vector,
						   current_length,
						   current_label);


	current_vec_index++;
	return ret_value;
}

template <class T>
SGSparseVector<T> CStreamingSparseFeatures<T>::get_vector()
{
	current_sgvector.features=current_vector;
	current_sgvector.num_feat_entries=current_length;
	current_sgvector.vec_index=current_vec_index;
	
	return current_sgvector;
}

template <class T>
float64_t CStreamingSparseFeatures<T>::get_label()
{
	ASSERT(has_labels);

	return current_label;
}
	
template <class T>
void CStreamingSparseFeatures<T>::release_example()
{
	parser.finalize_example();
}

template <class T>
int32_t CStreamingSparseFeatures<T>::get_dim_feature_space()
{
	return current_num_features;
}

template <class T>
	float64_t CStreamingSparseFeatures<T>::dot(CStreamingDotFeatures* df)
{
	SG_NOTIMPLEMENTED;
	return -1;
}

template <class T>
int32_t CStreamingSparseFeatures<T>::get_num_features()
{
	return current_num_features;
}

template <class T>
EFeatureClass CStreamingSparseFeatures<T>::get_feature_class()
{
	return C_STREAMING_SPARSE;
}

}
#endif // _STREAMING_SPARSEFEATURES__H__
