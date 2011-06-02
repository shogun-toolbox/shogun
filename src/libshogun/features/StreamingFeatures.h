/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 1999-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STREAMINGFEATURES__H__
#define _STREAMINGFEATURES__H__

#include "lib/common.h"
#include "features/Features.h"
#include "lib/File.h"
#include "lib/parser.h"
#include <pthread.h>

namespace shogun
{
/** @brief The class StreamingFeatures is the base class of 'streaming' feature objects.
 *  
 * Currently, this class implements 'simple' streaming features.
 * i.e., With feature vectors which are floats.
 * Also, assumes file input.
 */
	
	class CStreamingFeatures: public CFeatures
	{
		/** 
		 * Initialize members.
		 * 
		 */
		void init(void);

	public:
		
		/** 
		 * Default constructor.
		 * NOT IMPLEMENTED!
		 */
		CStreamingFeatures();
		
		/** 
		 * Constructor, taking a StreamingFile as arg.
		 * @param file StreamingFile from which to load features
		 * @param is_labelled Whether features are labelled or not, optional
		 * @param size Buffer size in MB, used while parsing
		 */
		CStreamingFeatures(CStreamingFile* file, bool is_labelled, int32_t size);
		
		/** 
		 * Copy constructor
		 * @param orig Object to copy from
		 */
		CStreamingFeatures(const CStreamingFeatures & orig);

		/** 
		 * Initialize, given a CFile.
		 * NOT IMPLEMENTED!
		 * 
		 * @param loader CFile object pointer.
		 */
		CStreamingFeatures(CFile* loader);

		/** 
		 * Destructor
		 */
		virtual ~CStreamingFeatures();

		/** 
		 * Gets the name.
		 * @return StreamingFeatures
		 */
		inline virtual const char* get_name() const { return "StreamingFeatures"; }

		/** 
		 * Returns feature type.
		 * This object works on real valued features.
		 * 
		 * @return F_DREAL
		 */
		inline EFeatureType get_feature_type()
		{
			return F_DREAL;
		}

		/** 
		 * Returns feature class
		 *
		 * @return C_STREAMING_SIMPLE
		 */
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }

		/** 
		 * Gets number of vectors.
		 * Not useful for StreamingFeatures.
		 * 
		 * @return -1
		 */
		virtual inline int32_t	get_num_vectors() { return -1; }

		/** 
		 * Gets size of the object in the memory.
		 * 
		 * @return size of object.
		 */
		virtual int32_t get_size() { return sizeof(*this); }
		
		/** 
		 * Duplicate the object.
		 * NOT IMPLEMENTED!
		 * 
		 * @return NULL always
		 */
		virtual CFeatures* duplicate() const
		{
			return NULL;
		}

		/** 
		 * Starts the parser in a separate thread.
		 * 
		 */
		void start_parser();
		
		/** 
		 * End the parser, close the thread.
		 * 
		 */
		void end_parser();

		/** 
		 * Gets length of the current feature vector
		 * 
		 * @return length of current fv
		 */
		virtual int32_t get_dim_feature_space()
		{
			return current_length;
		}

		/** 
		 * Fetches the next feature vector, setting values by reference.
		 * Waits for the parser to return an example if necessary.
		 * Assumes labelled examples.
		 * 
		 * @param feature_vector Pointer to fv, passed by ref
		 * @param length Length of the fv, passed by ref
		 * @param label Label of example, passed by ref
		 * 
		 * @return 1 if successful, 0 if no examples left
		 */
		virtual int32_t get_next_feature_vector(float64_t* &feature_vector, int32_t &length, float64_t &label);

		/** 
		 * Fetches the next feature vector, setting values by reference.
		 * Waits for the parser to return an example if necessary.
		 * Assumes unlabelled examples.
		 * 
		 * @param feature_vector Pointer to fv, passed by ref
		 * @param length Length of the fv, passed by ref
		 * 
		 * @return 1 if successful, 0 if no examples left
		 */
		virtual int32_t get_next_feature_vector(float64_t* &feature_vector, int32_t &length);

		/** 
		 * Frees the current feature vector, makes the buffer space available
		 * for storing new objects.
		 */
		virtual void free_feature_vector();
		

	protected:
		
		CInputParser parser;	/**< Parser object, to parse input data */

		CStreamingFile* working_file;

		float64_t* current_feature_vector; /**< Feature vector last fetched */
		float64_t current_label;	/**< Label of last fetched example */
		int32_t current_length;	/**< Features in last fetched
								 * example */
		bool has_labels; /**< Whether the examples are
						  * labelled or not */

	};
}
#endif	/* _STREAMINGFEATURES__H__ */
