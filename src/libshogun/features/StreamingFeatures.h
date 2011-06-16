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
#include "lib/InputParser.h"
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
		CStreamingFeatures(CStreamingFile* file, bool is_labelled=true, int32_t size=10);
		
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
		 * Frees the current example, makes the buffer space available
		 * for storing new objects.
		 */
		virtual void release_example();

		/** 
		 * Gets the next available example object from the parser.
		 * Stores the vector, length and label in the StreamingFeatures object.
		 *
		 * The example must be accessed using get_vector() and
		 * get_label(), after a call to fetch_example().
		 * 
		 * @return 1 if successful, 0 if no examples are left.
		 */
		virtual int32_t fetch_example();

		/** 
		 * Returns the feature vector of the example obtained through fetch_example().
		 * 
		 * @return Vector of type SGVector<float64_t>.
		 */
		virtual SGVector<float64_t> get_vector();

		/** 
		 * Returns the label of the example obtained through fetch_example().
		 * Will raise an SG_ERROR() if examples are specified to be unlabelled.
		 * 
		 * @return Label of type float64_t.
		 */
		virtual float64_t get_label();

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
