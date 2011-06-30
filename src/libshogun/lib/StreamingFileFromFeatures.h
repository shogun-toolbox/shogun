/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILEFROMFEATURES_H__
#define __STREAMING_FILEFROMFEATURES_H__

#include "lib/StreamingFile.h"
#include "features/Features.h"

namespace shogun
{
/** @brief Class StreamingFileFromFeatures to read vector-by-vector
 * from a CFeatures object.
 *
 * The object must be initialized with another CFeatures object.
 * It is upto the derived class to implement specialized functions
 * to return the vector.
 *
 * Only a subset of the functions defined in StreamingFile.h need
 * to be implemented, as appropriate for the CFeatures object
 * which the class works with.
 *
 * For example, a derived class based on SimpleFeatures should only
 * implement the get_(type)*_vector() functions, and a class based on
 * StringFeatures should only implement the get_(type)*_string()
 * functions.
 */
class CStreamingFileFromFeatures: public CStreamingFile
{
public:
	/** 
	 * Default constructor
	 */
	CStreamingFileFromFeatures();

	/** 
	 * Constructor taking a CFeatures object as argument
	 * 
	 * @param feat features object
	 */
	CStreamingFileFromFeatures(CFeatures* feat);

	/** 
	 * Destructor
	 */
	virtual ~CStreamingFileFromFeatures();

	/** 
	 * Set the features object
	 * 
	 * @param feat features object as argument
	 */
	virtual void set_features(CFeatures* feat)
	{
		features=feat;
	}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "StreamingFileFromFeatures";

	}

protected:

	/// Features object
	CFeatures* features;
};
}
#endif //__STREAMING_FILEFROMFEATURES_H__
