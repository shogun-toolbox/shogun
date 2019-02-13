/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Yuyu Zhang, Sergey Lisitsyn
 */
#ifndef __STREAMING_FILEFROMFEATURES_H__
#define __STREAMING_FILEFROMFEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/io/streaming/StreamingFile.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
class CFeatures;

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
 * For example, a derived class based on DenseFeatures should only
 * implement the get_(type)*_vector() functions, and a class based on
 * StringFeatures should only implement the get_(type)*_string()
 * functions.
 */
class SHOGUN_EXPORT CStreamingFileFromFeatures: public CStreamingFile
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
	 * Constructor taking a CFeatures object and labels as arguments
	 *
	 * @param feat features object
	 * @param lab labels as float64_t*
	 */
	CStreamingFileFromFeatures(CFeatures* feat, float64_t* lab);

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
		ASSERT(feat)
		features=feat;
	}

	/**
	 * Set the labels
	 *
	 * @param lab array of labels
	 */
	virtual void set_labels(float64_t* lab)
	{
		ASSERT(lab)
		labels=lab;
	}

	/** @return object name */
	virtual const char* get_name() const
	{

		return "StreamingFileFromFeatures";

	}

protected:

	/// Features object
	CFeatures* features;

	/// Labels (if applicable)
	float64_t* labels;
};
}
#endif //__STREAMING_FILEFROMFEATURES_H__
