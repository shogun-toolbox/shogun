/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __MEANSHIFTDATAGENERATOR_H_
#define __MEANSHIFTDATAGENERATOR_H_

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>

namespace shogun
{


/** Class to generate dense features data via the streaming features interface.
 * The core are pairs of methods to
 * a) set the data model and parameters, and
 * b) to generate a data vector using these model parameters
 * Both methods are automatically called when calling get_next_example()
 * This allows to treat generated data as a stream via the standard streaming
 * features interface.
 *
 * Streaming based data generator that samples from a distribution that is a
 * multivariate isotropic Gaussian in a certain dimensions, where one selected
 * dimension has its mean shifted by some value.
 */
class CMeanShiftDataGenerator: public CStreamingDenseFeatures<float64_t>
{
public:
	/** Constructor */
	CMeanShiftDataGenerator();

	/** Constructor
	 *
	 * @param mean_shift Selected dimension is shifted by this amount
	 * @param dimension Number of dimensions
	 * @param m_dimension_shift Dimension that gets shifted
	 */
	CMeanShiftDataGenerator(float64_t mean_shift, index_t dimension,
			index_t m_dimension_shift=0);

	/** Destructor */
	virtual ~CMeanShiftDataGenerator();

	/** @return name of SG_SERIALIZABLE */
	virtual const char* get_name() const
	{
		return "MeanShiftDataGenerator";
	}

	/** Set the mean shift model
	 *
	 * @param mean_shift Selected dimension is shifted by this amount
	 * @param dimension Number of dimensions
	 * @param m_dimension_shift Dimension that gets shifted
	 */
	void set_mean_shift_model(float64_t mean_shift, index_t dimension,
			index_t m_dimension_shift=0);

	/** get the next example from stream */
	bool get_next_example();

	/** release the example when done w/ processing */
	void release_example();

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

protected:
	/** model of data to generate */
	float64_t m_mean_shift;

	/** dimension */
	index_t m_dimension;

	/** Dimension that is shifted */
	index_t m_dimension_shift;
};

}

#endif /* __MEANSHIFTDATAGENERATOR_H_ */
