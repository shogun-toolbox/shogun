/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifndef __GAUSSIANBLOBSDATAGENERATOR_H_
#define __GAUSSIANBLOBSDATAGENERATOR_H_

#include <shogun/features/streaming/StreamingDenseFeatures.h>

namespace shogun
{

class CGaussianBlobsDataGenerator: public CStreamingDenseFeatures<float64_t>
{
public:
	/** Constructor */
	CGaussianBlobsDataGenerator();

	/** Constructor
	 */
	CGaussianBlobsDataGenerator(index_t sqrt_num_blobs, float64_t distance,
			float64_t stretch, float64_t angle);

	/** Destructor */
	virtual ~CGaussianBlobsDataGenerator();

	/** @return name of SG_SERIALIZABLE */
	virtual const char* get_name() const
	{
		return "MeanShiftDataGenerator";
	}

	/*
	 * set the blobs model
	 *
	 */
	void set_blobs_model(index_t sqrt_num_blobs, float64_t distance,
			float64_t stretch, float64_t angle);

	/** get the next example from stream */
	bool get_next_example();

	/** release the example when done w/ processing */
	void release_example();

private:
	/** registers all parameters and initializes variables with defaults */
	void init();

protected:
	index_t m_sqrt_num_blobs;
	float64_t m_distance;
	float64_t m_stretch;
	float64_t m_angle;
	SGMatrix<float64_t> m_cholesky;
};

}

#endif /* __GAUSSIANBLOBSDATAGENERATOR_H_ */
