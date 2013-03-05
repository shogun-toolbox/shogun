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

/** Class to generate dense features data via the streaming features interface.
 * The core are pairs of methods to
 * a) set the data model and parameters, and
 * b) to generate a data vector using these model parameters
 * Both methods are automatically called when calling get_next_example()
 * This allows to treat generated data as a stream via the standard streaming
 * features interface.
 *
 * Streaming based data generator that samples from a distribution that
 * is a 2D grid-like mixture of a number Gaussians at a certain distance from each
 * other, and where each Gaussian is stretched and rotated by a factor.
 */
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
		return "GaussianBlobsDataGenerator";
	}

	/** set the blobs model
	 *
	 * @param sqrt_num_blobs number of blobs per row/column in the grid
	 * @param distance distance of the individual Gaussians
	 * @param stretch first Eigenvalue of Gaussian will be set to this value.
	 * This effectively stretches the Gaussian
	 * @param angle Gaussians are rotated by this angle
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
	/** number of blobs per row/column in the grid */
	index_t m_sqrt_num_blobs;

	/** distance of the individual Gaussians */
	float64_t m_distance;

	/** first Eigenvalue of Gaussian will be set to this value */
	float64_t m_stretch;

	/** Gaussians are rotated by this angle */
	float64_t m_angle;

	/** Cholesky factor of covariance matrix of single Gaussians. Stored to
	 * increase sampling performance */
	SGMatrix<float64_t> m_cholesky;
};

}

#endif /* __GAUSSIANBLOBSDATAGENERATOR_H_ */
