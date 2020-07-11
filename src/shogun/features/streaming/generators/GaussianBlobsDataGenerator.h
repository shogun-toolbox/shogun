/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Yuyu Zhang
 */

#ifndef __GAUSSIANBLOBSDATAGENERATOR_H_
#define __GAUSSIANBLOBSDATAGENERATOR_H_

#include <shogun/lib/config.h>

#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/mathematics/RandomMixin.h>

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
class GaussianBlobsDataGenerator: public RandomMixin<StreamingDenseFeatures<float64_t>>
{
public:
	/** Constructor */
	GaussianBlobsDataGenerator();

	/** Constructor
	 */
	GaussianBlobsDataGenerator(index_t sqrt_num_blobs, float64_t distance,
			float64_t stretch, float64_t angle);

	/** Destructor */
	~GaussianBlobsDataGenerator() override;

	/** @return name of SG_SERIALIZABLE */
	const char* get_name() const override
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
	bool get_next_example() override;

	/** release the example when done w/ processing */
	void release_example() override;

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
